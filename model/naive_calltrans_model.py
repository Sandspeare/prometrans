import torch
import os
import math
import json 

from torch.nn import Parameter
from typing import Union, Any, Callable, Dict, Optional, List 
from transformers import BertTokenizer, BertModel
from model import JtransModel

import dgl

from prometrans_dataset import CallTransTrainDataset
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class BertEmbedding(nn.Module):
    def __init__(self, bert):
        super(BertEmbedding, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
    

    def forward(self, tokens, segments):

        _, pooled_output = self.bert(tokens, token_type_ids=segments, return_dict=False)
        x = self.dropout(pooled_output)
        return x

    def freeze_layers(self, finetune_layer_cnt=None, freeze_layer_cnt=None):
        if finetune_layer_cnt is not None and freeze_layer_cnt is not None:
            if finetune_layer_cnt + freeze_layer_cnt != 12:
                raise ValueError("finetune_layer_cnt + freeze_layer_cnt must be 12")
            layer_cnt = finetune_layer_cnt
        elif finetune_layer_cnt is not None:
            layer_cnt = finetune_layer_cnt
        elif freeze_layer_cnt is not None:
            layer_cnt = 12 - freeze_layer_cnt
        else:
            raise ValueError("finetune_layer_cnt or freeze_layer_cnt must be set")
        
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        for layer in self.bert.encoder.layer[:-layer_cnt]:
            for param in layer.parameters():
                param.requires_grad = False
        
        for layer in self.bert.encoder.layer[-layer_cnt:]:
            for param in layer.parameters():
                param.requires_grad = True

class NaiveCallTrans(torch.nn.Module):
    def __init__(self, 
                 bert_model_path: str,
                 model_path: str,
                 finetune_layer_cnt: int = 2,
                 layer_cnt: int = 2,
                 contains_external: bool = False,
                 hidden_dim: int = 256,
                 num_classes: int = 30):
        super().__init__()
        self.finetune_layer_cnt = finetune_layer_cnt
        self.layer_cnt = layer_cnt // 2
        self.contains_external = contains_external
        

        self.classifier = Classifier(768, hidden_dim, num_classes + 1)
        self.criterion = nn.CrossEntropyLoss()

        self.jtrans = JtransModel.from_pretrained(model_path)
        self.jtrans.freeze_layers(finetune_layer_cnt=self.finetune_layer_cnt)
        

        self.add_module('classifier', self.classifier)
        
        self.conv_list = torch.nn.ModuleList()
        
        for _ in range(layer_cnt):
            convs = {
                ("function", "calls", "function"):
                    dgl.nn.pytorch.GATv2Conv(
                        in_feats=768,
                        out_feats=768 // 8,
                        num_heads=8,
                        allow_zero_in_degree=True
                    ),
           
                ("function", "called_by", "function"):
                    dgl.nn.pytorch.GATv2Conv(
                        in_feats=768,
                        out_feats=768 // 8,
                        num_heads=8,
                        allow_zero_in_degree=True
                    ),
            }
            if self.contains_external:
                convs.update({
                    ("function", "calls", "external_function"):
                        dgl.nn.pytorch.GATv2Conv(
                            in_feats=768,
                            out_feats=768 // 8,
                            num_heads=8,
                            # residual=True,
                            allow_zero_in_degree=True
                        ),
                    ("external_function", "called_by", "function"):
                        dgl.nn.pytorch.GATv2Conv(
                            in_feats=768,
                            out_feats=768 // 8,
                            num_heads=8,
                            # residual=True,
                            allow_zero_in_degree=True
                        ),
                })
            self.conv_list.append(dgl.nn.pytorch.HeteroGraphConv(convs))
    
    def config_device(self,
                      bert_device_id: int,
                      graph_device_id: int,
                      model_device_ids: list[int]):
        self.jtrans = torch.nn.DataParallel(
            module=self.jtrans.to(torch.device("cuda", model_device_ids[0])),
            device_ids = model_device_ids,
        )    

        graph_device = torch.device("cuda", graph_device_id)
        self.conv_list.to(graph_device)
        self.classifier.to(graph_device) 
    
    def forward(self, batched_graph: dgl.DGLHeteroGraph, labels: int, external: str):
        graph_device = next(self.conv_list.parameters()).device
        bert_device = next(self.classifier.parameters()).device
        batched_graph = batched_graph.to(graph_device)
        function_feats = self.jtrans(
            input_ids=batched_graph.nodes["function"].data["input_ids"],
            attention_mask=batched_graph.nodes["function"].data["attention_mask"]
        ).pooler_output.to(graph_device)
        
        if self.contains_external:
            external_function_feats = batched_graph.nodes[
                "external_function"].data[
                "embedding"].to(graph_device)
            node_feats_dict = {
                "function": function_feats,
                "external_function": external_function_feats
            }
        else:
            node_feats_dict = {
                "function": function_feats
            }

        for conv in self.conv_list:
            new_node_feats_dict = conv(batched_graph, node_feats_dict)
            node_feats_dict["function"] = new_node_feats_dict["function"].reshape(
                [new_node_feats_dict["function"].shape[0], -1]) + node_feats_dict["function"]
            if self.contains_external:
                node_feats_dict["external_function"] = \
                    new_node_feats_dict["external_function"].reshape(
                        [new_node_feats_dict["external_function"].shape[0], -1]) + \
                    node_feats_dict["external_function"]
        
        index_count = 0
        indices = []
        for num in batched_graph.batch_num_nodes("function"):
            indices.append(index_count)
            index_count += num.item()


        func_embed = node_feats_dict["function"][indices]
        x = func_embed

        output = self.classifier(x)
        loss = self.criterion(output, torch.tensor(labels).to(device=x.device.index))

        return loss, output
        
    
    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        config_save_path = os.path.join(path, "config.json")
        model_save_path = os.path.join(path, "jtrans")

        state_dict_path = os.path.join(path, "state_dict.pt")
        config = {
            "finetune_layer_cnt": self.finetune_layer_cnt,
            "layer_cnt": self.layer_cnt, 
            "contains_external": self.contains_external
        }
        with open(config_save_path, "w") as f:
            json.dump(config, f)
        if isinstance(self.jtrans, torch.nn.DataParallel):
            self.jtrans.module.save_pretrained(model_save_path)
        else:
            self.jtrans.save_pretrained(model_save_path)
        

        torch.save(self.state_dict(), state_dict_path)
    
    def from_saved(path, args):
        if not os.path.exists(path):
            raise Exception(f"Path {path} does not exist.")
        config_save_path = os.path.join(path, "config.json")
        model_save_path = os.path.join(path, "jtrans")
        state_dict_path = os.path.join(path, "state_dict.pt")
        with open(config_save_path, "r") as f:
            config = json.load(f)
        finetune_layer_cnt = config["jtrans_finetune_layer_cnt"]
        layer_cnt = config["layer_cnt"]
        contains_external = config["contains_external"]
        result = NaiveCallTrans(
            bert_model_path="",
            model_path=model_save_path,
            finetune_layer_cnt=finetune_layer_cnt,
            layer_cnt=layer_cnt,
            contains_external=contains_external,
            num_classes=args.num_classes)
        result.load_state_dict(torch.load(state_dict_path), strict=False)
        return result
                                
                