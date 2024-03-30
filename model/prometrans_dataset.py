#!/usr/bin/env python3
from sqlalchemy import create_engine
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

import networkx as nx
import torch
import struct
import os
import json
import time
import zlib
import typing
import random
import base64
import torch_geometric
from collections.abc import Iterable
import warnings
from typing import Iterator, List, Optional, Tuple

from dataset import BinarySampleDataset, Function, ExternalCall
from jtrans_data import vectorize_token_sequence
from codebert_utils import FunctionNameEmbeddingDB
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import dgl

torch.multiprocessing.set_sharing_strategy('file_system')

"""
naivecalltrans dataset feeder

"""


def collate_dgl_graph(graph_sample_batch):
    """
    collate a batch of dgl graph
    """
    label = []
    tmp_buf = []
    index = []
    external = []
    proj = []
    func = []
    for graph_sample in graph_sample_batch:
        tmp_buf.append(graph_sample[0])
        label.append(graph_sample[1])
        index.append(graph_sample[2])
        external.append(graph_sample[3])
        proj.append(graph_sample[4])
        func.append(graph_sample[5])
    result = []
    result = dgl.batch(tmp_buf)
    return result, label, index, external, proj, func

class DynamicBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, 
                 dataset, 
                 max_num: int, 
                 shuffle: bool = False, 
                 skip_too_big: bool = False):
        self.dataset = dataset
        self.max_num = max_num
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        
        batch = [[]]
        batch_n = 0
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), dtype=torch.long)
        else:
            indices = torch.arange(len(self.dataset), dtype=torch.long)
        
        for idx in tqdm(indices, "Building Batch"):
            size = self.dataset.estimate_size(idx)  
            if batch_n + size > self.max_num:
                if batch_n == 0:
                    if self.skip_too_big:
                        continue
                    else:
                        warnings.warn("Size of data sample at index "
                                        f"{idx} is larger than {self.max_num}")
                else:
                    batch.append([])
                    batch_n = 0
            batch[-1].append(idx)
            batch_n += size
        self.batch = batch 
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batch)
        return iter(self.batch)

    def __len__(self):
        return len(self.batch)
        

class CallTransDatasetBase(BinarySampleDataset):
    def __init__(
            self, db_path: str, 
            num_classes : int = 5, 
            contains_caller: bool = False, layer_cnt: int = 2, 
            contains_external: bool = False,  
            function_name_cache_path: str = None, function_name_cache_batch: int = 64,
            optimization_levels: typing.List[str] = None): 
        self.contains_caller = contains_caller
        self.layer_cnt = layer_cnt  // 2
        self.contains_external = contains_external
        
        if contains_external:
            if function_name_cache_path is None:
                raise ValueError("function_name_cache_path must be provided "
                                 "when contains_external is True")
            self.function_name_embedding_db = FunctionNameEmbeddingDB(
                db_path, function_name_cache_batch)
        super(
            CallTransDatasetBase, self).__init__(  
            db_path=db_path, data_dir=None, num_classes=num_classes,
            optimization_levels=optimization_levels)

        size_key = f"{'all' if contains_caller else 'callee'}_graph_size" 
        with Session(self.engine) as session:
            res = session.query(Function.uuid, getattr(Function, size_key)).all()
            self.uuid_size_dict = {uuid: size for uuid, size in res} 
    
    def establish_connection(self):
        super(CallTransDatasetBase, self).establish_connection()
        if self.contains_external:
            self.function_name_embedding_db.establish_connection()
    
    def close_connection(self):
        super(CallTransDatasetBase, self).close_connection()
        if self.contains_external:
            self.function_name_embedding_db.close_connection()
    
    def convert_graph_to_data(self, uuid: str, 
                              graph: nx.MultiDiGraph) -> torch_geometric.data.HeteroData:
        self.establish_connection()
        node_list = list(graph.nodes)
        # move target node's uuid to the first place
        node_list.remove(uuid)
        node_list = [uuid] + node_list
        edge_index = []
        edge_attr = []
        #for caller_uuid, callee_uuid, call_index in graph.edges:
        for caller_uuid, callee_uuid, graph_edge_index in graph.edges:
            edge_index.append([node_list.index(caller_uuid), node_list.index(callee_uuid)])
            #edge_attr.append([call_index])
            edge_attr.append([graph[caller_uuid][callee_uuid][graph_edge_index]["index"]])
        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        
        x_list = []
        attention_list = []
        #for node in graph.nodes:
        for node in node_list:
            sequence_vector, attention_mask = vectorize_token_sequence(self.token_sequence(node))
            x_list.append(sequence_vector)
            attention_list.append(attention_mask)
            # Should be splited into two parts in model.forward
        x = torch.stack(x_list)
        attention = torch.stack(attention_list)
        if self.contains_external:

            external_calls = self.external_calls_from(node_list)
            external_function_list = list(set([_call[1] for _call in external_calls]))
            external_function_embeddings = self.function_name_embedding_db.get_embedding(
                func_name_list=external_function_list)
            if len(external_function_embeddings) == 0:
                # Add a dummy node to avoid error
                external_function_embeddings = torch.randn((1, 768))
            external_edge_index = []
            for caller_uuid, callee_name, call_index in external_calls:
                external_edge_index.append(
                    [node_list.index(caller_uuid), 
                     external_function_list.index(callee_name)])
            external_edge_attr = [[call_index] 
                                  for _, _, call_index in external_calls]
            if len(external_edge_index) == 0:
                external_edge_index = torch.zeros((2, 0), dtype=torch.long)
                external_edge_attr = torch.zeros((0, 1), dtype=torch.long)
            else:
                external_edge_index = torch.tensor(
                    external_edge_index, dtype=torch.long).t().contiguous()
                external_edge_attr = torch.tensor(
                    external_edge_attr, dtype=torch.long)
            dgl_data = dgl.heterograph(data_dict={
                ("function", "calls", "function"): (edge_index[0], edge_index[1]),
                ("function", "called_by", "function"): (edge_index[1], edge_index[0]),
                ("function", "calls", "external_function"): (external_edge_index[0], external_edge_index[1]),
                ("external_function", "called_by", "function"): (external_edge_index[1], external_edge_index[0])
            }, num_nodes_dict={
                "function": len(node_list),
                "external_function": len(external_function_list) if len(external_function_list) > 0 else 1
            })
            dgl_data.edges["function", "calls", "function"].data["callsite"] = edge_attr
            dgl_data.edges["function", "called_by", "function"].data["callsite"] = edge_attr
            dgl_data.edges["function", "calls", "external_function"].data["callsite"] = external_edge_attr
            dgl_data.edges["external_function", "called_by", "function"].data["callsite"] = external_edge_attr
            dgl_data.nodes["function"].data["input_ids"] = x
            dgl_data.nodes["function"].data["attention_mask"] = attention
            dgl_data.nodes["external_function"].data["embedding"] = external_function_embeddings
            
        else:
            dgl_data = dgl.heterograph(data_dict={
                ("function", "calls", "function"): (edge_index[0], edge_index[1]),
                ("function", "called_by", "function"): (edge_index[1], edge_index[0])
            }, num_nodes_dict={
                "function": len(node_list)
            })
            dgl_data.edges["function", "calls", "function"].data["callsite"] = edge_attr
            dgl_data.edges["function", "called_by", "function"].data["callsite"] = edge_attr
            dgl_data.nodes["function"].data["input_ids"] = x
            dgl_data.nodes["function"].data["attention_mask"] = attention

        
        return dgl_data
    
    def dataloader(self, 
                   max_num_nodes: int, 
                   shuffle: bool = False,
                   skip_too_big: bool = True,
                   **kwargs):
        self.close_connection()
        sampler = DynamicBatchSampler(dataset=self, 
                                      max_num=max_num_nodes, 
                                      shuffle=shuffle, 
                                      skip_too_big=skip_too_big) 
        return DataLoader(dataset=self, 
                          batch_sampler=sampler, 
                          collate_fn=collate_dgl_graph,
                          **kwargs)
    
    def __getitem__(self, idx: int):
        self.establish_connection()
        uuid = self.sample_uuid(idx, 1)[0]
        return self.sample_graph(uuid)


class CallTransTrainDataset(CallTransDatasetBase):
    def __init__(
            self, db_path: str, 
            function_name_cache_path: str = None, 
            contains_external: bool = True,
            num_classes: int = 5,
            layer_cnt: int = 2,
            contains_caller: bool = False, 
            function_name_cache_batch: int = 64,
            optimization_levels: typing.List[str] = None):
        super(
            CallTransTrainDataset, self).__init__(
            db_path=db_path, 
            num_classes = num_classes,
            contains_caller=contains_caller, layer_cnt=layer_cnt,      
            function_name_cache_path=function_name_cache_path,  
            function_name_cache_batch=function_name_cache_batch,   
            optimization_levels=optimization_levels, 
            contains_external=contains_external)    
    
    def contrastive_graphs(self, idx: int):
        self.establish_connection()

        target_uuid = self.function_list[idx]["uuid_dict"]["O0"]
        
        target_graph = self.load_graph(target_uuid, self.layer_cnt, self.contains_caller)
        #positive_graph = self.load_graph(positive_uuid, self.layer_cnt, self.contains_caller)
        #negative_graph = self.load_graph(negative_uuid, self.layer_cnt, self.contains_caller)
        
        target_data = self.convert_graph_to_data(target_uuid, target_graph)
        #positive_data = self.convert_graph_to_data(positive_uuid, positive_graph)
        #negative_data = self.convert_graph_to_data(negative_uuid, negative_graph)

        label = self.function_list[idx]["label"]
        external = self.function_list[idx]["external"]
        proj = self.function_list[idx]["project_name"]
        func = self.function_list[idx]["function_name"]
        return target_data, label, idx, external, proj, func

    
    def estimate_size(self, idx: int) -> int:
        self.establish_connection()

        target_uuid = self.function_list[idx]["uuid_dict"]["O0"]
        return self.uuid_size_dict[target_uuid]
    
    def __getitem__(self, idx: int):
        self.establish_connection()
        return self.contrastive_graphs(idx)


class CallTransTestDataset(CallTransDatasetBase):

    def __init__(
            self, db_path: str, 
            function_name_cache_path: str = None, 
            contains_external: bool = True,
            num_classes: int = 5,
            contains_caller: bool = False, 
            layer_cnt: int = 2,
            function_name_cache_batch: int = 64,
            optimization_levels: typing.List[str] = None):
        super(
            CallTransTestDataset, self).__init__(
            db_path=db_path, 
            num_classes=num_classes,
            function_name_cache_path=function_name_cache_path,
            function_name_cache_batch=function_name_cache_batch,
            optimization_levels=optimization_levels,
            contains_external=contains_external)

    def pair_graphs(self, idx: int):
        self.establish_connection()

        target_uuid = self.function_list[idx]["uuid_dict"]["O0"] 

        
        target_graph = self.load_graph(target_uuid, self.layer_cnt, self.contains_caller)

        
        target_data = self.convert_graph_to_data(target_uuid, target_graph)


        label = self.function_list[idx]["label"]
        external = self.function_list[idx]["external"]
        proj = self.function_list[idx]["project_name"]
        func = self.function_list[idx]["function_name"]
        return target_data, label, idx, external, proj, func   

    
    def estimate_size(self, idx: int) -> int:
        self.establish_connection()

        target_uuid = self.function_list[idx]["uuid_dict"]["O0"] 
        return self.uuid_size_dict[target_uuid]
    
    def __getitem__(self, idx: int):
        self.establish_connection()
        return self.pair_graphs(idx)