from transformers import BertModel
import torch
import torch.nn.functional as F

import wandb
import argparse
from pwn import log
from tqdm.auto import tqdm
import numpy as np

class Triplet_COS_Loss(torch.nn.Module):
    """
    loss function, can be grad to be trained
    """
    def __init__(self, margin):
        super(Triplet_COS_Loss, self).__init__()
        self.margin = margin

    def forward(self, target_result, positive_sample_result, negative_sample_result):
        positive_similarity = F.cosine_similarity(
            target_result, positive_sample_result)
        negative_similarity = F.cosine_similarity(
            target_result, negative_sample_result)
        loss = (self.margin - (positive_similarity -
                negative_similarity)).clamp(min=1e-6).mean()
        return loss

    
class JtransModel(BertModel):
    """
    jtrans model
    """
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings = self.embeddings.word_embeddings
    
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
        
        for param in self.embeddings.parameters():
            param.requires_grad = False
        
        for layer in self.encoder.layer[:-layer_cnt]:
            for param in layer.parameters():
                param.requires_grad = False
        
        for layer in self.encoder.layer[-layer_cnt:]:
            for param in layer.parameters():
                param.requires_grad = True
