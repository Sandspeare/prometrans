from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F

class JtransModel(BertModel):
    """
    jtrans model
    """
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings = self.embeddings.word_embeddings
    

    def freeze_all_layers(self):
        for param in self.embeddings.parameters():
            param.requires_grad = False
        
        for layer in self.encoder.layer:
            for param in layer.parameters():
                param.requires_grad = False

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

class PromeTransModel(nn.Module):
    """
    PromeTrans model
    """
    def __init__(self):
        super().__init__()