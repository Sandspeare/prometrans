from unicodedata import name
from transformers import BertTokenizer, BertForMaskedLM, BertModel
import torch.multiprocessing
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from data import load_paired_data, PromeTrainDataset
from transformers import AdamW
import torch.nn.functional as F
import argparse
import wandb
import logging
import sys
import time
import data
import pickle
from datautils.BertMLP import BertMLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models import JtransModel, PromeTransModel

WANDB = True

def train_dp(model, args, train_set, valid_set):


    if WANDB:
        wandb.init(project=f'POJ-dataset', name=f"jtrans_{args.split}")
        wandb.config.update(args)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, shuffle=True)   #, prefetch_factor=4)
    valid_dataloader = DataLoader(valid_set, batch_size=args.eval_batch_size, num_workers=0, shuffle=True)  #, prefetch_factor=4)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    optimizer_grouped_parameters.extend(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    global_steps = 0

    patience = 5
    best_acc = 0
    counter  = 0
    for epoch in range(args.epoch):
        model.train()
        criterion = nn.CrossEntropyLoss()
        train_iterator = tqdm(train_dataloader)
        loss_list = []
        for i, (embeddings, label) in enumerate(train_iterator):

            optimizer.zero_grad()
            output = model(embeddings)
            loss = criterion(output, label)
            loss.backward()
            loss_list.append(loss)
            optimizer.step()

            if (global_steps) % args.eval_every == 0:
                acc, precision, recall, f1 = finetune_eval(model, valid_dataloader)

                # 判断是否收敛
                if acc > best_acc:
                    best_acc = acc
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("acc has converged. Training stopped.")
                        break

                if WANDB:
                    wandb.log({'recall':recall},step=global_steps)
                    wandb.log({'f1':f1},step=global_steps)
                    wandb.log({'precision':precision},step=global_steps)
                    wandb.log({'acc':acc},step=global_steps)

            if (global_steps) % args.save_every == 0:
                torch.save(model.state_dict(),os.path.join(args.output_path, f"finetune_{global_steps+1}" + ".pth"))

            if (i+1) % args.log_every == 0:
                global_steps += 1
                tmp_lr = optimizer.param_groups[0]["lr"]
                # logger.info(f"[*] epoch: [{epoch}/{args.epoch+1}], steps: [{i}/{len(train_iterator)}], lr={tmp_lr}, loss={loss}")
                train_iterator.set_description(f"[*] epoch: [{epoch}/{args.epoch+1}], steps: [{i}/{len(train_iterator)}], lr={tmp_lr}, loss={loss}")
                if WANDB:
                    wandb.log({'loss':loss},step=global_steps)
                    wandb.log({'lr':tmp_lr},step=global_steps)


def finetune_eval(net, data_loader):
    net.eval()

    train_preds = []
    train_labels = []

    with torch.no_grad():

        eval_iterator = tqdm(data_loader)
        for embeddings, label in eval_iterator:

            output = model(embeddings)
            _, preds = torch.max(output, dim=1)

            train_preds.extend(preds.tolist())
            train_labels.extend(label.tolist())

        acc = accuracy_score(train_labels, train_preds)
        precision = precision_score(train_labels, train_preds, average='macro')
        recall = recall_score(train_labels, train_preds, average='macro')
        f1 = f1_score(train_labels, train_preds, average='macro')

    return acc, precision, recall, f1


class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings=self.embeddings.word_embeddings

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description="PromeTrans Finetune")
    ## model path
    parser.add_argument("--tokenizer", type=str, default='./jtrans_tokenizer', help='the path of tokenizer')
    parser.add_argument("--jtrans_model", type=str, default='./models/jtrans/jTrans-pretrain',  help='the path of pretrain model')
    parser.add_argument("--output_path", type=str, default='./models/jtrans/jTrans-finetune', help='the path where the finetune model be saved')
    parser.add_argument("--load_path", type=str, default='./models/BinaryCorp-3M/', help='load path')

    ## train parameters
    parser.add_argument("--lr", type=float, default=1e-5, help='learning rate')
    parser.add_argument("--warmup", type=int, default=1000, help='warmup steps')
    parser.add_argument("--gamma", type=float, default=0.99, help='scheduler gamma')
    parser.add_argument("--local_rank", type=int, default = 0, help='local rank used for ddp')
    parser.add_argument("--weight_decay", type=int, default = 1e-4, help='regularization weight decay')
    parser.add_argument("--epoch", type=int, default=1000, help='number of training epochs')
    parser.add_argument("--step_size", type=int, default=40000, help='scheduler step size')

    ## dataloader parameters
    parser.add_argument("--batch_size", type=int, default = 256, help='training batch size')
    parser.add_argument("--eval_batch_size", type=int, default = 512, help='evaluation batch size')
    
    ## action steps
    parser.add_argument("--log_every", type=int, default =1, help='logging frequency')
    parser.add_argument("--eval_every", type=int, default=2, help="evaluate the model every x epochs")
    parser.add_argument("--eval_every_step", type=int, default=1000, help="evaluate the model every x epochs")
    parser.add_argument("--save_every", type=int, default=20, help="save the model every x epochs")

    ## data path
    parser.add_argument("--train_path", type=str, default='../create_dataset/datasets/jtrans_asm_dataset_demo.pkl',  help='the path of train dataset')
    parser.add_argument("--test_path", type=str, default='./data/binarycorp/test',  help='the path of test dataset')

    ## evaluation parameters 
    parser.add_argument("--options", nargs="+", default=['O0','O1','O2','O3','Os'], help='a list of compile options in datasets' )
    parser.add_argument("--class_num", type=int, default=104, help='the number of class')
    args = parser.parse_args([])

    # 选择第一块显卡
    torch.cuda.set_device(args.device)
    device = torch.device("cuda")

    ft_train_dataset= PromeTrainDataset(device, path=args.train_path)

    ft_train_dataset= FunctionDataset_POJ_Load(device, jtrans, path=args.train_path)


    

    prometrans = PromeTransModel()

    
    ft_valid_dataset=FunctionDataset_POJ_Load(device, jtrans, path=args.test_path)
    
    train_dp(model, args, ft_train_dataset, ft_valid_dataset)

