import sys
import networkx
import os
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import re
import pickle
import torch
import random
import time
from datasets import load_from_disk
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from torch.utils.data import DataLoader


MAXLEN=512


def gen_funcstr(f,convert_jump):
    cfg=f[3]
    #print(hex(f[0]))
    bb_ls,code_lst,map_id=[],[],{}
    for bb in cfg.nodes:
        bb_ls.append(bb)
    bb_ls.sort()
    for bx in range(len(bb_ls)):
        bb=bb_ls[bx]
        asm=cfg.nodes[bb]['asm']
        map_id[bb]=len(code_lst)
        for code in asm:
            operator,operand1,operand2,operand3,annotation=readidadata.parse_asm(code)
            code_lst.append(operator)
            if operand1!=None:
                code_lst.append(operand1)
            if operand2!=None:
                code_lst.append(operand2)
            if operand3!=None:
                code_lst.append(operand3)
    for c in range(len(code_lst)):
        op=code_lst[c]
        if op.startswith('hex_'):
            jumpaddr=int(op[4:],base=16)
            if map_id.get(jumpaddr):
                jumpid=map_id[jumpaddr]
                if jumpid < MAXLEN:
                    code_lst[c]='JUMP_ADDR_{}'.format(jumpid)
                else:
                    code_lst[c]='JUMP_ADDR_EXCEEDED'
            else:
                code_lst[c]='UNK_JUMP_ADDR'
            if not convert_jump:
                code_lst[c]='CONST'
    func_str=' '.join(code_lst)
    return func_str

def load_unpair_data(datapath,filt=None,alldata=True,convert_jump=True,opt=None, fp=None):
    dataset = DatasetBase(datapath,filt, alldata)
    dataset.load_unpair_data()
    functions=[]
    for i in dataset.get_unpaird_data():  #proj, func_name, func_addr, asm_list, rawbytes_list, cfg, bai_featrue
        f = (i[2], i[3], i[4], i[5], i[6])
        func_str=gen_funcstr(f,convert_jump)
        if len(func_str) > 0:
            fp.write(func_str+"\n")

def load_paired_data(datapath,filt=None,alldata=True,convert_jump=True,opt=None,add_ebd=False):
   
    dataset = DatasetBase(datapath,filt,alldata, opt=opt)
    functions=[]
    func_emb_data=[]
    SUM=0
    for i in dataset.get_paired_data_iter():  #proj, func_name, func_addr, asm_list, rawbytes_list, cfg, bai_featrue
        functions.append([])
        if add_ebd:
            func_emb_data.append({'proj':i[0],'funcname':i[1]})
        for o in opt:
            if i[2].get(o):                   
                f=i[2][o]
                func_str=gen_funcstr(f,convert_jump)
                if len(func_str)>0:
                    if add_ebd:
                        func_emb_data[-1][o]=len(functions[-1])
                    functions[-1].append(func_str)
                    SUM+=1

    print('TOTAL ',SUM)
    return functions,func_emb_data

class FunctionDataset_CL(torch.utils.data.Dataset): #binary version dataset
    def __init__(self,tokenizer,path='../BinaryCorp/extract',filt=None,alldata=True,convert_jump_addr=True,opt=None,add_ebd=True):  #random visit
        functions,ebds=load_paired_data(datapath=path,filt=filt,alldata=alldata,convert_jump=convert_jump_addr,opt=opt,add_ebd=add_ebd)
        self.datas=functions
        self.ebds=ebds
        self.tokenizer=tokenizer
        self.opt=opt
        self.convert_jump_addr=True
    def __getitem__(self, idx):             #also return bad pair

        pairs=self.datas[idx]
        if self.opt==None:
            pos=random.randint(0,len(pairs)-1)
            pos2=random.randint(0,len(pairs)-1)
            while pos2==pos:
                pos2=random.randint(0,len(pairs)-1)
            f1=pairs[pos]   #give three pairs
            f2=pairs[pos2]
        else:
            pos=0
            pos2=1
            f1=pairs[pos]
            f2=pairs[pos2]
        ftype=random.randint(0,len(self.datas)-1)
        while ftype==idx:
            ftype=random.randint(0,len(self.datas)-1)
        pair_opp=self.datas[ftype]
        pos3=random.randint(0,len(pair_opp)-1)
        f3=pair_opp[pos3]
        ret1 = help_tokenize(f1)
        token_seq1=ret1['input_ids']
        mask1=ret1['attention_mask']

        ret2 = help_tokenize(f2)
        token_seq2=ret2['input_ids']
        mask2=ret2['attention_mask']

        ret3 = help_tokenize(f3)
        token_seq3=ret3['input_ids']
        mask3=ret3['attention_mask']

        return token_seq1,token_seq2,token_seq3,mask1,mask2,mask3
    def __len__(self):
        return len(self.datas)

class FunctionDataset_CL_Load(torch.utils.data.Dataset): #binary version dataset
    def __init__(self,tokenizer,path='../BinaryCorp/extract',filt=None,alldata=True,convert_jump_addr=True,opt=None,add_ebd=True, load=None):  #random visit
        if load:
            start = time.time()
            self.datas = pickle.load(open(load, 'rb'))
            print('load time:', time.time() - start)
            self.tokenizer=tokenizer
            self.opt=opt
            self.convert_jump_addr=True
        else:
            functions,ebds=load_paired_data(datapath=path,filt=filt,alldata=alldata,convert_jump=convert_jump_addr,opt=opt,add_ebd=add_ebd)
            self.datas=[]
            for func_list in functions:
                tmp = []
                for f in func_list:
                    tmp.append(help_tokenize(f))
                self.datas.append(tmp)
            self.ebds=ebds
            self.tokenizer=tokenizer
            self.opt=opt
            self.convert_jump_addr=True
    def __getitem__(self, idx):             #also return bad pair

        pairs=self.datas[idx]
        if self.opt==None:
            pos=random.randint(0,len(pairs)-1)
            pos2=random.randint(0,len(pairs)-1)
            while pos2==pos:
                pos2=random.randint(0,len(pairs)-1)
            f1=pairs[pos]   #give three pairs
            f2=pairs[pos2]
        else:
            pos=0
            pos2=1
            f1=pairs[pos]
            f2=pairs[pos2]
        ftype=random.randint(0,len(self.datas)-1)
        while ftype==idx:
            ftype=random.randint(0,len(self.datas)-1)
        pair_opp=self.datas[ftype]
        pos3=random.randint(0,len(pair_opp)-1)
        f3=pair_opp[pos3]

        token_seq1=f1['input_ids']
        mask1=f1['attention_mask']

        token_seq2=f2['input_ids']
        mask2=f2['attention_mask']

        token_seq3=f3['input_ids']
        mask3=f3['attention_mask']

        return token_seq1,token_seq2,token_seq3,mask1,mask2,mask3
    def __len__(self):
        return len(self.datas)

def load_filter_list(name):
    import csv
    f=csv.reader(open(name,'r'))
    S=set()
    for i in f:
        S.add(i[1])
    return list(S)


class PromeTrainDataset(torch.utils.data.Dataset): #binary version dataset

    def __init__(self, path, options):  #random visit

        with open(path, "rb") as fp:
            dataset = pickle.load(fp)

        self.dataset = dataset
        self.options = options
        self.index = 0

    def collate_gat(batch):
        """
        collate a batch of graph
        """
        f1_embed = [item[0] for item in batch]
        f2_embed = [item[1] for item in batch]
        f3_embed = [item[2] for item in batch]
        f1_adjs = [item[3] for item in batch]
        f2_adjs = [item[4] for item in batch]
        f3_adjs = [item[5] for item in batch]

        f1_embeds = []
        f2_embeds = []
        f3_embeds = []

        return f1_embed, f2_embed, f3_embed, f1_adjs, f2_adjs, f3_adjs

    def __getitem__(self, idx):             #also return bad pair

        data = self.dataset[idx]
        keys = list(self.dataset[idx].keys())
        pos1 = random.randint(1, len(data) // 2) * 2 - 1
        pos2 = random.randint(1, len(data) // 2) * 2 - 1
        while pos2 == pos1:
            pos2 = random.randint(1, len(data) // 2) * 2 - 1
        f1_embed = data[keys[pos1]]   #give three pairs
        f2_embed = data[keys[pos2]]  
        f1_adjs = data[keys[pos1+1]]
        f2_adjs = data[keys[pos2+1]]
 
        ftype = random.randint(0,len(self.dataset)-1)
        while ftype == idx:
            ftype = random.randint(0,len(self.dataset)-1)
        pair_opp = self.dataset[ftype]
        neg = random.randint(1, len(pair_opp) // 2) * 2 - 1

        neg_keys = list(pair_opp.keys())
        f3_embed = pair_opp[neg_keys[neg]]
        f3_adjs = pair_opp[neg_keys[neg+1]]

        return f1_embed, f2_embed, f3_embed, f1_adjs, f2_adjs, f3_adjs
    
    def __len__(self):
        return len(self.dataset)



if __name__ == '__main__':

    path = "../create_dataset/datasets/jtrans_asm_dataset_demo.pkl"
    options = ["O0", "O1", "O2", "O3", "Os"]
    fp_train_data = PromeTrainDataset(path, options)

    train_dataloader = DataLoader(fp_train_data, batch_size=8, num_workers=0, shuffle=True, collate_fn=fp_train_data.collate_gat)   #, prefetch_factor=4)

    for f1_embed, f2_embed, f3_embed, f1_adjs, f2_adjs, f3_adjs in tqdm(train_dataloader):

        print(f1_embed, f2_embed, f3_embed, f1_adjs, f2_adjs, f3_adjs)