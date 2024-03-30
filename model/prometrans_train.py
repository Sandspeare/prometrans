#!/usr/bin/env python3

import argparse
import torch
from prometrans_dataset import CallTransTrainDataset, CallTransTestDataset
from naive_calltrans_model import NaiveCallTrans, Classifier
from model import Triplet_COS_Loss
from torch.utils.tensorboard import SummaryWriter
from evaluate_util import calculate_recall_mrr
from tqdm.auto import tqdm
import os
import pickle
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

torch.multiprocessing.set_sharing_strategy('file_system')

@torch.no_grad()
def test(model: NaiveCallTrans, args: argparse.Namespace):
    model.eval()
    test_dataset = CallTransTestDataset(
        db_path=args.test_db_path,
        function_name_cache_path=args.function_name_embedding_cache_path,
        contains_external=args.contains_external,
        num_classes = args.num_classes
        )
    

    dataloader = test_dataset.dataloader(max_num_nodes=args.test_max_num_nodes,
                                         shuffle=False,
                                         skip_too_big=True,
                                         num_workers=0,
                                         #prefetch_factor=8,
                                         pin_memory=True)
    
    train_preds = []
    train_labels = []

    with torch.no_grad():
        for target_graph, labels, _, external in tqdm(dataloader):
            try:              
                _, output = model(target_graph, labels, external)
                _, preds = torch.max(output, dim=1)
                train_preds.extend(preds.tolist())
                train_labels.extend(labels)
                target_graph.cpu()

            except Exception as e:
                if "CUDA out of memory" in str(e):
                    print("CudaOutMemory")
                    continue


        acc = accuracy_score(train_labels, train_preds)
        precision = precision_score(train_labels, train_preds, average='macro')
        recall = recall_score(train_labels, train_preds, average='macro')
        f1 = f1_score(train_labels, train_preds, average='macro')

    return acc, precision, recall, f1

def record_history(index, output, target, recorder):
    pred = output.cpu().data
    for i, ind in enumerate(index):
        recorder[ind.item()].append(pred[i][target[i]].numpy().tolist())
    return

def train(model: NaiveCallTrans, args: argparse.Namespace):
    train_dataset = CallTransTrainDataset(
        db_path=args.train_db_path, 
        function_name_cache_path=args.function_name_embedding_cache_path, 
        contains_external=args.contains_external,
        num_classes = args.num_classes, 
        layer_cnt = args.layer_cnt 
        ) 
    
    
    train_dataloader = train_dataset.dataloader(
        max_num_nodes=args.max_num_nodes,  
        shuffle=True, skip_too_big=True,
        num_workers=0,
        pin_memory=True)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters.extend(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-4,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    )
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate)
    model.config_device(
        bert_device_id=args.bert_device,
        graph_device_id=args.graph_device,
        model_device_ids=[int(x) for x in args.model_device.split(",")])

    if args.log_every > 0 and args.log_path is not None:
        import datetime
        tz = datetime.timezone(datetime.timedelta(hours=8))
        utc_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        time_str = utc_now.astimezone(tz).strftime("%H_%M_%S")
        tb_writer = SummaryWriter(os.path.join(args.log_path, 
                                               "naive_calltrans_train",
                                               f"layer{args.layer_cnt}_{time_str}"))
    
    global_step = 0
    fail_count = 0

    recorder = [[] for i in range(train_dataset.__len__())]

    for epoch in range(args.epoch):
        iterator = tqdm(train_dataloader)
        for target_graph, labels, index, external in iterator:
            model.train()
            optimizer.zero_grad()
            try:

                loss, output = model(target_graph, labels, external)
                record_history(index, output, labels, recorder)
             
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    fail_count += 1
                    continue
                else:
                    raise
            
            to_log = {
                "global_step": global_step,
                "epoch": epoch, 
                "loss": loss.item(),
                "fail_count": fail_count
            }
            
            wandb.log({'loss':loss.item()},step=global_step)

            target_graph.cpu()
            loss.cpu()
            
            iterator.set_description(f"{to_log}")
            
            if args.test_every > 0 and global_step % args.test_every == 0:
                acc, precision, recall, f1 = test(model, args)
                to_log["recall"] = recall
                to_log["f1"] = f1
                to_log["precision"] = precision
                to_log["acc"] = acc

                wandb.log({'recall':recall},step=global_step)
                wandb.log({'f1':f1},step=global_step)
                wandb.log({'precision':precision},step=global_step)
                wandb.log({'acc':acc},step=global_step)
                
            if args.save_every > 0 and \
                global_step % args.save_every == 0 and \
                args.save_path is not None:
                model.save(args.save_path)
            

            global_step += 1

            with open(f"optrans_{str(args.num_classes)}.pickle", 'wb') as recordfile:
                pickle.dump(recorder, recordfile)
            
    
    wandb.finish()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # About dataset
    parser.add_argument("--train_db_path", type=str, default="/path/to/traindata")
    parser.add_argument("--contains_external", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default="/path/to/base_model")
    parser.add_argument("--bert_model_path", type=str, default="/path/to/bert_model")   
    parser.add_argument("--function_name_embedding_cache_path",
                        type=str, default=None)  
    
    # About model
    parser.add_argument("--layer_cnt", type=int, default=2)  
    parser.add_argument("--max_num_nodes", type=int, default=1000) 

    # About train
    parser.add_argument("--finetune_layer_cnt", type=int, default=2) 
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    device_count = torch.cuda.device_count()
    parser.add_argument("--model_device", type=str, default=",".join(
        [str(idx) for idx in range(device_count)]))
    parser.add_argument("--graph_device", type=int, 
                        default=0)
    parser.add_argument("--bert_device", type=int, 
                        default=device_count - 1)
    parser.add_argument("--triplet_margin", type=float, default=0.2)
    
    # About log
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--log_path", type=str)
    
    # About save
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="/path/to/save") 
    parser.add_argument("--bert_finetune_path", type=str, default="/path/to/save") 
    # About test
    parser.add_argument("--test_every", type=int, default=100)      
    parser.add_argument("--test_db_path", type=str, default="/path/to/testdata") 
    parser.add_argument("--num_classes", type=int, default=20)       # POJ : 104
    
    args = parser.parse_args()
    

    wandb.init(config=args,
                project="PromeTrans",
                name=f"PromeTrans_{str(args.num_classes)}_layer",
                job_type="train",
                reinit=True)


    model = NaiveCallTrans(
        bert_model_path=args.bert_model_path,
        model_path=args.model_path,
        model_finetune_layer_cnt=args.model_finetune_layer_cnt,
        layer_cnt=args.layer_cnt, 
        contains_external=args.contains_external,
        num_classes=args.num_classes
        )
    
    train(model, args)