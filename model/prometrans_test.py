#!/usr/bin/env python3

from prometrans_dataset import CallTransTestDataset
from naive_calltrans_model import NaiveCallTrans
from evaluate_util import calculate_recall_mrr

import argparse
import typing
import torch
import json
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

label_index = {
    "memory management module" : 1,
    "string manipulation module" : 2,
    "networking module" : 3,
    "data format module" : 4,
    "file i/o module" : 5,
    "logging module" : 6,
    "cryptographic module" : 7,
    "time and date module" : 8,
    "user interface module" : 9,
    "math module" : 10,
    "system call wrapper module" : 11,
    "authentication module" : 12,
    "compression module" : 13,
    "sort module" : 14,
    "sorting module" : 14, 
    "multithreading and synchronization module" : 15,
    "system resource module" : 16,
    "parsing module" : 17,
    "communication module" : 18, 
    "error handling module" : 19,
    "database module" : 20,

    "audio and video processing module" : 21,
    "input and output validation module" : 22,
    "hash module" : 23,
    "hashing module" :23,
    "gui module" : 25,
}

@torch.no_grad()
def test(model: NaiveCallTrans, args: argparse.Namespace):
    model.eval()
    label = {v: k for k, v in label_index.items()}


    test_dataset = CallTransTestDataset(
        db_path=args.test_db_path,
        function_name_cache_path=args.function_name_embedding_cache_path,
        contains_external=args.contains_external,
        num_classes = args.num_classes
        )
    
    dataloader = test_dataset.dataloader(max_num_nodes=args.max_num_nodes,
                                         shuffle=False,
                                         skip_too_big=True,
                                         num_workers=0,
                                         pin_memory=True)
    

    model.config_device(
        bert_device_id=args.graph_device,
        graph_device_id=args.graph_device,
        model_device_ids=[int(x) for x in args.model_devices.split(",")]
    )

    train_preds = []
    train_labels = []
    
    results = {}

    with torch.no_grad():
        for target_graph, labels, _, external, prj_name, func_name in tqdm(dataloader):
            try:      
                _, output = model(target_graph, labels, external)
                _, preds = torch.max(output, dim=1)
                sorted_predictions, indices = torch.sort(output, descending=True)
                train_preds.extend(preds.tolist())
                train_labels.extend(labels)

                for (sort, pred, proj, func) in zip(indices.tolist(), sorted_predictions.tolist(), prj_name, func_name):
                    output = []
                    for i in range(5):
                        output.append(label[sort[i]])
                        output.append(pred[i])
                    results[proj + "@" + func] = output

                target_graph.cpu()

            except Exception as e:
                if "CUDA out of memory" in str(e):
                    print("CudaOutMemory")
                    continue


        acc = accuracy_score(train_labels, train_preds)
        precision = precision_score(train_labels, train_preds, average='macro')
        recall = recall_score(train_labels, train_preds, average='macro')
        f1 = f1_score(train_labels, train_preds, average='macro')
    
    
    with open(f"results.json","w",encoding='utf8') as fp:
        json.dump(results,fp,indent=2)

    print(acc, precision, recall, f1)
    return acc, precision, recall, f1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_db_path", type=str, default="/path/to/testdata")
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument(
        "--function_name_embedding_cache_path", type=str, default=None)
    
    parser.add_argument("--max_num_nodes", type=int, default=1500)
    device_count = torch.cuda.device_count()
    parser.add_argument("--graph_device", type=int, 
                        default=1 if device_count > 1 else 0)
    parser.add_argument("--model_devices", type=str,
                        default=','.join([str(i) for i in range(device_count)]))
    parser.add_argument("--num_classes", type=int, default=20)       # POJ : 104
    parser.add_argument("--contains_external", type=bool, default=True)
    parser.add_argument("--bert_device", type=int, default=device_count - 1)
    args = parser.parse_args()
    
    
    model = NaiveCallTrans.from_saved(args.model_path, args)
    test(model, args)