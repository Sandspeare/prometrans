#!/usr/bin/env python3
from sqlalchemy import create_engine
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

import os
import sys
import torch
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from codebert_utils import FunctionNameEmbedding, Base, serialize_tensor, deserialize_tensor
from transformers import AutoTokenizer, AutoModel
from dataset import ExternalCall
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--source_db_path", type=str, required=True)
parser.add_argument("--cache_db_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
args = parser.parse_args()

source_engine = create_engine(url=f"sqlite:///{args.source_db_path}")
cache_engine = create_engine(url=f"sqlite:///{args.cache_db_path}")
Base.metadata.create_all(cache_engine)
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base", output_hidden_states=True)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
with Session(source_engine) as source_session, Session(cache_engine) as cache_session:
    result = source_session.query(ExternalCall.callee).distinct().all()
    function_name_list = [res[0] for res in result]
    cached_result = cache_session.query(FunctionNameEmbedding.function_name).all()
    cached_function_name_list = [res[0] for res in cached_result]
    function_name_list = list(set(function_name_list) - set(cached_function_name_list))
    for i in tqdm(range(0, len(function_name_list), args.batch_size)):
        inputs = tokenizer(function_name_list[i:i+args.batch_size],
                           padding=True,
                           truncation=True,
                           return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)[1].cpu()
        for func_name, embedding in zip(function_name_list[i:i+args.batch_size], outputs):
            cache_session.merge(FunctionNameEmbedding(function_name=func_name,
                                                  embedding=serialize_tensor(tensor=embedding)))
        cache_session.commit()