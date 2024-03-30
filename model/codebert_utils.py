#!/usr/bin/env python3
from sqlalchemy import create_engine
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from tqdm.auto import tqdm
import typing
import torch
import os
import io
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
import base64, zlib


def serialize_tensor(tensor: torch.Tensor) -> str:
    """
    Serialize a tensor to bytes
    """
    buffer = io.BytesIO()
    tensor = tensor.detach().cpu().clone()
    torch.save(tensor, buffer)
    res = buffer.getvalue()
    res = base64.b64encode(s=zlib.compress(res))
    return res

def deserialize_tensor(tensor_str: str) -> torch.Tensor:
    """
    Deserialize a tensor from bytes
    """
    tensor_bytes = zlib.decompress(base64.b64decode(s=tensor_str))
    buffer = io.BytesIO(tensor_bytes)
    tensor = torch.load(buffer)
    return tensor


def init_model() -> tuple[AutoTokenizer, AutoModel]:
    """
    Load the pretrained model and tokenizer
    """
    # microsoft/codebert-base
    # bert-base-uncased
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base",
                                      output_hidden_states=True)
    
    model = model.to(torch.device("cuda", 1))

    model.eval()
    return tokenizer, model


def hidden_states_handler(hidden_states):
    #! Deprecated
    """
    convert the hidden states to a tensor of shape (token_seq_length, 13, 768)
    """
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1, 0, 2)

    return token_embeddings


def get_sum_cls(token_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Sum the last four layers of the hidden states and return the first token.
    """
    # sum the last four layers
    # Stores the token vectors, with shape [token_seq_length x 768]
    token_vecs_sum = []

    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0) 
        token_vecs_sum.append(sum_vec)

    return token_vecs_sum[0]


def cos_distance(matrix: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(matrix, p=2, dim=1)
    matrix_dist = normalized.matmul(normalized.T)
    matrix_dist = matrix_dist.new_ones(matrix_dist.shape) - matrix_dist
    return matrix_dist


@torch.no_grad()
def calculate_function_name_embedding(func_name_list: list[str], 
                  batch_size: int = 256) -> torch.Tensor:
    """
    Get the embedding of function names.
    The tokenizer will tokenize the function name and add the special tokens, so the embedding
    of function name is actually the embedding of the `sentence`.
    """
    tokenizer, model = init_model()

    encoding_list = []
    if len(func_name_list) / batch_size > 100:
        iterator = tqdm(range(0, len(func_name_list), batch_size))
    else:
        iterator = range(0, len(func_name_list), batch_size)
    for i in iterator:
        inputs = tokenizer(func_name_list[i:i+batch_size],
                            padding=True,
                            return_tensors='pt',
                            truncation=True,)
        inputs = inputs.to(torch.device("cuda", 1))
        outputs = model(**inputs)[1]
        encoding_list.append(outputs)
    return torch.cat(encoding_list, dim=0)


class Base(DeclarativeBase):
    pass


class FunctionNameEmbedding(Base):
    __tablename__ = "function_name_embedding"
    function_name: Mapped[str] = mapped_column(primary_key=True)
    embedding: Mapped[str] = mapped_column()


class FunctionNameEmbeddingDB:
    def __init__(self, db_path: str, batch_size: int):
        self.db_path = os.path.abspath(db_path)
        self.engine = None
        self.batch_size = batch_size   
        self.establish_connection()  
        Base.metadata.create_all(self.engine) 

    def establish_connection(self):
        if self.engine is None:
            self.engine = create_engine(f"sqlite:///{self.db_path}")

    def close_connection(self):
        if self.engine is not None:
            self.engine.dispose()
            del self.engine
            self.engine = None

    def get_embedding(self, 
                      func_name_list: typing.List[str]) -> torch.Tensor:
        self.establish_connection()
        # deduplicate the function name list
        if len(func_name_list) == 0:
            return torch.empty((0, 768))
        deduped_func_name_list = list(set(func_name_list))
        with Session(self.engine) as session:
            results = session.query(FunctionNameEmbedding).filter(
                FunctionNameEmbedding.function_name.in_(deduped_func_name_list)).all()

            cached_function_embedding_dict = {
                result.function_name: deserialize_tensor(result.embedding)
                for result in results}
            uncached_function_name_list = [func_name
                                           for func_name in deduped_func_name_list
                                           if func_name
                                           not in
                                           cached_function_embedding_dict]
            if len(uncached_function_name_list) > 0:
                uncached_function_embedding_batch = calculate_function_name_embedding(
                    uncached_function_name_list, self.batch_size).detach().cpu()
                uncached_function_embedding_dict = {func_name: embedding for func_name, embedding in zip(
                    uncached_function_name_list, uncached_function_embedding_batch)}

                for func_name, embedding in uncached_function_embedding_dict.items():
                    session.merge(FunctionNameEmbedding(
                        function_name=func_name,
                        embedding=serialize_tensor(embedding)
                    ))
                session.commit()
                function_embedding_dict = {**cached_function_embedding_dict,
                                           **uncached_function_embedding_dict}
            else:
                function_embedding_dict = cached_function_embedding_dict
        return torch.stack([function_embedding_dict[func_name] for func_name in func_name_list])