#!/usr/bin/env python3
import multiprocessing
import argparse
import pickle
import random
import typing
from tqdm.auto import tqdm

import torch
import numpy as np


def calc_similarity_matrix(target_function_embeddings: torch.Tensor,
                      positive_function_embeddings: torch.Tensor):
    # (..., poolsize, embedding_dim)
    embedding_dim = target_function_embeddings.shape[-1]
    
    # Normalization (..., poolsize, embedding_dim)
    target_function_embeddings = (
        target_function_embeddings / torch.linalg.norm(target_function_embeddings, axis=-1)[..., None])
    positive_function_embeddings = (
        positive_function_embeddings / torch.linalg.norm(positive_function_embeddings, axis=-1)[..., None])

    # Calculate similarity (..., pool_size, pool_size)
    similarity_matrix = torch.matmul(
        target_function_embeddings, positive_function_embeddings.transpose(-1, -2))
    return similarity_matrix

def shuffle_and_group(target_function_embeddings: torch.Tensor, 
                      positive_function_embeddings: torch.Tensor,
                      poolsize: int):
    # (data_len, embedding_dim) to (data_len // poolsize, poolsize, embedding_dim)
    assert(target_function_embeddings.shape == positive_function_embeddings.shape)
    data_len = target_function_embeddings.shape[0]
    data_len = data_len - data_len % poolsize
    
    target_function_embeddings = target_function_embeddings[:data_len]
    positive_function_embeddings = positive_function_embeddings[:data_len]
    
    # Shuffle
    shuffle_index = list(range(data_len))
    random.shuffle(shuffle_index)
    target_function_embeddings = target_function_embeddings[shuffle_index]
    positive_function_embeddings = positive_function_embeddings[shuffle_index]
    
    embedding_dim = target_function_embeddings.shape[-1]
    # Group
    grouped_target_function_embeddings = target_function_embeddings.reshape(
        -1, poolsize, embedding_dim)
    grouped_positive_function_embeddings = positive_function_embeddings.reshape(
        -1, poolsize, embedding_dim)
    return grouped_target_function_embeddings, grouped_positive_function_embeddings

def calculate_recall_mrr(target_function_embeddings: torch.Tensor,
             positive_function_embeddings: torch.Tensor,
             poolsize: int):
    
    device = target_function_embeddings.device
    
    # (data_len, embedding_dim) to (data_len // poolsize, poolsize, embedding_dim)
    target_function_embeddings, positive_function_embeddings = shuffle_and_group(
        target_function_embeddings, positive_function_embeddings, poolsize)
    
    # Calculate similarity (group_num, pool_size, pool_size)
    similarity_matrix = calc_similarity_matrix(
        target_function_embeddings, positive_function_embeddings)
    
    # recall@1
    most_similar_value, most_similar_index = similarity_matrix.max(dim=-1)
    recall_1 = most_similar_index.eq(torch.arange(poolsize).to(device)).float().mean()
    
    # mrr
    sort_matrix = similarity_matrix.argsort(dim=-1, stable=True, descending=True)
    target_matrix = torch.arange(poolsize).to(device).repeat([poolsize, 1]).t().expand_as(sort_matrix)
    indice_matrix = (sort_matrix == target_matrix)
    mrr = (indice_matrix.nonzero()[:, -1] + 1).float().reciprocal().mean()
    
    return recall_1.item(), mrr.item()


def recall_1(target_function_embeddings: torch.Tensor,
             positive_function_embeddings: torch.Tensor,
             poolsize: int):
    # (datalen, embedding_dim)
    assert (target_function_embeddings.shape ==
            positive_function_embeddings.shape)
    data_len = target_function_embeddings.shape[0]
    embedding_dim = target_function_embeddings.shape[1]
    data_len = data_len - data_len % poolsize
    target_function_embeddings = target_function_embeddings[:data_len]
    positive_function_embeddings = positive_function_embeddings[:data_len]

    # DEBUG shuffling
    tmp = torch.cat((target_function_embeddings,
                    positive_function_embeddings), dim=0)
    shuffle_index = list(range(data_len))
    random.shuffle(shuffle_index)
    target_function_embeddings = target_function_embeddings[shuffle_index]
    positive_function_embeddings = positive_function_embeddings[shuffle_index]

    # Normalization (data_len, embedding_dim)
    target_function_embeddings = (
        target_function_embeddings.T / np.linalg.norm(target_function_embeddings, axis=-1)).T
    positive_function_embeddings = (
        positive_function_embeddings.T / np.linalg.norm(positive_function_embeddings, axis=-1)).T

    # grouped by pooling (group_num, pool_size, embedding_dim)
    pooled_target_function_embeddings = target_function_embeddings.reshape(
        -1, poolsize, embedding_dim)
    pooled_positive_function_embeddings = positive_function_embeddings.reshape(
        -1, poolsize, embedding_dim)

    # Calculate similarity (group_num, pool_size, pool_size)
    similarity_matrix = torch.matmul(
        pooled_target_function_embeddings, pooled_positive_function_embeddings.transpose(-1, -2))

    # (group_num, pool_size)
    most_similar_value, most_similar_index = similarity_matrix.max(dim=-1)

    return most_similar_index.eq(torch.arange(poolsize)).sum().item() / data_len
