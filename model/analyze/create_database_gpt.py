#!/usr/bin/env python3

from sqlalchemy import create_engine
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from pwn import log

from tqdm.auto import tqdm
import networkx as nx
import multiprocessing
import argparse
import time
import json
import random
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import Base, Function, Call, ExternalCall, ExternalFunctions, serialize_graph

def parse_file_name(file_name: str):
    current_str = file_name
    last_seperator_index = current_str.rfind("-")
    assert(last_seperator_index != -1)
    md5sum = current_str[last_seperator_index + 1:]
    current_str = current_str[:last_seperator_index]
    last_seperator_index = current_str.rfind("-")
    assert(last_seperator_index != -1)
    optimization_level = current_str[last_seperator_index + 1:]
    project_name = current_str[:last_seperator_index]
    return project_name, optimization_level, md5sum

def build_graph(function_dict: dict) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for uuid in function_dict.keys():
        graph.add_node(uuid)
    for uuid in function_dict.keys():
        for callee_uuid in function_dict[uuid]["call_to"].keys():
            if function_dict[callee_uuid]["function_name"] == "split_function_control":
                call_site = 1
            elif function_dict[callee_uuid]["function_name"] == "split_function_block":
                call_site = 2
            else:
                call_site = 3
            graph.add_edge(uuid, callee_uuid, index=call_site)

    return graph

def get_neighborhood_subgraph(graph: nx.MultiDiGraph, uuid: str, 
                              depth: int = 3, max_sample: int = 5, 
                              sample_caller: bool = True) -> nx.MultiDiGraph:
    neighborhood = []
    for neighbor in nx.ego_graph(graph, uuid,
                                 radius=depth, center=True,
                                 undirected=True):
        neighborhood.append(neighbor)
    graph = graph.subgraph(neighborhood)
    
    # Sample
    node_set = set([uuid])
    for _ in range(depth):
        new_node_set = set(node_set)
        for node in node_set:
            successors = [_node for _node in graph.successors(node) if _node not in new_node_set]
            random.shuffle(successors)
            new_node_set.update(successors[:max_sample])
            if sample_caller:
                predecessors = [_node for _node in graph.predecessors(node) if _node not in new_node_set]
                random.shuffle(predecessors)
                new_node_set.update(predecessors[:max_sample])
        node_set = new_node_set
    graph = graph.subgraph(node_set)
    return graph

def get_self_subgraph(graph: nx.MultiDiGraph, uuid: str) -> nx.MultiDiGraph:
    neighborhood = [uuid]
    graph = graph.subgraph(neighborhood)
    return graph

def parse_json(engine_url: str, json_path: str, label: dict, task: str):

    optimization_level = "O0"
    task = "is_" + task

    file_name = os.path.basename(json_path)
    file_name = os.path.splitext(file_name)[0]
    project_name = file_name

    
    engine = create_engine(engine_url)
    with open(json_path, "r") as f:
        function_dict = json.load(f)
    call_graph = build_graph(function_dict)

    session = Session(engine)
    
    for uuid in function_dict.keys():

        function_info = function_dict[uuid]

        func_label = 1

        callee_graph = get_neighborhood_subgraph(call_graph, uuid, 1, 20, False)

        function = Function(
            uuid=uuid, name=function_info["function_name"],
            project_name=project_name, optimization_level=optimization_level,
            token_sequence=json.dumps(function_info["token_sequence"]),

            callee_graph=serialize_graph(callee_graph),
            callee_graph_size=len(callee_graph.nodes),

            label = func_label
        )

        session.add(function)
        external_function_str = ""

        for external_callee_name in function_info["call_external"].keys():
            external_function_str += external_callee_name + " "
            for idx in function_info["call_external"][external_callee_name]:
                external_call = ExternalCall(
                    caller=uuid,
                    callee=external_callee_name,
                    index=idx
                )
                session.add(external_call)
        
        external_functions = ExternalFunctions(uuid=uuid, external=external_function_str)
        session.add(external_functions)

    session.commit()
    
    for uuid in function_dict.keys():
        function_info = function_dict[uuid]
        for callee_uuid in function_info["call_to"].keys():
                
            for idx in function_info["call_to"][callee_uuid]:
                call = Call(
                    caller=uuid,
                    callee=callee_uuid,
                    index=idx
                )
                session.add(call)
    session.commit()

def main(args, task):
    engine_url = f"sqlite:///{args.db_path}"
    engine = create_engine(engine_url)
    Base.metadata.create_all(engine)

    with open(args.label_path, "r") as f:
        label = json.load(f)
    
    with Session(engine) as session:
        res = session.query(Function.uuid).all()
        existing_uuid_set = set([uuid for (uuid,) in res])
    
    engine.dispose()
    if os.path.isdir(args.json_path):
        pool = multiprocessing.Pool(args.num_workers)
        for file_name in tqdm(os.listdir(args.json_path)):
            if file_name.endswith(".json"):
                with open(os.path.join(args.json_path, file_name), "r") as f:
                    function_dict = json.load(f)
                uuid_set = set(function_dict.keys())
                if uuid_set.issubset(existing_uuid_set):
                    continue
                while True:
                    if len(pool._cache) < args.num_workers * 2:
                        break
                    time.sleep(1)
                if args.num_workers == 1:
                    parse_json(engine_url, os.path.join(args.json_path, file_name), label, task)
                else:
                    pool.apply_async(parse_json, 
                                     args=(engine_url, 
                                           os.path.join(args.json_path, file_name), label, task))
        
        if args.num_workers > 1:
            prog = log.progress("Waiting for all processes to finish")
            while True:
                prog.status(f"{len(pool._cache)} processes are still running")
                if len(pool._cache) == 0:
                    break
                time.sleep(1)
            pool.close()
            pool.join()
            prog.success("All processes finished")
        
    else:
        parse_json(engine_url, args.json_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--label_path", type=str, default=f"/path/to/DSZ-dataset")
    args = parser.parse_args()

    label_path = args.label_path

    for split in ["train", "test", "valid"]:

        print(split)
        args.label_path = label_path + split + "_label.json"
        args.db_path = f"/path/to/{split}data.db.dsz"
        args.json_path = f"/path/to/json/{split}"
        
        main(args, split)

    