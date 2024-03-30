#!/usr/bin/env python3
from sqlalchemy import create_engine
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

import networkx as nx
import sqlalchemy
import functools
import random
import typing
import torch
import zlib
import base64
import json
import os

from tqdm.auto import tqdm

"""
Define necessary data structures and operations on dataset
"""

def serialize_graph(graph: nx.MultiDiGraph) -> str:
    return base64.b64encode(
                zlib.compress(json.dumps(
                    nx.node_link_data(graph)
                    ).encode()))

def deserialize_graph(graph_str: str) -> nx.MultiDiGraph:
    return nx.node_link_graph(
                json.loads(
                    zlib.decompress(
                        base64.b64decode(graph_str)
                        ).decode()))

class Base(DeclarativeBase):
    pass

"""
Below defines the db format.
"""

class Function(Base):
    __tablename__ = "function"
    """
    Each (function + optlevel) function instance is uuid'ed
    The ultimate primary key for every compiled function every compilation
    """
    uuid: Mapped[str] = mapped_column(primary_key=True)
    """
    Function name
    """
    name: Mapped[str]
    """
    Project name
    """
    project_name: Mapped[str]
    """
    Optimization level as string ("O0", "O1", "O2", "O3", "Os")
    """
    optimization_level: Mapped[str]
    """
    token sequence
    Logically it is List[str]; in db it is json serialized string
    """
    token_sequence: Mapped[str]
    
    """
    serilized networkx graph
    base64.b64encode(zlib.compress(json.dumps(nx.link_node_data(G).encode())))
    """
    # all_graph_1: Mapped[str]
    # all_graph_1_size: Mapped[int]
    # all_graph_2: Mapped[str]
    # all_graph_2_size: Mapped[int]
    # all_graph_3: Mapped[str]
    # all_graph_3_size: Mapped[int]
    # all_graph_4: Mapped[str]
    # all_graph_4_size: Mapped[int]
    
    callee_graph: Mapped[str]
    callee_graph_size: Mapped[int]
    # callee_graph_3: Mapped[str]
    # callee_graph_3_size: Mapped[int]
    # callee_graph_4: Mapped[str]
    # callee_graph_4_size: Mapped[int]
    label: Mapped[int]


class Call(Base):
    __tablename__ = "call"
    """
    function uuid for caller
    """
    caller: Mapped[str] = mapped_column(
        ForeignKey("function.uuid"),
        primary_key=True)
    """
    function uuid for callee
    """
    callee: Mapped[str] = mapped_column(
        ForeignKey("function.uuid"),
        primary_key=True)
    """
    the call is ith in caller body
    """
    index: Mapped[int] = mapped_column(primary_key=True)

class ExternalCall(Base):
    __tablename__ = "external_call"
    """
    function uuid for caller
    """
    caller: Mapped[str] = mapped_column(
        ForeignKey("function.uuid"),
        primary_key=True)
    """
    function NAME for callee
    """
    callee: Mapped[str] = mapped_column(primary_key=True)
    """
    the call is ith in caller body
    """
    index: Mapped[int] = mapped_column(primary_key=True)

class ExternalFunctions(Base):
    __tablename__ = "external_functions"
    """
    Each (function + optlevel) function instance is uuid'ed
    The ultimate primary key for every compiled function every compilation
    """
    uuid: Mapped[str] = mapped_column(primary_key=True)
    """
    functions NAME for external
    """
    external: Mapped[str] = mapped_column(primary_key=True)


class BinaryDataset(torch.utils.data.Dataset):
    """
    abstract layer including logics about sqlite
    not a independently usable layer
    """
    def __init__(self, db_path: str,
                 data_dir: str):
        if not os.path.exists(db_path):
            raise Exception("Database file doesn't exist")
        self.db_path = os.path.abspath(db_path)
        if data_dir is not None:
            self.data_dir = os.path.abspath(data_dir)
        self.engine = None
        self.establish_connection()

    def establish_connection(self):
        """
        idempotent function
        if not established, establish it
        """
        if self.engine is None:
            self.engine = create_engine(f"sqlite:///{self.db_path}")

    def close_connection(self):
        """
        idempotent function
        if established, close it
        """
        if self.engine is not None:
            self.engine.dispose()
            del self.engine
            self.engine = None

    def dataloader(self, *args, **kwargs):
        """
        This function considers that DataLoader can fork, and db connection
        does not live well across forked processes.
        """
        # Close the connection before creating the dataloader
        self.close_connection()
        _dataloader = torch.utils.data.DataLoader(self, *args, **kwargs)
        return _dataloader

    @functools.lru_cache(maxsize=1024)
    def token_sequence(self, uuid: str) -> typing.List[str]:
        """
        Lookup token sequence for a func uuid
        returns List[str(token)]
        """
        self.establish_connection()
        with Session(self.engine) as session:
            function = session.query(Function).filter(
                Function.uuid == uuid).one()
            token_sequence = json.loads(function.token_sequence)
            return token_sequence
    
    def load_graph(self, uuid: str, layer_cnt: int, sample_caller: bool) -> nx.MultiDiGraph:
        self.establish_connection()
        key_name = f"{'all' if sample_caller else 'callee'}_graph"
        with Session(self.engine) as session:
            compressed_graph = session.query(
                getattr(Function, key_name)).filter(
                Function.uuid == uuid).one()[0]
            graph = deserialize_graph(graph_str=compressed_graph)
        return graph

    def external_calls_from(self, uuid: typing.Union[str, typing.List[str]]) -> typing.List[typing.Tuple[str, str, int]]:
        self.establish_connection()
        with Session(self.engine) as session:
            if isinstance(uuid, list):
                calls = session.query(ExternalCall).filter(
                    ExternalCall.caller.in_(uuid)).all()
            elif isinstance(uuid, str):
                calls = session.query(ExternalCall).filter(
                    ExternalCall.caller == uuid).all()
            else:
                raise Exception(f"uuid must be str or list[str], not {type(uuid)}")
        return [(call.caller, call.callee, call.index) for call in calls]

    @functools.lru_cache(maxsize=2048)
    def calls_from(self, uuid: str, limit: int = None, callee_in: typing.
                   List[str] = None) -> typing.List[typing.Tuple[str, str, int]]:
        """
        Lookup calls from a func by its uuid
        * not considering external calls
        * can specify callee subset or max number of records
        
        call index i: the call appears ith in caller body
        
        returns List[Tuple[str(caller uuid), str(callee uuid), int(call index)]]
        """
        self.establish_connection()
        if limit is not None and callee_in is not None:
            raise Exception("Cannot specify both limit and callee_in")
        with Session(self.engine) as session:
            if limit is not None:
                calls = session.query(Call).filter(
                    Call.caller == uuid).order_by(
                    func.random()).limit(limit).all()
            elif callee_in is not None:
                calls = session.query(Call).filter(
                    Call.caller == uuid).filter(
                    Call.callee.in_(callee_in)).all()
            else:
                calls = session.query(Call).filter(Call.caller == uuid).all()
            return [(call.caller, call.callee, call.index) for call in calls]

    @functools.lru_cache(maxsize=2048)
    def calls_to(self, uuid: str, limit: int = None, caller_in: typing.List
                 [str] = None) -> typing.List[typing.Tuple[str, str, int]]:
        """
        Lookup calls to a func by its uuid
        * can specify callee subset or max number of records
        
        call index i: the call appears ith in caller body
        returns List[Tuple[str(caller uuid), str(callee uuid), int(call index)]]
        """
        self.establish_connection()
        if limit is not None and caller_in is not None:
            raise Exception("Cannot specify both limit and caller_in")
        with Session(self.engine) as session:
            if limit is not None:
                calls = session.query(Call).filter(
                    Call.callee == uuid).order_by(
                    func.random()).limit(limit).all()
            elif caller_in is not None:
                calls = session.query(Call).filter(
                    Call.callee == uuid).filter(
                    Call.caller.in_(caller_in)).all()
            else:
                calls = session.query(Call).filter(Call.callee == uuid).all()
            return [(call.caller, call.callee, call.index) for call in calls]

    def call_bulk(self,
                  caller_in: typing.List[str],
                  callee_in: typing.List[str]) -> typing.List[typing.Tuple[str, str, int]]:
        """
        Lookup calls from a list of callers to a list of callees
        
        returns List[Tuple[str(caller uuid), str(callee uuid), int(call index)]]
        """
        self.establish_connection()
        with Session(self.engine) as session:
            calls = session.query(Call).filter(
                Call.caller.in_(caller_in)).filter(
                Call.callee.in_(callee_in)).all()
            return [(call.caller, call.callee, call.index) for call in calls]

class BinaryIterDataset(BinaryDataset):
    """
    Helper class for traversal on the whole dataset
    """
    def __init__(self, db_path: str, data_dir: str):
        super(BinaryIterDataset, self).__init__(db_path, data_dir)
        self.establish_connection()

        with Session(self.engine) as session:
            self.uuid_list = [r[0] for r in session.query(Function.uuid).all()]

    def __len__(self):
        return len(self.uuid_list)


class BinarySampleDataset(BinaryDataset):
    """
    Helper class for sampling on the dataset
    """
    def __init__(self, db_path: str, data_dir: str, num_classes: int,
                 optimization_levels: typing.List[str] = None):
        """
        selecting all functions with all supported optimization levels
        if optimization_levels is not specified, any subset support is accepted
        will fill member variable:
        function_list:
            List[{
                "project_name": str,
                "function_name": str,
                "uuid_dict": Dict[str(optim level), str(func uuid)]
            }]
        """
        super(BinarySampleDataset, self).__init__(db_path, data_dir)
        self.establish_connection()

        if optimization_levels is None:
            #self.optimization_levels = supported_optimization_levels
            self.function_list = []
            with Session(self.engine) as session:
                uuid_opl = func.group_concat(
                    Function.uuid + ":" + Function.optimization_level, "|")
                result = session.query(
                    Function.project_name, Function.name, uuid_opl, Function.label
                ).group_by(
                    Function.project_name, Function.name
                ).all()
                for project_name, name, uuid_opl, label in result: 
                    if label < 1 or label > num_classes:
                        continue

                    function_dict = {}                  
                    function_dict["project_name"] = project_name
                    function_dict["function_name"] = name
                    function_dict["label"] = label
                    function_dict["uuid_dict"] = {}
                    for uuid_opl_pair in uuid_opl.split("|"):  
                        uuid, opl = uuid_opl_pair.split(":")
                        function_dict["uuid_dict"][opl] = uuid 
                    
                    ExternalFuncs = session.query(ExternalFunctions).filter(
                        ExternalFunctions.uuid == uuid).one()
                    function_dict["external"] = ExternalFuncs.external

                    self.function_list.append(function_dict)


    def sample_uuid(self, idx, count):
        """
        index: int(index on self.function_list)
        count: int(number of optimization levels to sample)
        
        return List[str(func uuid)]
        """

        function_dict = self.function_list[idx]
        try:
            selected_optimization_levels = random.sample(
                function_dict["uuid_dict"].keys(), count)
        except Exception:
            print(function_dict)
        uuid_list = [
            function_dict["uuid_dict"][opl]
            for opl in selected_optimization_levels]
    
        return uuid_list

    def remove_non_inline_function(self, save_json = None):
        """
        If functions in `self.function_list` have no inline functions in O0~Os, 
        just remove them.
        """
        self.add_call_statistics()
        
        new_function_list = self.get_maybe_inline_functions_fast(save_json)
        self.function_list = new_function_list
    
    def add_call_statistics(self):
        """
        Add the callee statistic to the function list, 
        add new dict into function_dict:
        "calls_info": {
            "O0":{
                "callee_1_uuid": called_count,
                ...
            },
            "O1":{
                "callee_1_uuid": called_count,
                ...
            },
            ...
        }
        """
        itertor = tqdm(self.function_list, desc = "add function call statistics ... ")
        
        for func_dict in itertor:
            calls_info = {}
            
            # iterate the optimization levels in the `uuid_dict`
            for opt_level in func_dict["uuid_dict"].keys():
                uuid = func_dict["uuid_dict"][opt_level]
                
                # get the callee statistics of the function in the specific optimization level
                single_func_callinfo = get_function_callinfo(self.calls_from(uuid))
                calls_info[opt_level] = single_func_callinfo
            
            func_dict["calls_info"] = calls_info
    
    
    def get_maybe_inline_functions_fast(self, save_json:str = None) -> typing.List[typing.Tuple[str, str]]:
        """
        Return the new function list, which contains the functions that may be inlined.
        #! This function should be called after `add_call_statistics`
        #? Fast means we only compare the callsite count of the function, but not get the inlined functions
        """
        
        # A list of `function dict`
        new_function_list = []
        
        # A list contains (project_name, function_name) of the functions that may be inlined
        maybe_inline_functions = []
        
        # judge if the `add_call_statistics` has been called
        if "calls_info" not in self.function_list[0]:
            raise Exception("Please call `add_call_statistics` before calling this function.")
        
        itertor = tqdm(self.function_list, desc = "Getting inline functions")
        for func_dict in itertor:
            # the count will always >=0 , -1 means not initialized
            total_count = -1
            for opt_level in func_dict["calls_info"].keys():
                callinfo = func_dict["calls_info"][opt_level]
                prev_total_count = total_count
                total_count = get_callsite_count_by_callinfo(callinfo)
                
                if prev_total_count != -1 and prev_total_count != total_count:
                    new_function_list.append(func_dict)
                    maybe_inline_functions.append((func_dict["project_name"], func_dict["function_name"]))
                    break
        
        if save_json:
            with open(save_json, "w") as f:
                json.dump(maybe_inline_functions, f, indent = 4)
                
        print(f"Total {len(new_function_list)} functions may have inline functions.")
        
        return new_function_list
    
    def __len__(self):
        return len(self.function_list)

