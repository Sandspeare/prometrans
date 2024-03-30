#!/usr/bin/env python3
import argparse
import sqlite3
import sys
import os

from tqdm.auto import tqdm

def do_consolidate(args: argparse.Namespace):
    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()
    to_delete = []
    remaining = []
    cursor.execute("SELECT project_name, function_name, COUNT(optimization_level) FROM functions GROUP BY project_name, function_name;")
    for project_name, function_name, optimization_level_count in cursor.fetchall():
        if optimization_level_count == 5:
            remaining.append((project_name, function_name))
        else:
            to_delete.append((project_name, function_name))
    print("DELETE BEGIN")
    for project_name, function_name in tqdm(to_delete):
        cursor.execute("DELETE FROM functions WHERE project_name=? AND function_name=?", (project_name, function_name))
    connection.commit()
    print("DELETE OK")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, help="The path of sqlite3 db file", required=True)
    args = parser.parse_args()
    
    do_consolidate(args)