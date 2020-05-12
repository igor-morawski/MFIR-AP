import json
import os

def read_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data

def read_json_key(filepath, key):
    data = read_json(filepath)
    return data[key]
