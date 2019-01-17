import json
import os
import uuid
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import pickle
import torch

import sys
sys.setrecursionlimit(100000)


def parallel(worker, jobs, threads=cpu_count()):
    if threads > 1:
        p = Pool(threads)
        mymap = p.imap_unordered
    else:
        mymap = map
    for res in mymap(worker, jobs):
        yield res


def save_json(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    tmp_filename = filename + '.' + str(uuid.uuid4()) + '.tmp'
    with open(tmp_filename, "w") as fd:
        json.dump(obj, fd)
    if os.path.exists(filename):
        os.unlink(filename)
    os.link(tmp_filename, filename)
    os.unlink(tmp_filename)


def save_pickle(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    tmp_filename = filename + str(uuid.uuid4()) + '.tmp'
    with open(tmp_filename, "wb") as fd:
        pickle.dump(obj, fd, -1)
    if os.path.exists(filename):
        os.unlink(filename)
    os.link(tmp_filename, filename)
    os.unlink(tmp_filename)

def save_torch(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    tmp_filename = filename + str(uuid.uuid4()) + '.tmp'
    torch.save(obj, tmp_filename, pickle_protocol=-1)
    if os.path.exists(filename):
        os.unlink(filename)
    os.link(tmp_filename, filename)
    os.unlink(tmp_filename)

def load_json(filename):
    with open(filename, "r") as fd:
        return json.load(fd)

def load_pickle(filename):
    with open(filename, "rb") as fd:
        return pickle.load(fd)

load_torch = torch.load
