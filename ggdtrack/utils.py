import json
import os
import uuid
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import pickle
from urllib.parse import urlparse

import torch

from torch.utils.model_zoo import _download_url_to_file

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

default_torch_device = torch.device('cpu')

def download_file(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    filename = os.path.join(dest_dir, filename)
    if not os.path.exists(filename):
        print("Downloading", url)
        _download_url_to_file(url, filename, None, True)
