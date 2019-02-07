import json
import os
import uuid
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import pickle
from urllib.parse import urlparse
import requests

import torch

import sys

from tqdm import tqdm

sys.setrecursionlimit(100000)


def parallel(worker, jobs, threads=cpu_count(), tqdm_label=None):
    if threads is None:
        threads=cpu_count()
    if threads > 1:
        p = Pool(threads)
        mymap = p.imap_unordered
    else:
        mymap = map
    for res in tqdm(mymap(worker, jobs), tqdm_label, len(jobs), disable=tqdm_label is None):
        yield res

def parallel_run(worker, jobs, threads=cpu_count(), tqdm_label=None):
    for _ in parallel(worker, jobs, threads, tqdm_label):
        pass

class AtomicFile:
    def __init__(self, filename, mode="wb"):
        self.filename = filename
        self.tmp_filename = filename + '.' + str(uuid.uuid4()) + '.tmp'
        self.mode = mode

    def __enter__(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self.fd = open(self.tmp_filename, self.mode)
        return self.fd

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fd.close()
        if os.path.exists(self.filename):
            os.unlink(self.filename)
        os.link(self.tmp_filename, self.filename)
        os.unlink(self.tmp_filename)


def save_json(obj, filename):
    with AtomicFile(filename, "w") as fd:
        json.dump(obj, fd)


def save_pickle(obj, filename):
    with AtomicFile(filename) as fd:
        pickle.dump(obj, fd, -1)


def save_torch(obj, filename):
    with AtomicFile(filename) as fd:
        torch.save(obj, fd, pickle_protocol=-1)

def load_json(filename):
    with open(filename, "r") as fd:
        return json.load(fd)

def load_pickle(filename):
    with open(filename, "rb") as fd:
        return pickle.load(fd)


def save_graph(graph, filename, promote_again=True):
    indexes = {d: i for i, d in enumerate(graph)}
    for d in graph:
        d.demote_state(indexes)
    save_pickle(graph, filename)
    if promote_again:
        for d in graph:
            d.promote_state(graph)


def load_graph(filename):
    graph = load_pickle(filename)
    for d in graph:
        d.promote_state(graph)
    return graph

load_torch = torch.load

if not torch.cuda.is_available():
    default_torch_device = torch.device('cpu')
else:
    default_torch_device = torch.device('cuda')

def download_file(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    filename = os.path.join(dest_dir, filename)
    if not os.path.exists(filename):
        print("Downloading", url)
        req = requests.get(url, stream=True)
        file_size = int(req.headers["Content-Length"])

        with AtomicFile(filename) as fd:
            with tqdm(total=file_size) as pbar:
                while True:
                    buffer = req.raw.read(8192)
                    if len(buffer) == 0:
                        break
                    fd.write(buffer)
                    pbar.update(len(buffer))

