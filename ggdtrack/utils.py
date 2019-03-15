import json
import os
import uuid
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import pickle
from random import shuffle
from urllib.parse import urlparse
import requests

import torch

import sys

from torch.utils.data import Subset
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
        dir_name = os.path.dirname(self.filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
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

def demote_graph(graph):
    indexes = {d: i for i, d in enumerate(graph)}
    for d in graph:
        assert graph[indexes[d]] is d
    for d in graph:
        d.demote_state(indexes)

def promote_graph(graph):
    for d in graph:
        d.promote_state(graph)

def save_graph(graph, filename, promote_again=True):
    demote_graph(graph)
    save_pickle(graph, filename)
    if promote_again:
        promote_graph(graph)

def load_graph(filename):
    graph = load_pickle(filename)
    promote_graph(graph)
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

class WorkerPoolDoneWork: pass

class WorkerPool:
    def __init__(self, num_workers, worker_fn, output_queue_len=None):
        if output_queue_len is None:
            output_queue_len = 2 * num_workers + 2
        self.input_queue = torch.multiprocessing.Queue(num_workers + 1)
        self.output_queue = torch.multiprocessing.Queue(output_queue_len)
        self.workers = []
        self.worker_fn = worker_fn

        for i in range(num_workers):
            w = torch.multiprocessing.Process(target=self.worker_loop, daemon=True)
            self.workers.append(w)
            w.start()

    def worker_loop(self):
        while True:
            work = self.input_queue.get()
            if work is WorkerPoolDoneWork:
                print("Done")
                break
            result = self.worker_fn(work)
            self.output_queue.put(result)

    def __del__(self):
        for _ in self.workers:
            self.input_queue.put(WorkerPoolDoneWork)

    def get(self, block=True, timeout=None):
        return self.output_queue.get(block, timeout)

    def put(self, work, block=True, timeout=None):
        self.input_queue.put(work, block, timeout)

def single_example_passthrough(batch):
    assert len(batch) == 1
    return batch[0]

class RandomSubset(Subset):
    def __init__(self, dataset, n):
        n = int(n)
        self.dataset = dataset
        indices = list(range(len(dataset)))
        shuffle(indices)
        self.indices = indices[:n]
        self.unused_indices = indices[n:]
