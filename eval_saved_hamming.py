import re
from glob import glob
from os import stat
from random import shuffle

import torch

from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import eval_hamming, ConnectionBatch, prep_eval_graphs
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.utils import  save_json


dataset = Duke("data")
model = NNModelGraphresPerConnection()

prep_eval_graphs(dataset, NNModelGraphresPerConnection(), parts=["train"])

models = glob("cachedir/logdir/model*")
shuffle(models)

hammings = []
for fn in models:
    model.load_state_dict(torch.load(fn))
    hamming = eval_hamming(dataset, None, model)
    print(hamming)
    t = stat(fn).st_ctime_ns / 1e9
    hammings.append((t, hamming))
    save_json(hammings, "cachedir/logdir/hammings.json")

