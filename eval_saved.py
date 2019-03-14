import re
from glob import glob
from os import stat
from random import shuffle

import torch

from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import eval_prepped_tracks, prep_eval_tracks
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.utils import save_pickle, save_json

dataset = Duke("data")
model = NNModelGraphresPerConnection()

models = glob("cachedir/logdir/model*")
shuffle(models)

motas = []
for fn in models:
    model.load_state_dict(torch.load(fn))
    prep_eval_tracks(dataset, None, model, 'eval', threads=1)
    res, res_int = eval_prepped_tracks(dataset, 'eval')
    mota = float(re.split(r'\s+', res_int.split('\n')[-1])[4].replace('%', ''))
    t = stat(fn).st_ctime_ns / 1e9
    motas.append((t, mota))
    save_json(motas, "cachedir/logdir/motas.json")

