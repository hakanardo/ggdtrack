import re
from glob import glob
from os import stat

import torch

from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import eval_prepped_tracks, prep_eval_tracks
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.utils import save_pickle, save_json

dataset = Duke("data")
model = NNModelGraphresPerConnection()

motas = []
times = []
for fn in sorted(glob("cachedir/logdir/model*"))[:10]:
    model.load_state_dict(torch.load(fn))
    prep_eval_tracks(dataset, None, model, 'eval', threads=1)
    res, res_int = eval_prepped_tracks(dataset, 'eval')
    mota = float(re.split(r'\s+', res_int.split('\n')[-1])[4].replace('%', ''))
    motas.append(mota)
    times.append(stat(fn).st_ctime_ns / 1e9)
    save_json(motas, "cachedir/logdir/motas.json")

# from matplotlib.pyplot import plot, show
# times = [t-times[0] for t in times]
# plot(times, motas)
# show()