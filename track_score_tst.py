from glob import glob

import torch
import numpy as np

from ggdtrack.model import NNModelSimple
from ggdtrack.mot16_dataset import Mot16
from ggdtrack.utils import load_pickle

dataset = Mot16('/home/hakan/src/ggdtrack/data/')
scene = dataset.scene("train__MOT16-04"),
tracks = load_pickle("cachedir/tracks/MOT16_fold0_graph_train__MOT16-04_00000001.pck")
tr1 = tracks[12]
tr2 = tracks[35]

model = NNModelSimple()
fn = sorted(glob("%s/snapshot_???.pyt" % dataset.logdir))[-1]
print(fn)
model.load_state_dict(torch.load(fn)['model_state'])
model.eval()

def sum_connection_weights(tr):
    sa = 0
    for i in range(len(tr) - 1):
        d1, d2 = tr[i], tr[i+1]
        f = torch.tensor(model.connection_weight_feature(d1, d2)[0].astype(np.float32))
        sa += model.edge_model.klt_model(f)
        print(sa)
        exit()
    return sa

def sum_detection_weights(tr):
    sa = 0
    for det in tr:
        f = torch.tensor(model.detecton_weight_feature(det))
        sa += model.detection_model(f)
    return sa

# d1 = sum_detection_weights(tr1).item()
# d2 = sum_detection_weights(tr2).item()
c1 = sum_connection_weights(tr1).item()
c2 = sum_connection_weights(tr2).item()
print(d1, '+', c1, '=', d1 + c1)
print(d2, '+', c2, '=', d2 + c2)