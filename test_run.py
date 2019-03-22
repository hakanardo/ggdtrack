import os
from glob import glob

import torch

from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import prep_eval_tracks, eval_prepped_tracks, ConnectionBatch, prep_eval_gt_tracks, \
    eval_prepped_tracks_csv
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.train import train_graphres_minimal

def main():
    max_extra = 3
    dataset = Duke("data")

    logdir = "cachedir/logdir_1.0_0.2"
    model = NNModelGraphresPerConnection()
    fn = sorted(glob("%s/snapshot_???.pyt" % logdir))[-max_extra-1]
    snapshot = torch.load(fn)
    model.load_state_dict(snapshot['model_state'])

    prep_eval_tracks(dataset, logdir, model, 'test', threads=1)
    eval_prepped_tracks_csv(dataset, logdir, 'test')

    os.system("cat  %s/result_duke_test_int/*_submit.txt > %s/duke.txt" % (logdir, logdir))
    if os.path.exists("%s/duke.zip" % logdir):
        os.unlink("%s/duke.zip" % logdir)
    os.system("cd %s; zip duke.zip duke.txt" % logdir)


if __name__ == '__main__':
    main()

