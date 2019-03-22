from glob import glob

from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import prep_eval_tracks, eval_prepped_tracks, ConnectionBatch, prep_eval_gt_tracks
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.train import train_graphres_minimal

def main():
    max_extra = 3
    dataset = Duke("data")


    for train_amount in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        for itt in range(10):
            logdir = "cachedir/logdir_%8.6f_%.2d" % (train_amount, itt)
            model = NNModelGraphresPerConnection()
            train_graphres_minimal(dataset, logdir, model, epochs=1000, max_worse_eval_epochs=max_extra, train_amount=train_amount)

            fn = sorted(glob("%s/snapshot_???.pyt" % logdir))[-max_extra-1]
            prep_eval_tracks(dataset, fn, model, 'eval', threads=1)
            res, res_int = eval_prepped_tracks(dataset, 'eval')
            open(logdir + "/eval_results.txt", "w").write(res)
            open(logdir + "/eval_results_int.txt", "w").write(res_int)

    prep_eval_gt_tracks(dataset, NNModelGraphresPerConnection)
    res, res_int = eval_prepped_tracks(dataset, 'eval')
    open("cachedir/eval_gt_results.txt", "w").write(res)
    open("cachedir/eval_gt_results_int.txt", "w").write(res_int)


if __name__ == '__main__':
    main()

