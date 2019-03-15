from glob import glob

from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import prep_eval_tracks, eval_prepped_tracks, ConnectionBatch
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.train import train_graphres_minimal

def main():
    max_extra = 3
    dataset = Duke("data")

    for train_amount in [0.01, 0.1, 1.0]:
        for itt in range(10):
            logdir = "cachedir/logdir_%4.2f_%.2d" % (train_amount, itt)
            model = NNModelGraphresPerConnection()
            train_graphres_minimal(dataset, logdir, model, epochs=1000, max_worse_eval_epochs=max_extra, train_amount=train_amount)

            fn = sorted(glob("%s/snapshot_???.pyt" % logdir))[-max_extra-1]
            print(fn)
            prep_eval_tracks(dataset, logdir, model, 'eval', threads=1)
            res, res_int = eval_prepped_tracks(dataset, 'eval')
            open(logdir + "/eval_results.txt", "w").write(res)
            open(logdir + "/eval_results_int.txt", "w").write(res_int)

if __name__ == '__main__':
    main()

