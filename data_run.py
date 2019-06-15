import os
from glob import glob

import click

from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import prep_eval_tracks, eval_prepped_tracks, ConnectionBatch, prep_eval_gt_tracks, prep_eval_graphs
from ggdtrack.graph_diff import prep_minimal_graph_diffs
from ggdtrack.klt_det_connect import prep_training_graphs
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.train import train_graphres_minimal

@click.command()
@click.option("--threads", default=None, type=int, help="The number of threads to use. Default is one per CPU core.")
def main(threads):
    max_extra = 3
    dataset = Duke("data")
    logdir = dataset.logdir

    prep_training_graphs(dataset, dataset.cachedir, threads=threads)

    model = NNModelGraphresPerConnection()
    prep_minimal_graph_diffs(dataset, model, threads=threads)
    prep_eval_graphs(dataset, model, threads=threads)

    for train_amount in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        for itt in range(10):
            dataset.logdir = logdir +  "_%8.6f_%.2d" % (train_amount, itt)
            model = NNModelGraphresPerConnection()
            train_graphres_minimal(dataset, model, epochs=1000, max_worse_eval_epochs=max_extra, train_amount=train_amount)

            fn = sorted(glob("%s/snapshot_???.pyt" % dataset.logdir))[-max_extra-1]
            prep_eval_tracks(dataset, fn, model, 'eval', threads=1)
            res, res_int = eval_prepped_tracks(dataset, 'eval')
            open(os.path.join(dataset.logdir, "eval_results.txt"), "w").write(res)
            open(os.path.join(dataset.logdir, "eval_results_int.txt"), "w").write(res_int)

    prep_eval_gt_tracks(dataset, NNModelGraphresPerConnection)
    res, res_int = eval_prepped_tracks(dataset, 'eval')
    open(os.path.join(dataset.cachedir, "eval_gt_results.txt"), "w").write(res)
    open(os.path.join(dataset.cachedir, "eval_gt_results_int.txt"), "w").write(res_int)


if __name__ == '__main__':
    main()

