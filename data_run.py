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
@click.option("--cachedir", default="cachedir", help="Directory into which intermediate results are cached between runs")
@click.option("--train-amounts", default="0.001,0.01,0.1,1.0", help="Amount of data to use")
@click.option("--itterations", default=10, type=int, help="Amount of data to use")
@click.option("--logdir-prefix", default="", help="Prepended to logdir path")
def main(threads, cachedir, train_amounts, itterations, logdir_prefix):
    max_extra = 3
    dataset = Duke("data")
    logdir = logdir_prefix + '/' + dataset.logdir


    global_skip = {"LongConnectionOrder", "LongFalsePositiveTrack"}

    for train_amount in map(float, train_amounts.split(',')):
        for itt in range(int(itterations)):

            prep_training_graphs(dataset, cachedir, limit_train_amount=train_amount, threads=threads, seed=hash(logdir)+itt)
            model = NNModelGraphresPerConnection()
            prep_minimal_graph_diffs(dataset, model, threads=threads, skipped_ggd_types=global_skip)
            prep_eval_graphs(dataset, model, threads=threads)

            dataset.logdir = logdir +  "_%8.6f_%.2d" % (train_amount, itt)
            train_graphres_minimal(dataset, model, epochs=1000, max_worse_eval_epochs=max_extra, train_amount=train_amount)

            fn = sorted(glob("%s/snapshot_???.pyt" % dataset.logdir))[-max_extra-1]
            prep_eval_tracks(dataset, model, 'eval', threads=1, snapshot=fn)
            res, res_int = eval_prepped_tracks(dataset, 'eval')
            open(os.path.join(dataset.logdir, "eval_results.txt"), "w").write(res)
            open(os.path.join(dataset.logdir, "eval_results_int.txt"), "w").write(res_int)

    prep_eval_gt_tracks(dataset, NNModelGraphresPerConnection)
    res, res_int = eval_prepped_tracks(dataset, 'eval')
    open(os.path.join(dataset.cachedir, "eval_gt_results.txt"), "w").write(res)
    open(os.path.join(dataset.cachedir, "eval_gt_results_int.txt"), "w").write(res_int)


if __name__ == '__main__':
    main()

