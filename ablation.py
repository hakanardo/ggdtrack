import os

import click

from ggdtrack.duke_dataset import Duke, DukeMini
from ggdtrack.visdrone_dataset import VisDrone
from ggdtrack.mot16_dataset import Mot16
from ggdtrack.eval import prep_eval_graphs, prep_eval_tracks, eval_prepped_tracks, eval_prepped_tracks_joined
from ggdtrack.graph_diff import prep_minimal_graph_diffs, find_minimal_graph_diff
from ggdtrack.klt_det_connect import prep_training_graphs
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.train import train_graphres_minimal

global_skip = {
    "LongConnectionOrder",
    "LongFalsePositiveTrack"
}

ggd_types = {
    'FalsePositive',
    'SplitFromFalsePositive',
    'ExtraFirst',
    'DualFalsePositive',
    'DetectionSkipp',
    'SkippLast',
    'Split',
    'SkippFirst',
    'SplitToFalsePositive',
    'ExtraLast',
    'IdSwitch',
    'DoubleSplitAndMerge',
    'Merge',
    'SplitAndMerge',
    'TooShortTrack',
    'LongTrack',
    'LongConnectionOrder',
    'LongFalsePositiveTrack',
}

@click.command()
@click.option("--dataset", default="Duke", help="Dataset loader class (Mot16, Duke or VisDrone)")
@click.option("--datadir", default="data", help="Directory into which the Duke dataset will be downloaded")
@click.option("--threads", default=None, type=int, help="The number of threads to use. Default is one per CPU core.")
@click.option("--segment-length", default=10, type=int, help="The length in seconds of video used for each garph")
@click.option("--cachedir", default="cachedir", help="Directory into which intermediate results are cached between runs")
@click.option("--minimal-confidence", default=None, type=float, help="Minimal confidense of detection to consider")
@click.option("--fold", default=None, type=int)
@click.option("--max-connect", default=5, type=int)
@click.option("--max-worse-eval-epochs", default=float('Inf'), type=float)
@click.option("--epochs", default=10, type=int)
@click.option("--too-short-track", default=2, type=int)
@click.option("--logdir-prefix", default="", help="Prepended to logdir path")
def main(dataset, datadir, threads, segment_length, cachedir, minimal_confidence, fold, max_connect,
         max_worse_eval_epochs, epochs, too_short_track, logdir_prefix):
    opts = dict(cachedir=cachedir, default_min_conf=minimal_confidence)
    if fold is not None:
        opts['fold'] = fold
    dataset = eval(dataset)(datadir, **opts)
    logdir = logdir_prefix + '/' + dataset.logdir

    find_minimal_graph_diff.too_short_track = too_short_track
    find_minimal_graph_diff.long_track = too_short_track * 2


    for skipped in ggd_types:
        if skipped in global_skip:
            continue

        prep_training_graphs(dataset, cachedir, limit_train_amount=0.1, threads=threads, segment_length_s=segment_length,
                             worker_params=dict(max_connect=max_connect))

        dataset.logdir = logdir + "_skipped_" + skipped
        model = NNModelGraphresPerConnection()
        prep_minimal_graph_diffs(dataset, model, threads=threads, skipped_ggd_types=global_skip.union([skipped]))
        prep_eval_graphs(dataset, model, threads=threads)

        train_graphres_minimal(dataset, model, epochs=epochs, max_worse_eval_epochs=max_worse_eval_epochs)

        prep_eval_tracks(dataset, model, 'eval', threads=1)
        res, res_int = eval_prepped_tracks(dataset, 'eval')
        open(os.path.join(dataset.logdir, "eval_results.txt"), "w").write(res)
        open(os.path.join(dataset.logdir, "eval_results_int.txt"), "w").write(res_int)

if __name__ == '__main__':
    main()
