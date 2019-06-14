# if __name__ == '__main__':
#     from torch.multiprocessing import set_start_method
#     set_start_method('forkserver')
# import warnings
# warnings.filterwarnings("ignore")
import os

import click

from ggdtrack.duke_dataset import Duke
from ggdtrack.visdrone_dataset import VisDrone
from ggdtrack.eval import prep_eval_graphs, prep_eval_tracks, eval_prepped_tracks, eval_prepped_tracks_csv
from ggdtrack.graph_diff import prep_minimal_graph_diffs
from ggdtrack.klt_det_connect import prep_training_graphs
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.train import train_graphres_minimal

@click.command()
@click.option("--dataset", default="Duke", help="Dataset loader class (Duke or VisDrone)")
@click.option("--datadir", default="data", help="Directory into which the Duke dataset will be downloaded")
@click.option("--limit", default=None, type=int, help="The number of graphs to use. Default is all of them.")
@click.option("--threads", default=None, type=int, help="The number of threads to use. Default is one per CPU core.")
@click.option("--segment-length", default=10, type=int, help="The length in seconds of video used for each garph")
@click.option("--cachedir", default="cachedir", help="Directory into which intermediate results are cached between runs")
@click.option("--minimal-confidence", default=None, type=float, help="Minimal confidense of detection to consider")
@click.option("--fold", default=None, type=int)
def main(dataset, datadir, limit, threads, segment_length, cachedir, minimal_confidence, fold):
    opts = dict(cachedir=cachedir, default_min_conf=minimal_confidence)
    if fold is not None:
        opts['fold'] = fold
    dataset = eval(dataset)(datadir, **opts)
    dataset.download()
    dataset.prepare()

    prep_training_graphs(dataset, cachedir, limit=limit, threads=threads, segment_length_s=segment_length)

    model = NNModelGraphresPerConnection()
    prep_minimal_graph_diffs(dataset, model, threads=threads)
    prep_eval_graphs(dataset, model, threads=threads)

    train_graphres_minimal(dataset, model)

    prep_eval_tracks(dataset, model, 'eval', threads=1)
    res, res_int = eval_prepped_tracks(dataset, 'eval')
    open(os.path.join(dataset.logdir, "eval_results.txt"), "w").write(res)
    open(os.path.join(dataset.logdir, "eval_results_int.txt"), "w").write(res_int)
    eval_prepped_tracks_csv(dataset, 'eval')

    prep_eval_tracks(dataset, model, 'test', threads=1)
    eval_prepped_tracks_csv(dataset, 'test')
    dataset.prepare_submition()


if __name__ == '__main__':
    main()