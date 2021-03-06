# if __name__ == '__main__':
#     from torch.multiprocessing import set_start_method
#     set_start_method('forkserver')
# import warnings
# warnings.filterwarnings("ignore")
import os

import click

from ggdtrack.eval import prep_eval_graphs, prep_eval_tracks, eval_prepped_tracks, eval_prepped_tracks_csv, \
    prep_eval_gt_tracks
from ggdtrack.graph_diff import prep_minimal_graph_diffs
from ggdtrack.mot16_dataset import Mot16
from ggdtrack.simple_det_connect import prep_training_graphs
from ggdtrack.model import NNModelSimple, NNModelSimpleMLP1, NNModelSimpleMLP2
from ggdtrack.train import train_graphres_minimal

@click.command()
@click.option("--datadir", default="data", help="Directory into which the Duke dataset will be downloaded")
@click.option("--limit", default=None, type=int, help="The number of graphs to use. Default is all of them.")
@click.option("--threads", default=None, type=int, help="The number of threads to use. Default is one per CPU core.")
@click.option("--segment-length", default=10, type=int, help="The length in seconds of video used for each garph")
@click.option("--cachedir", default="cachedir", help="Directory into which intermediate results are cached between runs")
@click.option("--minimal-confidence", default=None, type=float, help="Minimal confidense of detection to consider")
@click.option("--fold", default=0, type=int)
@click.option("--model", default="NNModelSimple")
@click.option("--evalgt", default=False, is_flag=True)
def main(datadir, limit, threads, segment_length, cachedir, minimal_confidence, fold, model, evalgt):
    # dataset = eval(dataset)(datadir, cachedir=cachedir, default_min_conf=minimal_confidence)
    # dataset.download()
    # dataset.prepare()

    dataset = Mot16(datadir, cachedir=cachedir, default_min_conf=minimal_confidence, fold=fold)
    dataset.logdir += '_' + model
    model = eval(model)()

    prep_training_graphs(dataset, cachedir, limit=limit, threads=threads, segment_length_s=segment_length)
    prep_minimal_graph_diffs(dataset, model, threads=threads)
    prep_eval_graphs(dataset, model, threads=threads)

    if evalgt:
        dataset.logdir += '_gt'
        prep_eval_gt_tracks(dataset, model, 'eval', split_on_no_edge=False)
    else:
        train_graphres_minimal(dataset, model)
        prep_eval_tracks(dataset, model, 'eval', threads=1)


    res, res_int = eval_prepped_tracks(dataset, 'eval')
    open(os.path.join(dataset.logdir, "eval_results.txt"), "w").write(res)
    open(os.path.join(dataset.logdir, "eval_results_int.txt"), "w").write(res_int)
    # eval_prepped_tracks_csv(dataset, 'eval')
    #
    # prep_eval_tracks(dataset, model, 'test', threads=1)
    # eval_prepped_tracks_csv(dataset, 'test')
    # dataset.prepare_submition()


if __name__ == '__main__':
    main()