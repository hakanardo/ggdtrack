# if __name__ == '__main__':
#     from torch.multiprocessing import set_start_method
#     set_start_method('forkserver')
# import warnings
# warnings.filterwarnings("ignore")

import click

from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import prep_eval_graphs, prep_eval_tracks, eval_prepped_tracks
from ggdtrack.graph_diff import prep_minimal_graph_diffs
from ggdtrack.klt_det_connect import prep_training_graphs
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.train import train_graphres_minimal

@click.command()
@click.option("--datadir", default="data", help="Directory into which the Duke dataset will be downloaded")
@click.option("--limit", default=None, type=int, help="The number of graphs to use. Default is all of them.")
@click.option("--threads", default=None, type=int, help="The number of threads to use. Default is one per CPU core.")
def main(datadir, limit, threads):
    dataset = Duke(datadir)
    dataset.download()
    dataset.prepare()

    prep_training_graphs(dataset, limit=limit, threads=threads)

    model = NNModelGraphresPerConnection()
    prep_minimal_graph_diffs(dataset, model, threads=threads)
    prep_eval_graphs(dataset, model, threads=threads)

    train_graphres_minimal(dataset, "cachedir/logdir", model)

    prep_eval_tracks(dataset, "cachedir/logdir", model, 'eval', threads=1)
    res, res_int = eval_prepped_tracks(dataset, 'eval')
    open("cachedir/logdir/eval_results.txt", "w").write(res)
    open("cachedir/logdir/eval_results_int.txt", "w").write(res_int)

    prep_eval_tracks(dataset, "cachedir/logdir", model, 'test', threads=1)

if __name__ == '__main__':
    main()