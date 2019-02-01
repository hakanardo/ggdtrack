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
def main(datadir, limit):
    dataset = Duke(datadir)
    dataset.download()
    dataset.prepare()

    prep_training_graphs(dataset, limit=limit)

    model = NNModelGraphresPerConnection()
    prep_minimal_graph_diffs(dataset, model)
    prep_eval_graphs(dataset, model)

    train_graphres_minimal(dataset, "cachedir/logdir", model)

    prep_eval_tracks(dataset, "cachedir/logdir", model, threads=1)
    eval_prepped_tracks(dataset)

if __name__ == '__main__':
    main()