# if __name__ == '__main__':
#     from torch.multiprocessing import set_start_method
#     set_start_method('forkserver')
# import warnings
# warnings.filterwarnings("ignore")
import os

import click

from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import prep_eval_graphs, prep_eval_tracks, eval_prepped_tracks, eval_prepped_tracks_csv
from ggdtrack.graph_diff import prep_minimal_graph_diffs
from ggdtrack.klt_det_connect import prep_training_graphs
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.train import train_graphres_minimal, train_frossard


@click.command()
@click.option("--datadir", default="data", help="Directory into which the Duke dataset will be downloaded")
def main(datadir):
    dataset = Duke(datadir)

    model = NNModelGraphresPerConnection()

    prep_eval_graphs(dataset, NNModelGraphresPerConnection(), parts=["train"])
    # train_frossard(dataset, "cachedir/logdir_fossard", model, resume_from="cachedir/logdir/model_0000.pyt", epochs=10)
    train_frossard(dataset, "cachedir/logdir_fossard", model, mean_from="cachedir/logdir/snapshot_009.pyt", epochs=1000)

    prep_eval_tracks(dataset, "cachedir/logdir_fossard", model, 'eval', threads=1)
    res, res_int = eval_prepped_tracks(dataset, 'eval')

    open("cachedir/logdir_fossard/eval_results.txt", "w").write(res)
    open("cachedir/logdir_fossard/eval_results_int.txt", "w").write(res_int)


if __name__ == '__main__':
    main()