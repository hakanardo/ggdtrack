import os
import click

from ggdtrack.duke_dataset import Duke
from ggdtrack.visdrone_dataset import VisDrone
from ggdtrack.eval import prep_eval_gt_tracks, eval_prepped_tracks
from ggdtrack.model import NNModelGraphresPerConnection

@click.command()
@click.option("--dataset", default="Duke", help="Dataset loader class (Duke or VisDrone)")
@click.option("--datadir", default="data", help="Directory into which the Duke dataset will be downloaded")
@click.option("--cachedir", default="cachedir", help="Directory into which intermediate results are cached between runs")
@click.option("--minimal-confidence", default=None, type=float, help="Minimal confidense of detection to consider")
def main(dataset, datadir, cachedir, minimal_confidence):
    model = NNModelGraphresPerConnection()
    dataset = eval(dataset)(datadir, cachedir=cachedir, default_min_conf=minimal_confidence)
    prep_eval_gt_tracks(dataset, model, 'eval', threads=1)
    res, res_int = eval_prepped_tracks(dataset, 'eval')
    open(os.path.join(dataset.logdir, "eval_results_gt.txt"), "w").write(res)
    open(os.path.join(dataset.logdir, "eval_results_gt_int.txt"), "w").write(res_int)

if __name__ == '__main__':
    main()