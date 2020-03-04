import os
from glob import glob

import click

from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import prep_eval_tracks, eval_prepped_tracks
from ggdtrack.model import NNModelGraphresPerConnection

@click.command()
@click.option("--threads", default=None, type=int, help="The number of threads to use. Default is one per CPU core.")
@click.option("--cachedir", default="cachedir", help="Directory into which intermediate results are cached between runs")
@click.option("--logdir-glob", help="Logdirs to investigate")
def main(threads, cachedir, logdir_glob):
    max_extra = 3
    dataset = Duke("data")

    for logdir in glob(logdir_glob):
        dataset.logdir = logdir
        model = NNModelGraphresPerConnection()
        fn = sorted(glob("%s/snapshot_???.pyt" % dataset.logdir))[-max_extra-1]
        prep_eval_tracks(dataset, model, 'eval', threads=1, snapshot=fn)
        res, res_int = eval_prepped_tracks(dataset, 'eval')
        open(os.path.join(dataset.logdir, "eval_results.txt"), "w").write(res)
        open(os.path.join(dataset.logdir, "eval_results_int.txt"), "w").write(res_int)

if __name__ == '__main__':
    main()

