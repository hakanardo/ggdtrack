from ggdtrack.eval import eval_prepped_tracks_folds
from ggdtrack.mot16_dataset import Mot16

datasets = [Mot16("data", fold=fold) for fold in range(4)]
for d in datasets:
    d.logdir += '_NNModelSimple_gt'
eval_prepped_tracks_folds(datasets)