from ggdtrack.eval import eval_prepped_tracks_folds
from ggdtrack.mot16_dataset import Mot16

datasets = [Mot16("data", fold=fold) for fold in range(4)]
eval_prepped_tracks_folds(datasets)