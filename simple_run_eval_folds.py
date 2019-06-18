from ggdtrack.eval import eval_prepped_tracks_joined
from ggdtrack.mot16_dataset import Mot16

print('Linear')
datasets = [Mot16("data", fold=fold) for fold in range(4)]
eval_prepped_tracks_joined(datasets)

print('NNModelSimpleMLP1')
datasets = [Mot16("data", fold=fold) for fold in range(4)]
for d in datasets:
    d.logdir += '_NNModelSimpleMLP1'
eval_prepped_tracks_joined(datasets)

print('NNModelSimpleMLP2')
datasets = [Mot16("data", fold=fold) for fold in range(4)]
for d in datasets:
    d.logdir += '_NNModelSimpleMLP2'
eval_prepped_tracks_joined(datasets)
