from ggdtrack.eval import eval_prepped_tracks
from ggdtrack.visdrone_dataset import VisDrone

dataset = VisDrone("data")
res, res_int = eval_prepped_tracks(dataset, 'eval')
