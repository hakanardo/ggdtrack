from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import prep_eval_graphs, prep_eval_tracks, eval_prepped_tracks
from ggdtrack.graph_diff import prep_minimal_graph_diffs
from ggdtrack.klt_det_connect import prep_training_graphs
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.train import train_graphres_minimal

dataset = Duke("/home/hakan/src/duke")
dataset.download()
exit()

dataset.prepare()
prep_training_graphs(dataset)

model = NNModelGraphresPerConnection()
prep_minimal_graph_diffs(dataset, model)
prep_eval_graphs(dataset, model)

train_graphres_minimal(dataset, "logdir", model)

prep_eval_tracks(dataset, "logdir", model)
eval_prepped_tracks(dataset)


