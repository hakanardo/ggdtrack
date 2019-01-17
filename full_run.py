from ggdtrack.duke_dataset import Duke
from ggdtrack.eval import prep_eval_graphs
from ggdtrack.graph_diff import prep_minimal_graph_diffs
from ggdtrack.klt_det_connect import prep_training_graphs
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.train import train_graphres_minimal

dataset = Duke("/mnt/storage/duke")
dataset.download()

dataset.prepare()
prep_training_graphs(dataset)

model = NNModelGraphresPerConnection()
prep_minimal_graph_diffs(dataset, model)
prep_eval_graphs(dataset, model)

train_graphres_minimal(dataset, "logdir", model)

