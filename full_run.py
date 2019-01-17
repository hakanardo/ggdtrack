from ggdtrack.duke_dataset import Duke
from ggdtrack.graph_diff import prep_minimal_graph_diffs
from ggdtrack.klt_det_connect import prep_training_graphs
from ggdtrack.model import NNModelGraphresPerConnection

dataset = Duke("/mnt/storage/duke")
dataset.download()
dataset.prepare()

prep_training_graphs(dataset)
prep_minimal_graph_diffs(dataset, NNModelGraphresPerConnection)

