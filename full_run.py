from ggdtrack.duke_dataset import Duke
from ggdtrack.klt_det_connect import prep_training_graphs

dataset = Duke("/mnt/storage/duke")
dataset.download()
dataset.prepare()

prep_training_graphs(dataset)
