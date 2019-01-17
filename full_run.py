from ggdtrack.duke_dataset import Duke

dataset = Duke("/mnt/storage/duke/DukeMTMC/")
# dataset.download()
dataset.prepare()
