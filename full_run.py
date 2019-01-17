from ggdtrack.duke_dataset import Duke

dataset = Duke("/mnt/storage/duke")
# dataset.download()
dataset.prepare()
