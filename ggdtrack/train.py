import os
from collections import defaultdict
from glob import glob
from shutil import rmtree
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ggdtrack.graph_diff import GraphDiffList, make_ggd_batch

if True: # Patch pytorch
    string_classes = torch._six.string_classes
    import collections.abc
    container_abcs = collections.abc

    def pin_memory_batch(batch):
        if isinstance(batch, torch.Tensor):
            return batch.pin_memory()
        elif isinstance(batch, string_classes):
            return batch
        elif isinstance(batch, container_abcs.Mapping):
            return {k: pin_memory_batch(sample) for k, sample in batch.items()}
        elif isinstance(batch, tuple):
            return batch.__class__(*[pin_memory_batch(sample) for sample in batch])
        elif isinstance(batch, container_abcs.Sequence):
            return [pin_memory_batch(sample) for sample in batch]
        else:
            return batch

    torch.utils.data.dataloader.pin_memory_batch = pin_memory_batch

class NormStats:
    def __init__(self):
        self.n = 0
        self.sa = 0
        self.sa2 = 0

    def extend(self, lst):
        self.n += len(lst)
        self.sa += sum(lst)
        self.sa2 += sum(lst**2)

    @property
    def mean(self):
        return self.sa / self.n

    @property
    def var(self):
        return self.sa2 / self.n - self.sa**2 / self.n**2


def train_graphres_minimal(dataset, logdir, model, device=torch.device('cpu'), limit=None, epochs=10,
                           resume=False, run_eval=True, mean_from=None):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    if resume:
        fn = sorted(glob("%s/???_snapshot.pyt" % (logdir)))[-1]
        snapshot = torch.load(fn)
        model.load_state_dict(snapshot['model_state'])
        optimizer.load_state_dict(snapshot['optimizer_state'])
        start_epoch = snapshot['epoch'] + 1
    else:
        if os.path.exists(logdir):
            rmtree(logdir)
        os.makedirs(logdir)
        start_epoch = 0


    train_data = GraphDiffList("minimal_graph_diff/%s_%s_train" % (dataset.name, model.feature_name), model, "r")
    train_loader = DataLoader(train_data, 256, pin_memory=True, collate_fn=make_ggd_batch, num_workers=6)
    if run_eval:
        eval_data = GraphDiffList("minimal_graph_diff/%s_%s_eval" % (dataset.name, model.feature_name), model, "r")
        eval_loader = DataLoader(eval_data, 256, pin_memory=True, collate_fn=make_ggd_batch, num_workers=6)

    if mean_from is not None:
        mean_model = model.__class__()
        mean_model.load_state_dict(torch.load(mean_from))
        model.detection_model.mean = mean_model.detection_model.mean
        model.detection_model.std = mean_model.detection_model.std
        model.edge_model.klt_model.mean = mean_model.edge_model.klt_model.mean
        model.edge_model.klt_model.std = mean_model.edge_model.klt_model.std
        model.edge_model.long_model.mean = mean_model.edge_model.long_model.mean
        model.edge_model.long_model.std = mean_model.edge_model.long_model.std
        model.to(device)
    elif not resume:
        model.to(device)
        det_stats = NormStats()
        edge_klt_stats = NormStats()
        edge_long_stats = NormStats()
        for batch in train_loader:
            det_stats.extend(batch.detections)
            edge_klt_stats.extend(batch.klt_data)
            edge_long_stats.extend(batch.long_data)
        model.detection_model.mean = det_stats.mean
        model.detection_model.std = torch.sqrt(det_stats.var)
        model.edge_model.klt_model.mean = torch.tensor(edge_klt_stats.mean, dtype=torch.float32)
        model.edge_model.klt_model.std = torch.sqrt(torch.tensor(edge_klt_stats.var, dtype=torch.float32))
        model.edge_model.long_model.mean = torch.tensor(edge_long_stats.mean, dtype=torch.float32)
        model.edge_model.long_model.std = torch.sqrt(torch.tensor(edge_long_stats.var, dtype=torch.float32))

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = batches = examples = correct = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            diffs = model.ggd_batch_forward(batch)
            examples += len(diffs)
            correct += (diffs>0).sum().item()
            labels = torch.ones_like(diffs)
            loss = criterion(diffs, labels)
            total_loss += loss.item()
            batches += 1
            loss.backward()
            optimizer.step()
        train_acc = correct / examples

        if run_eval:
            model.eval()
            correct = examples = 0
            for batch in eval_loader:
                batch = batch.to(device)
                diffs = model.ggd_batch_forward(batch)
                examples += len(diffs)
                correct += (diffs>0).sum().item()
            eval_acc = correct / examples
        else:
            eval_acc = -1

        loss = total_loss / batches
        print('%3d Loss: %9.6f, Train Accuracy: %9.6f %%, Eval Accuracy: %9.6f %%' % (epoch, loss, 100 * train_acc, 100 * eval_acc))
        snapp = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss,
            'train_acc': train_acc,
            'eval_acc': eval_acc,
        }
        torch.save(snapp, os.path.join(logdir, "snapshot_%.3d.pyt" % epoch))

if __name__ == '__main__':
    from ggdtrack.duke_dataset import Duke
    from ggdtrack.model import NNModelGraphresPerConnection
    train_graphres_minimal(Duke('/home/hakan/src/duke'), "logdir", NNModelGraphresPerConnection())