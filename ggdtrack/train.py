import os
from glob import glob
from shutil import rmtree
import time
import numpy as np

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

from ggdtrack.graph_diff import GraphDiffList, make_ggd_batch
from ggdtrack.utils import default_torch_device

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


def train_graphres_minimal(dataset, logdir, model, device=default_torch_device, limit=None, epochs=10,
                           resume=False, mean_from=None,
                           batch_size=256, learning_rate=1e-3, max_time=np.inf):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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


    train_data = GraphDiffList("cachedir/minimal_graph_diff/%s_%s_train" % (dataset.name, model.feature_name), model, "r", lazy=True)
    eval_data = GraphDiffList("cachedir/minimal_graph_diff/%s_%s_eval" % (dataset.name, model.feature_name), model, "r", lazy=True)
    if limit:
        n = int(limit * len(train_data))
        train_data, _ = torch.utils.data.random_split(train_data, [n, len(train_data) - n])
        n = int(limit * len(eval_data))
        eval_data, _ = torch.utils.data.random_split(eval_data, [n, len(eval_data) - n])
    train_loader = DataLoader(train_data, batch_size, pin_memory=True, collate_fn=make_ggd_batch, num_workers=6)
    eval_loader = DataLoader(eval_data, batch_size, pin_memory=True, collate_fn=make_ggd_batch, num_workers=6)

    writer = SummaryWriter(logdir, comment="_bs=%d_lr=1e%4.1f" % (batch_size, np.log10(learning_rate)))

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
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            det_stats.extend(batch.detections)
            edge_klt_stats.extend(batch.klt_data)
            edge_long_stats.extend(batch.long_data)
            if i > 100: # We should have a good estimate by now
                break
        model.detection_model.mean = det_stats.mean
        model.detection_model.std = torch.sqrt(det_stats.var)
        model.edge_model.klt_model.mean = edge_klt_stats.mean
        model.edge_model.klt_model.std = torch.sqrt(edge_klt_stats.var)
        model.edge_model.long_model.mean = edge_long_stats.mean
        model.edge_model.long_model.std = torch.sqrt(edge_long_stats.var)

    start_time = time.time()
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
            if time.time() - start_time > max_time:
                return
        train_acc = correct / examples
        train_loss = total_loss / examples

        model.eval()
        correct = examples = total_loss = 0
        for batch in eval_loader:
            batch = batch.to(device)
            diffs = model.ggd_batch_forward(batch)
            examples += len(diffs)
            correct += (diffs>0).sum().item()
            labels = torch.ones_like(diffs)
            loss = criterion(diffs, labels)
            total_loss += loss.item()
        eval_acc = correct / examples
        eval_loss = total_loss / examples

        loss = total_loss / batches
        print('%3d Loss: %9.6f, Train Accuracy: %9.6f %%, Eval Accuracy: %9.6f %%' % (epoch, train_loss, 100 * train_acc, 100 * eval_acc))
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Eval', eval_loss, epoch)
        writer.add_scalar('Accuracy/Train', 100 * train_acc, epoch)
        writer.add_scalar('Accuracy/Eval', 100 * eval_acc, epoch)

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