import os
from glob import glob
from random import shuffle
from shutil import rmtree
import time
import numpy as np

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ggdtrack.dataset import ground_truth_tracks
from ggdtrack.graph_diff import GraphDiffList, make_ggd_batch, split_track_on_missing_edge
from ggdtrack.klt_det_connect import graph_names
from ggdtrack.lptrack import lp_track, show_tracks, interpolate_missing_detections
from ggdtrack.utils import default_torch_device, load_graph, promote_graph

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
        mean_model.load_state_dict(torch.load(mean_from)['model_state'])
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

def train_frossard(dataset, logdir, model, mean_from=None, device=default_torch_device, limit=None, epochs=1000, resume_from=None):

    if mean_from is None and resume_from is None:
        raise NotImplementedError

    optimizer = optim.Adam(model.parameters(), 1e-5)

    if resume_from is not None:
        fn = sorted(glob("%s/snapshot_???.pyt" % (resume_from)))[-1]
        print("Resuming from", fn)
        snapshot = torch.load(fn)
        model.load_state_dict(snapshot['model_state'])
        optimizer.load_state_dict(snapshot['optimizer_state'])
        start_epoch = snapshot['epoch'] + 1
    else:
        if os.path.exists(logdir):
            rmtree(logdir)
        os.makedirs(logdir)
        start_epoch = 0
        for t in model.parameters():
            torch.nn.init.normal_(t, 0, 1e-3)

        mean_model = model.__class__()
        mean_model.load_state_dict(torch.load(mean_from)['model_state'])
        model.detection_model.mean = mean_model.detection_model.mean
        model.detection_model.std = mean_model.detection_model.std
        model.edge_model.klt_model.mean = mean_model.edge_model.klt_model.mean
        model.edge_model.klt_model.std = mean_model.edge_model.klt_model.std
        model.edge_model.long_model.mean = mean_model.edge_model.long_model.mean
        model.edge_model.long_model.std = mean_model.edge_model.long_model.std
        model.to(device)

    entries = graph_names(dataset, "train")
    if limit is not None:
        shuffle(entries)
        entries = entries[:limit]

    for epoch in range(start_epoch, start_epoch + epochs):
        shuffle(entries)
        epoch_hamming_distance = 0
        for name, cam in tqdm(entries, "Epoch %s" % epoch, disable=True):
            scene = dataset.scene(cam)

            model.eval()
            graph, detection_weight_features, connection_batch = torch.load(name + '-%s-eval_graph' % model.feature_name)
            promote_graph(graph)
            detection_weight_features = detection_weight_features.to(device)
            connection_batch = connection_batch.to(device)

            gt_tracks, graph_frames = ground_truth_tracks(scene.ground_truth(), graph)
            for det in graph:
                det.gt_entry = 0.0
                det.gt_present = 0.0 if det.track_id is None else 1.0
                det.gt_next = [0.0] * len(det.next)
            for tr in gt_tracks:
                tr[0].gt_entry = 1.0
                prv = None
                for det in tr:
                    if prv is not None:
                        prv.gt_next[prv.next.index(det)] = 1.0

            tracks = lp_track(graph, connection_batch, detection_weight_features, model) #, add_gt_hamming=True)
            # interpolate_missing_detections(tracks)
            # show_tracks(scene, tracks)

            model.train()
            optimizer.zero_grad()

            hamming_distance = loss = 0
            connection_weights = model.connection_batch_forward(connection_batch)
            detection_weights = model.detection_model(detection_weight_features)
            for det in graph:
                assert len(det.next) == len(det.outgoing)
                loss += (det.present.value - det.gt_present) * detection_weights[det.index]
                loss += (det.entry.value - det.gt_entry) * model.entry_weight_parameter
                loss += sum(connection_weights[i] * (v.value - gt)
                            for v, i, gt in zip(det.outgoing, det.weight_index, det.gt_next))
                hamming_distance += det.present.value != det.gt_present
                hamming_distance += det.entry.value != det.gt_entry
                hamming_distance += sum(v.value != gt for v, gt in zip(det.outgoing, det.gt_next))
            # print(hamming_distance)
            epoch_hamming_distance += hamming_distance

            loss.backward()
            optimizer.step()

        snapp = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss,
            'train_hamming': epoch_hamming_distance,
        }
        torch.save(snapp, os.path.join(logdir, "snapshot_%.3d.pyt" % epoch))
        print('Hamming:', epoch_hamming_distance)


    # writer = SummaryWriter(logdir)


if __name__ == '__main__':
    from ggdtrack.duke_dataset import Duke
    from ggdtrack.model import NNModelGraphresPerConnection
    from ggdtrack.eval import prep_eval_graphs

    dataset = Duke('data')
    # train_graphres_minimal(dataset, "logdir", NNModelGraphresPerConnection())

    prep_eval_graphs(dataset, NNModelGraphresPerConnection(), parts=["train"])
    # train_frossard(dataset, "cachedir/logdir_fossard", NNModelGraphresPerConnection(), mean_from="cachedir/logdir/snapshot_009.pyt")
    train_frossard(dataset, "cachedir/logdir_fossard", NNModelGraphresPerConnection(), resume_from="cachedir/logdir")
