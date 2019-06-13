import os
from glob import glob
from queue import Empty
from random import shuffle, seed
from shutil import rmtree
import time
import numpy as np

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

from ggdtrack.graph_diff import GraphDiffList, make_ggd_batch
from ggdtrack.klt_det_connect import graph_names
from ggdtrack.lptrack import show_tracks, interpolate_missing_detections, lp_track_weights
from ggdtrack.utils import default_torch_device, promote_graph, demote_graph, single_example_passthrough, \
    RandomSubset
from ggdtrack.eval import EvalGtGraphs

import torch.multiprocessing as multiprocessing


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


def train_graphres_minimal(dataset, model, device=default_torch_device, limit=None, epochs=10,
                           resume=False, mean_from=None,
                           batch_size=256, learning_rate=1e-3, max_time=np.inf, save_every=None,
                           max_worse_eval_epochs=float('Inf'), train_amount=None, eval_amount=None):
    logdir = dataset.logdir
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if resume:
        fn = sorted(glob("%s/???_snapshot.pyt" % (logdir)))[-1]
        snapshot = torch.load(fn)
        model.load_state_dict(snapshot['model_state'])
        optimizer.load_state_dict(snapshot['optimizer_state'])
        start_epoch = snapshot['epoch'] + 1
    else:
        start_epoch = 0

    if logdir != resume:
        if os.path.exists(logdir):
            rmtree(logdir)
        os.makedirs(logdir)

    if limit is not None:
        assert train_amount is None
        assert eval_amount is None
        train_amount = eval_amount = limit


    train_data = GraphDiffList(os.path.join(dataset.cachedir, "minimal_graph_diff", "%s_%s_train" % (dataset.name, model.feature_name)), model, "r", lazy=True)
    eval_data = GraphDiffList(os.path.join(dataset.cachedir, "minimal_graph_diff", "%s_%s_eval" % (dataset.name, model.feature_name)), model, "r", lazy=True)
    if train_amount is not None:
        train_data = RandomSubset(train_data, train_amount * len(train_data))
    if eval_amount is not None:
        eval_data = RandomSubset(eval_data, eval_amount * len(eval_data))
    train_loader = DataLoader(train_data, batch_size, pin_memory=True, collate_fn=make_ggd_batch, num_workers=6, shuffle=True)
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

    save_count = 0
    start_time = last_save = time.time()
    best_epoch = best_accuracy = -1
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
            if save_every and time.time() - last_save > save_every:
                last_save = time.time()
                torch.save(model.state_dict(), os.path.join(logdir, "model_%.4d.pyt" % save_count))
                save_count += 1

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

        if eval_acc >= best_accuracy:
            best_accuracy = eval_acc
            best_epoch = epoch

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

        if epoch - best_epoch >= max_worse_eval_epochs:
            break

def train_frossard(dataset, logdir, model, mean_from=None, device=default_torch_device, limit=None, epochs=1000,
                   resume_from=None, save_every=None):

    if mean_from is None and resume_from is None:
        raise NotImplementedError

    model.to(device)
    optimizer = optim.Adam(model.parameters(), 1e-3)

    if resume_from is not None:
        if os.path.isdir(resume_from):
            fn = sorted(glob("%s/snapshot_???.pyt" % (resume_from)))[-1]
        else:
            fn = resume_from
        print("Resuming from", fn)
        snapshot = torch.load(fn)
        if isinstance(snapshot, dict) and 'model_state' in snapshot:
            model.load_state_dict(snapshot['model_state'])
            # optimizer.load_state_dict(snapshot['optimizer_state'])
            start_epoch = snapshot['epoch'] + 1
        else:
            model.load_state_dict(snapshot)
            start_epoch = 0
        model.to(device)
    else:
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

    if logdir != resume_from:
        if os.path.exists(logdir):
            rmtree(logdir)
        os.makedirs(logdir)

    entries = graph_names(dataset, "train")
    if limit is not None:
        shuffle(entries)
        entries = entries[:limit]

    threads = multiprocessing.cpu_count() - 2

    train_data = EvalGtGraphs(dataset, entries, '-%s-eval_graph' % model.feature_name)
    train_loader = DataLoader(train_data, 1, True, collate_fn=single_example_passthrough, num_workers=threads)

    def worker(work):
        graph, connection_weights, detection_weights, entry_weight, connection_batch, detection_weight_features = work
        promote_graph(graph)
        lp_track_weights(graph, connection_weights, detection_weights, entry_weight, add_gt_hamming=True)
        demote_graph(graph)
        return graph, connection_batch, detection_weight_features

    # lp_tracker_pool = WorkerPool(threads, worker)
    lp_tracker_pool = None

    save_count = 0
    last_save = time.time()
    epoch_hamming_distance = batch_count = 0
    epoch = start_epoch
    while True:
    #     shuffle(entries)
        for graph, detection_weight_features, connection_batch in train_loader:
        # for graph, detection_weight_features, connection_batch in train_data:

            model.eval()
            connection_weights = model.connection_batch_forward(connection_batch.to(device))
            detection_weights = model.detection_model(detection_weight_features.to(device))

            if lp_tracker_pool is not None:
                lp_tracker_pool.put((graph, connection_weights.detach().cpu(), detection_weights.detach().cpu(), model.entry_weight, connection_batch, detection_weight_features))
            else:
                promote_graph(graph)
                lp_track_weights(graph, connection_weights, detection_weights, model.entry_weight, add_gt_hamming=True)

            while True:
                if lp_tracker_pool is not None:
                    try:
                        graph, connection_batch, detection_weight_features = lp_tracker_pool.get(block=False)
                    except Empty:
                        break
                    promote_graph(graph)
                if not graph:
                    continue


                if batch_count >= len(train_data):
                    snapp = {
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'train_hamming': epoch_hamming_distance,
                    }
                    torch.save(snapp, os.path.join(logdir, "snapshot_%.3d.pyt" % epoch))
                    print('%3d Hamming:' % epoch, epoch_hamming_distance)
                    epoch_hamming_distance = batch_count = 0
                    epoch += 1
                    if epoch  >= start_epoch + epochs:
                        return
                    continue
                batch_count += 1

                if save_every and time.time() - last_save > save_every:
                    last_save = time.time()
                    torch.save(model.state_dict(), os.path.join(logdir, "model_%.4d.pyt" % save_count))
                    save_count += 1


                # interpolate_missing_detections(tracks)
                # show_tracks(scene, tracks, gt_graph_frames)

                model.train()
                optimizer.zero_grad()

                hamming_distance_present = hamming_distance_entry = hamming_distance_connect = loss = 0
                connection_weights = model.connection_batch_forward(connection_batch.to(device))
                detection_weights = model.detection_model(detection_weight_features.to(device))
                for det in graph:
                    assert len(det.next) == len(det.outgoing)
                    loss += (det.present.value - det.gt_present) * detection_weights[det.index]
                    loss += (det.entry.value - det.gt_entry) * model.entry_weight_parameter
                    loss += sum(connection_weights[i] * (v.value - gt)
                                for v, i, gt in zip(det.outgoing, det.weight_index, det.gt_next))
                    hamming_distance_present += det.present.value != det.gt_present
                    hamming_distance_entry += det.entry.value != det.gt_entry
                    hamming_distance_connect += sum(v.value != gt for v, gt in zip(det.outgoing, det.gt_next))
                epoch_hamming_distance += hamming_distance_present + hamming_distance_entry + hamming_distance_connect

                # print(loss.item(), hamming_distance_present, hamming_distance_entry, hamming_distance_connect)
                loss.backward()
                # print(model.entry_weight_parameter, model.entry_weight_parameter.grad, hamming_distance_entry)
                optimizer.step()

                if lp_tracker_pool is None:
                    break

    # writer = SummaryWriter(logdir)


if __name__ == '__main__':
    from ggdtrack.duke_dataset import Duke
    from ggdtrack.model import NNModelGraphresPerConnection
    from ggdtrack.eval import prep_eval_graphs, EvalGtGraphs

    t0 = time.time()

    dataset = Duke('data')
    # train_graphres_minimal(dataset, "logdir", NNModelGraphresPerConnection())
    train_graphres_minimal(dataset, "cachedir/logdir", NNModelGraphresPerConnection(), save_every=1)


    # train_frossard(dataset, "cachedir/logdir_fossard", NNModelGraphresPerConnection(), mean_from="cachedir/logdir/snapshot_009.pyt")
    seed(42)
    # train_frossard(dataset, "cachedir/logdir_fossard", NNModelGraphresPerConnection(), resume_from="cachedir/logdir_fossard", limit=1)

    # train_frossard(dataset, "cachedir/logdir_fossard2", NNModelGraphresPerConnection(), mean_from="cachedir/logdir/snapshot_009.pyt", limit=1)
    # train_frossard(dataset, "cachedir/logdir_fossard2", NNModelGraphresPerConnection(), resume_from="cachedir/logdir/snapshot_009.pyt", epochs=10)
    print(time.time() - t0)

