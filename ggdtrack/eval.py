import os
from collections import defaultdict
from collections import namedtuple
from glob import glob
from random import shuffle
from tempfile import TemporaryDirectory
from time import sleep

import motmetrics
import torch
import numpy as np
from lap._lapjv import lapjv
from shapely.geometry import Point
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from tqdm import tqdm

from ggdtrack.dataset import ground_truth_tracks
from ggdtrack.graph_diff import split_track_on_missing_edge
from ggdtrack.klt_det_connect import graph_names
from ggdtrack.lptrack import lp_track, interpolate_missing_detections, lp_track_weights
from ggdtrack.mmap_array import VarHMatrixList
from ggdtrack.utils import load_pickle, save_torch, parallel, default_torch_device, save_pickle, parallel_run, \
    load_graph, demote_graph, promote_graph, single_example_passthrough, WorkerPool, load_json


class ConnectionBatch(namedtuple('ConnectionBatch', ['klt_idx', 'klt_data', 'long_idx', 'long_data'])):
    def to(self, device):
        return ConnectionBatch(*[x.to(device) for x in self])

def prep_eval_graph_worker(args):
    model, graph_name, cachedir = args
    ofn = graph_name + '-%s-eval_graph' % model.feature_name
    if os.path.exists(ofn):
        return ofn
    graph = load_graph(graph_name)

    with TemporaryDirectory(dir=cachedir, prefix="tmp_eval", suffix="_mmaps") as tmpdir:
        detection_weight_features = []
        edge_weight_features_klt = VarHMatrixList(tmpdir, 'klt_data', 'klt_index', model.klt_feature_length)
        edge_weight_features_long = VarHMatrixList(tmpdir, 'long_data', 'long_index', model.long_feature_length)
        for idx, d in enumerate(graph):
            d.index = idx
            d.next = []
            d.weight_index = []
            for n, weight_data in d.next_weight_data.items():
                d.next.append(n)
                klt, long = model.connection_weight_feature(d, n)
                d.weight_index.append(len(edge_weight_features_klt))
                edge_weight_features_klt.append(klt)
                edge_weight_features_long.append(long)
            d.prev = list(d.prev)
            detection_weight_features.append(model.detecton_weight_feature(d))
            if hasattr(d, 'max_intra_iou'):
                del d.max_intra_iou
            if hasattr(d, 'max_intra_ioa'):
                del d.max_intra_ioa
            if hasattr(d, 'post_vs'):
                del d.post_vs
            if hasattr(d, 'pre_vs'):
                del d.pre_vs
            if hasattr(d, 'next_weight_data'):
                del d.next_weight_data
        assert len(edge_weight_features_klt) == len(edge_weight_features_long)
        detection_weight_features = torch.tensor(detection_weight_features)
        connection_batch = ConnectionBatch(torch.tensor(edge_weight_features_klt.index.data),
                                           torch.tensor(edge_weight_features_klt.data),
                                           torch.tensor(edge_weight_features_long.index.data),
                                           torch.tensor(edge_weight_features_long.data))

    demote_graph(graph)
    save_torch((graph, detection_weight_features, connection_batch), ofn)
    return ofn

def prep_eval_graphs(dataset, model, threads=None, parts=["eval", "test"]):
    jobs = [(model, name, dataset.cachedir) for part in parts for name, cam in graph_names(dataset, part)]
    parallel_run(prep_eval_graph_worker, jobs, threads, "Prepping eval graphs")

def prep_eval_tracks_worker(args):
    model, name, device, logdir = args
    ofn = os.path.join(logdir, "tracks", os.path.basename(name))

    graph, detection_weight_features, connection_batch = torch.load(name + '-%s-eval_graph' % model.feature_name)
    promote_graph(graph)
    detection_weight_features = detection_weight_features.to(device)
    connection_batch = connection_batch.to(device)
    tracks = lp_track(graph, connection_batch, detection_weight_features, model)
    for tr in tracks:
        for det in tr:
            det.__dict__ = {}
    save_pickle(tracks, ofn)

    return ofn

def prep_eval_tracks(dataset, model, part='eval', device=default_torch_device, threads=None, limit=None):
    logdir = dataset.logdir
    if logdir is not None:
        if os.path.isfile(logdir):
            fn = logdir
        else:
            fn = sorted(glob("%s/snapshot_???.pyt" % logdir))[-1]
        model.load_state_dict(torch.load(fn)['model_state'])
    model.eval()
    model.to(device)

    if limit is None:
        limit = graph_names(dataset, part)
    jobs = [(model, name, device, dataset.logdir) for name, cam in limit]
    shuffle(jobs)

    parallel_run(prep_eval_tracks_worker, jobs, threads, "Prepping eval tracks for %s" % part)

class MotMetrics:
    def __init__(self, overall=False, iou_threshold=0.5):
        self.accumulators = []
        self.names = []
        self.overall = overall
        self.iou_threshold = iou_threshold

    def add(self, tracks, gt_frames, name='test', frame_range=None):
        assert frame_range is not None
        if not tracks:
            return
        acc = motmetrics.MOTAccumulator()
        self.accumulators.append(acc)
        self.names.append(name.split('.')[0][-40:])
        self.update(tracks, gt_frames, frame_range)

    def update(self, tracks, gt_frames, frame_range):
        track_frames = defaultdict(list)
        for i, tr in enumerate(tracks):
            for det in tr:
                det.track_id = i
                track_frames[det.frame].append(det)
        acc = self.accumulators[-1]
        for f in frame_range:
            gt = [d for d in gt_frames[f]]
            id_gt = [d.id for d in gt]
            id_res = [d.track_id for d in track_frames[f]]
            dists = np.array([[1 - d1.iou(d2) for d2 in track_frames[f]] for d1 in gt])
            dists[dists > self.iou_threshold] = np.nan
            acc.update(id_gt, id_res, dists, f)

    def summary(self):
        mh = motmetrics.metrics.create()
        # metrics = ['idf1', 'idp', 'idr', 'mota', 'motp', 'num_frames']
        # metrics = ['idf1', 'idp', 'idr', 'mota', 'motp', 'num_frames', 'num_switches', 'num_fragmentations', 'mostly_tracked', 'partially_tracked', 'mostly_lost']
        # metrics = ['mota', 'num_switches', 'num_fragmentations', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_frames']
        metrics = ['mota', 'num_switches', 'num_misses', 'num_false_positives', 'num_objects', 'num_fragmentations', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_frames']
        summary= mh.compute_many(self.accumulators, metrics=metrics,
                               names=self.names, generate_overall=self.overall)
        return motmetrics.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=motmetrics.io.motchallenge_metric_names
        )


def filter_out_non_roi_dets(scene, tracks):
    roi = Polygon(scene.roi())
    tracks[:] = [[det for det in tr if roi.contains(Point(det.cx, det.bottom))]
                 for tr in tracks]
    tracks[:] = [tr for tr in tracks if tr]


def eval_prepped_tracks(dataset, part='eval'):
    return eval_prepped_tracks_folds([dataset], part)


def eval_prepped_tracks_folds(datasets, part='eval'):
    metrics = MotMetrics(True)
    metrics_int = MotMetrics(True)
    for dataset in datasets:
        for name, cam in tqdm(graph_names(dataset, part), 'Evaluating tracks'):
            meta = load_json(name + '-meta.json')
            frame_range = range(meta['first_frame'], meta['first_frame'] + meta['length'])
            scene = dataset.scene(cam)
            gt_frames = scene.ground_truth()
            tracks_name = os.path.join(dataset.logdir, "tracks", os.path.basename(name))
            tracks = load_pickle(tracks_name)
            filter_out_non_roi_dets(scene, tracks)

            metrics.add(tracks, gt_frames, name, frame_range)
            interpolate_missing_detections(tracks)
            metrics_int.add(tracks, gt_frames, name + 'i', frame_range)

    res = metrics.summary()
    res_int = metrics_int.summary()
    print("Result")
    print(res)
    print("\nResult interpolated")
    print(res_int)
    return res, res_int


def eval_prepped_tracks_csv(dataset, part='eval'):
    logdir = dataset.logdir
    base = '%s/result_%s_%s' % (logdir, dataset.name, part)
    os.makedirs(base, exist_ok=True)
    os.makedirs(base + '_int', exist_ok=True)
    prev_cam = prev_track_frames = all_tracks = prev_tracks = None
    entries = list(sorted(graph_names(dataset, part), key=lambda e: (e[1], e[0]))) + [(None, None)]
    for name, cam in tqdm(entries, 'Evaluating tracks CSV'):
        if cam != prev_cam:
            if all_tracks is not None:
                csv_eval, csv_submit = make_duke_csv(all_tracks, prev_cam)
                np.savetxt('%s/%s_eval.txt' % (base, prev_cam), csv_eval, delimiter=',', fmt='%d')
                np.savetxt('%s/%s_submit.txt' % (base, prev_cam), csv_submit, delimiter=',', fmt='%d')

                interpolate_missing_detections(all_tracks)
                csv_eval, csv_submit = make_duke_csv(all_tracks, prev_cam)
                np.savetxt('%s_int/%s_eval.txt' % (base, prev_cam), csv_eval, delimiter=',', fmt='%d')
                np.savetxt('%s_int/%s_submit.txt' % (base, prev_cam), csv_submit, delimiter=',', fmt='%d')

            prev_track_frames = None
            prev_cam = cam
        if name is None:
            break

        tracks_name = os.path.join(dataset.logdir, "tracks", os.path.basename(name))
        tracks = load_pickle(tracks_name)
        track_frames = defaultdict(list)
        for i, tr in enumerate(tracks):
            for det in tr:
                det.idx = i
                track_frames[det.frame].append(det)

        if prev_track_frames is not None:
            if len(track_frames.keys()) and len(prev_track_frames.keys()):
                overlap = range(min(track_frames.keys()), max(prev_track_frames.keys())+1)
                counts = np.zeros((len(prev_tracks), len(tracks)))
                for f in overlap:
                    for prv in prev_track_frames[f]:
                        for nxt in track_frames[f]:
                            if prv.left == nxt.left and prv.right == nxt.right and prv.top == nxt.top and prv.bottom == nxt.bottom:
                                counts[prv.idx, nxt.idx] += 1

                cost, _, nxt_matches = lapjv(-counts, extend_cost=True)
                assert len(nxt_matches) == len(tracks)
                for i in range(len(tracks)):
                    j = nxt_matches[i]
                    if counts[j][i] > 0:
                        prev_tracks[j] += tracks[i]
                        tracks[i] = prev_tracks[j]
                    else:
                        all_tracks.append(tracks[i])
            else:
                all_tracks.extend(tracks)
        else:
            all_tracks = tracks
        prev_track_frames = track_frames
        prev_tracks = tracks

def make_duke_csv(all_tracks, prev_cam):
    csv_eval, csv_submit = [], []
    for track_id, tr in enumerate(all_tracks):
        tr.sort(key=lambda d: d.frame)
        prv = -1
        for det in tr:
            if det.frame > prv:
                csv_eval.append([det.frame, track_id, det.left, det.top, det.width, det.height, -1, -1])
                csv_submit.append([prev_cam, track_id, det.frame, det.left, det.top, det.width, det.height])
            prv = det.frame
    return csv_eval, csv_submit

class EvalGtGraphs:
    def __init__(self, dataset, entries, suffix):
        self.dataset = dataset
        self.entries = entries
        self.suffix = suffix

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        name, cam = self.entries[item]
        scene = self.dataset.scene(cam)

        graph, detection_weight_features, connection_batch = torch.load(name + self.suffix)
        promote_graph(graph)

        gt_tracks, gt_graph_frames = ground_truth_tracks(scene.ground_truth(), graph)
        gt_tracks = split_track_on_missing_edge(gt_tracks)

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
                prv = det

        demote_graph(graph)
        return graph, detection_weight_features, connection_batch

def eval_hamming_worker(work):
    graph, connection_weights, detection_weights, entry_weight = work
    promote_graph(graph)
    lp_track_weights(graph, connection_weights, detection_weights, entry_weight, add_gt_hamming=True)

    hamming = 0
    for det in graph:
        assert len(det.next) == len(det.outgoing)
        hamming += det.present.value != det.gt_present
        hamming += det.entry.value != det.gt_entry
        hamming += sum(v.value != gt for v, gt in zip(det.outgoing, det.gt_next))
    return hamming
eval_hamming_worker.pool = None

def eval_hamming(dataset, logdir, model, device=default_torch_device):
    if logdir is not None:
        if os.path.isfile(logdir):
            fn = logdir
        else:
            fn = sorted(glob("%s/snapshot_???.pyt" % logdir))[-1]
        model.load_state_dict(torch.load(fn)['model_state'])
    model.eval()
    model.to(device)

    threads = torch.multiprocessing.cpu_count() - 2
    entries = graph_names(dataset, "train")
    shuffle(entries)
    entries = entries[:20]
    train_data = EvalGtGraphs(dataset, entries, '-%s-eval_graph' % model.feature_name)
    train_loader = DataLoader(train_data, 1, True, collate_fn=single_example_passthrough, num_workers=threads)


    if eval_hamming_worker.pool is None:
        eval_hamming_worker.pool = WorkerPool(threads, eval_hamming_worker, output_queue_len=None)


    model.eval()
    hamming = 0
    for graph, detection_weight_features, connection_batch in train_loader:
        connection_weights = model.connection_batch_forward(connection_batch.to(device))
        detection_weights = model.detection_model(detection_weight_features.to(device))
        eval_hamming_worker.pool.put((graph, connection_weights.detach().cpu(), detection_weights.detach().cpu(), model.entry_weight))
        # hamming += worker((graph, connection_weights.detach().cpu(), detection_weights.detach().cpu(), model.entry_weight))

    for _ in range(len(train_data)):
        hamming += eval_hamming_worker.pool.get()
    return hamming

def prep_eval_gt_tracks_worker(args):
    model, name, scene, logdir, split_on_no_edge = args
    ofn = os.path.join(logdir, "tracks", os.path.basename(name))

    graph, detection_weight_features, connection_batch = torch.load(name + '-%s-eval_graph' % model.feature_name)
    promote_graph(graph)
    tracks, gt_graph_frames = ground_truth_tracks(scene.ground_truth(), graph)
    if split_on_no_edge:
        tracks = split_track_on_missing_edge(tracks)
    for tr in tracks:
        for det in tr:
            det.__dict__ = {}
    save_pickle(tracks, ofn)

    return ofn

def prep_eval_gt_tracks(dataset, model, part='eval', threads=None, split_on_no_edge=False):
    limit = graph_names(dataset, part)
    jobs = [(model, name, dataset.scene(cam), dataset.logdir, split_on_no_edge)
            for name, cam in limit]
    shuffle(jobs)
    parallel_run(prep_eval_gt_tracks_worker, jobs, threads, "Prepping eval tracks for %s" % part)



if __name__ == '__main__':
    from ggdtrack.duke_dataset import Duke
    from ggdtrack.model import NNModelGraphresPerConnection
    from ggdtrack.visdrone_dataset import VisDrone
    # dataset = Duke('data')
    dataset = VisDrone('data')

    # prep_eval_graphs(dataset, NNModelGraphresPerConnection)
    # prep_eval_tracks(dataset, "logdir", NNModelGraphresPerConnection())
    # eval_prepped_tracks(dataset)
    # eval_prepped_tracks_csv(dataset, "cachedir/logdir", "test")
    # prep_eval_graphs(dataset, NNModelGraphresPerConnection(), parts=["eval"])
    # print(eval_hamming(dataset, "cachedir/logdir", NNModelGraphresPerConnection()))
    # prep_eval_gt_tracks(dataset, NNModelGraphresPerConnection)
    # res, res_int = eval_prepped_tracks(dataset, 'eval')

    prep_eval_tracks(dataset, "cachedir/logdir_%s" % dataset.name, NNModelGraphresPerConnection(), 'eval', threads=1)
    # eval_prepped_tracks_csv(dataset, "cachedir/logdir_%s" % dataset.name, 'eval')
