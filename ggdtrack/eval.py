import os
from collections import defaultdict
from collections import namedtuple
from glob import glob
from random import shuffle
from tempfile import TemporaryDirectory

import motmetrics
import torch
import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon
from tqdm import tqdm

from ggdtrack.klt_det_connect import graph_names
from ggdtrack.lptrack import lp_track, interpolate_missing_detections
from ggdtrack.mmap_array import VarHMatrixList
from ggdtrack.utils import load_pickle, save_torch, parallel, default_torch_device, save_pickle, parallel_run, \
    load_graph, demote_graph, promote_graph


class ConnectionBatch(namedtuple('ConnectionBatch', ['klt_idx', 'klt_data', 'long_idx', 'long_data'])):
    def to(self, device):
        return ConnectionBatch(*[x.to(device) for x in self])

def prep_eval_graph_worker(args):
    model, graph_name = args
    ofn = graph_name + '-%s-eval_graph' % model.feature_name
    if os.path.exists(ofn):
        return ofn
    graph = load_graph(graph_name)

    with TemporaryDirectory(prefix=os.path.join(os.getcwd(), "cachedir")) as tmpdir:
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
            del d.max_intra_iou
            del d.max_intra_ioa
            del d.post_vs
            del d.pre_vs
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

def prep_eval_graphs(dataset, model, threads=None):
    jobs = [(model, name) for part in ["eval", "test"] for name, cam in graph_names(dataset, part)]
    parallel_run(prep_eval_graph_worker, jobs, threads, "Prepping eval graphs")

def prep_eval_tracks_worker(args):
    model, name, device = args
    ofn = os.path.join("cachedir/tracks", os.path.basename(name))
    print(ofn)
    if not os.path.exists(ofn):
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

def prep_eval_tracks(dataset, logdir, model, part='eval', device=default_torch_device, threads=None, limit=None):
    if os.path.isfile(logdir):
        fn = logdir
    else:
        fn = sorted(glob("%s/snapshot_???.pyt" % logdir))[-1]
    model.load_state_dict(torch.load(fn)['model_state'])
    model.eval()
    model.to(device)

    if limit is None:
        limit = graph_names(dataset, part)
    jobs = [(model, name, device) for name, cam in limit]
    shuffle(jobs)

    parallel_run(prep_eval_tracks_worker, jobs, threads, "Prepping eval tracks for %s" % part)

class MotMetrics:
    def __init__(self, overall=False, iou_threshold=0.5):
        self.accumulators = []
        self.names = []
        self.overall = overall
        self.iou_threshold = iou_threshold

    def add(self, tracks, gt_frames, name='test'):
        if not tracks:
            return
        acc = motmetrics.MOTAccumulator()
        self.accumulators.append(acc)
        self.names.append(name)
        self.update(tracks, gt_frames)

    def update(self, tracks, gt_frames):
        track_frames = defaultdict(list)
        for i, tr in enumerate(tracks):
            for det in tr:
                det.track_id = i
                track_frames[det.frame].append(det)
        frames = track_frames.keys()
        frames = range(min(frames), max(frames)+1)
        acc = self.accumulators[-1]
        for f in frames:
            gt = [d for d in gt_frames[f]]
            id_gt = [d.id for d in gt]
            id_res = [d.track_id for d in track_frames[f]]
            dists = np.array([[1 - d1.iou(d2) for d2 in track_frames[f]] for d1 in gt])
            dists[dists > self.iou_threshold] = np.nan
            acc.update(id_gt, id_res, dists, f)

    def summary(self):
        mh = motmetrics.metrics.create()
        summary= mh.compute_many(self.accumulators, metrics=['idf1', 'idp', 'idr', 'mota', 'motp', 'num_frames'],
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
    metrics = MotMetrics(True)
    metrics_int = MotMetrics(True)
    for name, cam in tqdm(graph_names(dataset, part), 'Evaluating tracks'):
        scene = dataset.scene(cam)
        gt_frames = scene.ground_truth()
        tracks_name = os.path.join("cachedir/tracks", os.path.basename(name))
        tracks = load_pickle(tracks_name)
        filter_out_non_roi_dets(scene, tracks)

        metrics.add(tracks, gt_frames, name)
        interpolate_missing_detections(tracks)
        metrics_int.add(tracks, gt_frames, name + 'i')

    res = metrics.summary()
    res_int = metrics_int.summary()
    print("Result")
    print(res)
    print("\nResult interpolated")
    print(res_int)
    return res, res_int


if __name__ == '__main__':
    from ggdtrack.duke_dataset import Duke
    from ggdtrack.model import NNModelGraphresPerConnection
    dataset = Duke('/home/hakan/src/duke')
    # prep_eval_graphs(dataset, NNModelGraphresPerConnection)
    # prep_eval_tracks(dataset, "logdir", NNModelGraphresPerConnection())
    eval_prepped_tracks(dataset)
