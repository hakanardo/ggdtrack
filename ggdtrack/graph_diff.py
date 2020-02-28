import os
from collections import defaultdict, namedtuple
from glob import glob
from random import shuffle
from shutil import rmtree
from tempfile import TemporaryDirectory

import torch

from ggdtrack.dataset import ground_truth_tracks, false_positive_tracks
from ggdtrack.klt_det_connect import graph_names
from ggdtrack.mmap_array import as_database, VarHMatrixList, ScalarList
from ggdtrack.utils import parallel, save_json, save_torch, load_pickle, load_json, load_graph, save_pickle

import numpy as np


def as_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)
    return x

class GraphBatch(namedtuple('GraphBatch', ['edge', 'detection', 'entries'])):
    def __new__(cls, edge, detection, entries):
        if not isinstance(edge, torch.Tensor):
            edge = [(as_tensor(klt), as_tensor(long)) for klt, long in edge]
        if not isinstance(detection, torch.Tensor):
            detection = torch.tensor([d for d in detection], dtype=torch.float)
        if not isinstance(entries, torch.Tensor):
            entries = torch.tensor(entries, dtype=torch.float)
        return super(GraphBatch, cls).__new__(cls, edge, detection, entries)

    def size(self):
        return len(self.edge) + len(self.detection)


GraphBatchPair = namedtuple('GraphBatchPair', ['pos', 'neg', 'name'])


GraphDiffData = namedtuple('GraphDiffData', ['klt_idx', 'klt_data', 'long_idx', 'long_data', 'edge_signs',
                                             'detections', 'detection_signs', 'entry_diff'])


class GraphDiffBatch(namedtuple('GraphDiffBatch', ['klt_idx', 'klt_data', 'long_idx', 'long_data', 'edge_signs',
                                                  'detections', 'detection_signs', 'entry_diffs',
                                                  'edge_idx', 'detection_idx'])):
    def to(self, device):
        return GraphDiffBatch(*[x.to(device) for x in self])

class GraphDiffList:
    def __init__(self, db, model, mode=None, lazy=False):
        self.db = db
        self.mode = mode
        self.model = model
        if lazy:
            self.klts = None
        else:
            self.init()

    def init(self):
        db = self.db = as_database(self.db, self.mode)
        self.klts = VarHMatrixList(db, 'klt_data', 'klt_index', self.model.klt_feature_length)
        self.longs = VarHMatrixList(db, 'long_data', 'long_index', self.model.long_feature_length)
        self.edge_signs = VarHMatrixList(db, 'edge_signs_data', 'edge_signs_index', 1)
        self.edge_index = ScalarList(db, 'edge_index', int)
        if len(self.edge_index) == 0:
            self.edge_index.append(0)
        self.detections = VarHMatrixList(db, 'detection_data', 'detection_index', self.model.detecton_feature_length)
        self.detection_signs = VarHMatrixList(db, 'detection_signs_data', 'detection_signs_index', 1)
        self.entry_diff = ScalarList(db, 'entry_diff', np.float32)


    def append(self, graph_diff):
        if self.klts is None:
            self.init()
        edges = list(graph_diff.pos.edge) + list(graph_diff.neg.edge)
        self.klts.extend([e[0] for e in edges])
        self.longs.extend([e[1] for e in edges])
        assert len(self.klts) == len(self.longs)
        self.edge_index.append(len(self.klts))
        signs = [1] * len(graph_diff.pos.edge) + [-1] * len(graph_diff.neg.edge)
        self.edge_signs.append(np.array(signs).reshape((-1,1)))
        dets = [np.asarray(a) for a in (graph_diff.pos.detection, graph_diff.neg.detection)]
        dets = [a for a in dets if a.size]
        if dets:
            dets = np.vstack(dets)
        else:
            dets = np.empty((0, self.model.detecton_feature_length))
        self.detections.append(dets)
        signs = [1] * len(graph_diff.pos.detection) + [-1] * len(graph_diff.neg.detection)
        self.detection_signs.append(np.array(signs).reshape((-1,1)))
        self.entry_diff.append(graph_diff.pos.entries - graph_diff.neg.entries)

    def __len__(self):
        if self.klts is None:
            self.init()
        return len(self.detections)

    def get_batch_pair(self, item):
        if self.klts is None:
            self.init()
        i1 = self.edge_index[item]
        i2 = self.edge_index[item + 1]
        klts = [self.klts[i] for i in range(i1, i2)]
        longs = [self.longs[i] for i in range(i1, i2)]
        signs = self.edge_signs[item]
        pos_klt = [klt for klt, sign in zip(klts, signs) if sign == 1]
        neg_klt = [klt for klt, sign in zip(klts, signs) if sign == -1]
        pos_long = [long for long, sign in zip(longs, signs) if sign == 1]

        neg_long = [long for long, sign in zip(longs, signs) if sign == -1]
        assert len(pos_klt) + len(neg_klt) == len(klts)
        assert len(pos_long) + len(neg_long) == len(longs)

        detections = self.detections[item]
        signs = self.detection_signs[item]
        pos_detection = [detection for detection, sign in zip(detections, signs) if sign == 1]
        neg_detection = [detection for detection, sign in zip(detections, signs) if sign == -1]

        entry_diff = self.entry_diff[item]
        pos_entry = neg_entry = 0
        if entry_diff > 0:
            pos_entry = entry_diff
        else:
            neg_entry = -entry_diff

        return GraphBatchPair(GraphBatch(list(zip(pos_klt, pos_long)), pos_detection, pos_entry),
                              GraphBatch(list(zip(neg_klt, neg_long)), neg_detection, neg_entry),
                              'Unknown')


    def __getitem__(self, item):
        if self.klts is None:
            self.init()
        i1 = self.edge_index[item]
        i2 = self.edge_index[item + 1]

        klt_idx, klt_data = self.klts[i1:i2]
        long_idx, long_data = self.longs[i1:i2]
        return GraphDiffData(
            klt_idx=klt_idx, klt_data=klt_data, long_idx=long_idx, long_data=long_data,
            edge_signs=self.edge_signs[item],
            detections=self.detections[item],
            detection_signs=self.detection_signs[item],
            entry_diff=float(self.entry_diff[item])
        )


def make_graph_batch(tr, model):
    prv = tr[0]
    connection_features = []
    for nxt in tr[1:]:
        connection_features.append(model.connection_weight_feature(prv, nxt))
        prv = nxt
    detection_features = [model.detecton_weight_feature(d) for d in tr]
    return GraphBatch(connection_features, detection_features, 0)



def find_minimal_graph_diff(scene, graph, model, empty=torch.tensor([])):
    graph_diff = []
    gt_tracks, graph_frames = ground_truth_tracks(scene.ground_truth(), graph)
    gt_tracks = split_track_on_missing_edge(gt_tracks)
    max_len = 10000

    for tr in gt_tracks:
        prv = tr[0]
        for nxt in tr[1:]:
            prv.gt_next = nxt
            prv = nxt
        prv.gt_next = None

    for det in graph:
        if det.track_id is None:
            # False positive
            graph_diff.append(GraphBatchPair(GraphBatch(empty, empty, 0),
                                             GraphBatch(empty, [model.detecton_weight_feature(det)], 1), 'FalsePositive'))
            for nxt, weight_data in det.next_weight_data.items():
                f = model.connection_weight_feature(det, nxt)
                if nxt.track_id is not None:
                    for d in nxt.prev:
                        if d.track_id == nxt.track_id:
                            # Split from False Positive
                            graph_diff.append(GraphBatchPair(GraphBatch([model.connection_weight_feature(d, nxt)], empty, 0),
                                                             GraphBatch([f], [model.detecton_weight_feature(det)], 0), 'SplitFromFalsePositive'))
                            break
                    else:
                        # Extra first
                        graph_diff.append(GraphBatchPair(GraphBatch(empty, empty, 0),
                                                         GraphBatch([f], [model.detecton_weight_feature(det)], 0), 'ExtraFirst'))
                else:
                    # Dual false positive
                    graph_diff.append(GraphBatchPair(GraphBatch(empty, empty, 0),
                                                     GraphBatch([f], [model.detecton_weight_feature(det), model.detecton_weight_feature(nxt)], 1), 'DualFalsePositive'))
        else:
            same_track = []
            other_track = []
            for nxt, weight_data in det.next_weight_data.items():
                if not weight_data:
                    continue
                if det.track_id == nxt.track_id:
                    same_track.append(nxt)
                else:
                    other_track.append(nxt)
            same_track.sort(key=lambda d: d.frame)

            if same_track:
                f1 = model.connection_weight_feature(det, same_track[0])
                if len(same_track) > 1:
                    if same_track[0].next_weight_data[same_track[1]]:
                        # Detection skipp
                        f2 = model.connection_weight_feature(same_track[0], same_track[1])
                        f_skip = model.connection_weight_feature(det, same_track[1])
                        graph_diff.append(GraphBatchPair(GraphBatch([f1, f2], [model.detecton_weight_feature(same_track[0])], 0),
                                                         GraphBatch([f_skip], empty, 0), 'DetectionSkipp'))
                else:
                    # Skipp last
                    graph_diff.append(GraphBatchPair(GraphBatch([f1], [model.detecton_weight_feature(same_track[0])], 0),
                                                     GraphBatch(empty, empty, 0), 'SkippLast'))
                # Split
                graph_diff.append(GraphBatchPair(GraphBatch([f1], empty, 0),
                                                 GraphBatch(empty, empty, 1), 'Split'))

                if det.track_id not in {d.track_id for d in det.prev}:
                    # Skipp first
                    graph_diff.append(GraphBatchPair(GraphBatch([f1], [model.detecton_weight_feature(det)], 0),
                                                     GraphBatch(empty, empty, 0), 'SkippFirst'))

            # Long connection order
            prv_tr = None
            tr = [det]
            for same_nxt in same_track[1:]:
                while tr[-1] is not same_nxt:
                    tr.append(tr[-1].gt_next)
                prv_tr = tr
                tr = [det, same_nxt]
                graph_diff.append(GraphBatchPair(make_graph_batch(prv_tr, model), make_graph_batch(tr, model), "LongConnectionOrder"))
                # print([d.frame for d in prv_tr], '>', [d.frame for d in tr])

            for other_nxt in other_track:
                if other_nxt.track_id is None:
                    f_bad = model.connection_weight_feature(det, other_nxt)
                    if same_track:
                        # Split to false positive
                        f_ok = model.connection_weight_feature(det, same_track[0])
                        graph_diff.append(GraphBatchPair(GraphBatch([f_ok], empty, 0),
                                                         GraphBatch([f_bad], [model.detecton_weight_feature(other_nxt)], 1), 'SplitToFalsePositive'))
                    else:
                        # Extra last
                        graph_diff.append(GraphBatchPair(GraphBatch(empty, empty, 0),
                                                         GraphBatch([f_bad], [model.detecton_weight_feature(other_nxt)], 0), 'ExtraLast'))

                else:
                    prev = [d for d in other_nxt.prev if d.track_id == other_nxt.track_id]
                    if prev and same_track:
                        nxt = same_track[0]
                        other = max(prev, key=lambda d: d.frame)
                        ok1 = model.connection_weight_feature(det, nxt)
                        ok2 = model.connection_weight_feature(other, other_nxt)
                        bad1 = model.connection_weight_feature(det, other_nxt)
                        if other.next_weight_data[nxt]:
                            # ID switch
                            bad2 = model.connection_weight_feature(other, nxt)
                            graph_diff.append(GraphBatchPair(GraphBatch([ok1, ok2], empty, 0),
                                                             GraphBatch([bad1, bad2], empty, 0), 'IdSwitch'))
                        else:
                            # Double split and merge
                            graph_diff.append(GraphBatchPair(GraphBatch([ok1, ok2], empty, 0),
                                                             GraphBatch([bad1], empty, 1), 'DoubleSplitAndMerge'))

                    elif not same_track:
                        # Merge
                        f = model.connection_weight_feature(det, other_nxt)
                        graph_diff.append(GraphBatchPair(GraphBatch(empty, empty, 1),
                                                         GraphBatch([f], empty, 0), 'Merge'))
                    else:
                        # Split and merge
                        ok = model.connection_weight_feature(det, same_track[0])
                        bad = model.connection_weight_feature(det, other_nxt)
                        graph_diff.append(GraphBatchPair(GraphBatch([ok], empty, 1),
                                                         GraphBatch([bad], empty, 0), 'SplitAndMerge'))
        if len(graph_diff) > max_len:
            yield graph_diff
            graph_diff = []

    # Not too short tracks
    for tr in gt_tracks:
        detections = []
        edges = []
        prv = None
        for det in tr:
            detections.append(model.detecton_weight_feature(det))
            if prv is not None:
                edges.append(model.connection_weight_feature(prv, det))
            if len(detections) == find_minimal_graph_diff.too_short_track:
                graph_diff.append(GraphBatchPair(GraphBatch(empty, empty, 0),
                                                 GraphBatch(edges, detections, 1), 'TooShortTrack'))
            elif len(detections) == find_minimal_graph_diff.long_track:
                graph_diff.append(GraphBatchPair(GraphBatch(edges, detections, 1),
                                                 GraphBatch(empty, empty, 0), 'LongTrack'))

                detections = []
                edges = []
            prv = det
            if len(graph_diff) > max_len:
                yield graph_diff
                graph_diff = []

    # Long false positive tracks
    for tr in false_positive_tracks(gt_tracks, graph):
        if len(tr) <= 2:  # Covered by FalsePositive and DualFalsePositive above
            continue
        detections = []
        edges = []
        prv = None
        for det in tr:
            detections.append(model.detecton_weight_feature(det))
            if prv is not None:
                edges.append(model.connection_weight_feature(prv, det))
            prv = det
        graph_diff.append(GraphBatchPair(GraphBatch(empty, empty, 0),
                                         GraphBatch(edges, detections, 1), 'LongFalsePositiveTrack'))
        if len(graph_diff) > max_len:
            yield graph_diff
            graph_diff = []

    yield graph_diff

find_minimal_graph_diff.too_short_track = 2
find_minimal_graph_diff.long_track = 4


def prep_minimal_graph_diff_worker(arg):
    dataset, cam, part, model, fn, base_bfn = arg
    if not os.path.exists(base_bfn):
        bfns = []
        for i, graph_diff in enumerate(find_minimal_graph_diff(dataset.scene(cam), load_graph(fn),  model)):
            bfn = base_bfn + "_%.4d" % i
            save_torch(graph_diff, bfn)
            bfns.append(bfn)
        open(base_bfn, "w").close()
    else:
        bfns = glob(base_bfn + "_*")
    return part, base_bfn, bfns

def prep_minimal_graph_diffs(dataset, model, threads=None, limit=None, skipped_ggd_types=()):
    trainval = {'train': [], 'eval': []}
    final_trainval = {n: [] for n in trainval.keys()}
    diff_lists = {}
    jobs = []
    os.makedirs(os.path.join(dataset.cachedir, "minimal_graph_diff"), exist_ok=True)
    for part in trainval.keys():
        if limit is None:
            entries = graph_names(dataset, part)
        elif isinstance(limit, list):
            entries = limit[part]
        else:
            entries = graph_names(dataset, part)
            shuffle(entries)
            entries = entries[:limit]
        for fn, cam in entries:
            bfn = os.path.join(dataset.cachedir, "minimal_graph_diff", model.feature_name + '-' + os.path.basename(fn))
            jobs.append((dataset, cam, part, model, fn, bfn))
            final_trainval[part].append(bfn)

    trainval_name = os.path.join(dataset.cachedir, "minimal_graph_diff", "%s_%s_trainval.json" % (dataset.name, model.feature_name))
    skipped_ggd_types_name = os.path.join(dataset.cachedir, "minimal_graph_diff", "%s_%s_skipped_ggd_types.pck" % (dataset.name, model.feature_name))
    if os.path.exists(trainval_name) and os.path.exists(skipped_ggd_types_name):
        current_trainval = load_json(trainval_name)
        for part in trainval.keys():
            if set(current_trainval[part]) != set(final_trainval[part]):
                break
        else:
            if load_pickle(skipped_ggd_types_name) == skipped_ggd_types:
                return

    for part in trainval.keys():
        dn = os.path.join(dataset.cachedir, "minimal_graph_diff", "%s_%s_%s_mmaps" % (dataset.name, model.feature_name, part))
        if os.path.exists(dn):
            rmtree(dn)
        diff_lists[part] = GraphDiffList(dn, model)

    save_pickle(skipped_ggd_types, skipped_ggd_types_name)
    for part, base_bfn, bfns in parallel(prep_minimal_graph_diff_worker, jobs, threads, "Prepping minimal graph diffs"):
        trainval[part].append(base_bfn)
        save_json(trainval, trainval_name)
        for bfn in bfns:
            graphdiff = torch.load(bfn)
            lst = diff_lists[part]
            for gd in graphdiff:
                if gd.name not in skipped_ggd_types:
                    lst.append(gd)

def split_track_on_missing_edge(gt_tracks):
    tracks = defaultdict(list)
    track_id = 0
    for tr in gt_tracks:
        prv = None
        for det in tr:
            if prv is not None:
                if hasattr(prv, 'next_weight_data'):
                    if not prv.next_weight_data[det]:
                        track_id += 1
                elif det not in prv.next:
                    track_id += 1
            tracks[track_id].append(det)
            prv = det
        track_id += 1
    tracks = list(tracks.values())

    for track_id, tr in enumerate(tracks):
        for det in tr:
            det.track_id = track_id
    return tracks

def merge_idx(idxes):
    idx = [0]
    i0 = 0
    for ii in idxes:
        for i in ii[1:]:
            idx.append(i + i0)
        i0 = idx[-1]
    return idx

def make_ggd_batch(ggds):
    return GraphDiffBatch(
        klt_data=torch.tensor(np.vstack([d.klt_data for d in ggds]), dtype=torch.float32),
        klt_idx=torch.tensor(merge_idx(d.klt_idx for d in ggds), dtype=torch.long),
        long_data=torch.tensor(np.vstack([d.long_data for d in ggds]), dtype=torch.float32),
        long_idx=torch.tensor(merge_idx(d.long_idx for d in ggds), dtype=torch.long),
        edge_signs=torch.tensor(np.vstack([d.edge_signs for d in ggds]), dtype=torch.float32),
        edge_idx=torch.tensor(np.cumsum([0] + [len(d.edge_signs)  for d in ggds]), dtype=torch.long),

        detections=torch.tensor(np.vstack([d.detections for d in ggds]), dtype=torch.float32),
        detection_signs=torch.tensor(np.vstack([d.detection_signs for d in ggds]), dtype=torch.float32),
        detection_idx=torch.tensor(np.cumsum([0] + [len(d.detection_signs)  for d in ggds]), dtype=torch.long),

        entry_diffs=torch.tensor(np.array([d.entry_diff for d in ggds], dtype=np.float32), dtype=torch.float32))


if __name__ == '__main__':
    from ggdtrack.duke_dataset import Duke
    from ggdtrack.visdrone_dataset import VisDrone
    from ggdtrack.model import NNModelGraphresPerConnection
    # prep_minimal_graph_diffs(Duke('/home/hakan/src/duke'), NNModelGraphresPerConnection)

    prep_minimal_graph_diff_worker((VisDrone('data'), 'train__uav0000013_00000_v', 'train', NNModelGraphresPerConnection,
                                    "cachedir/graphs/VisDrone_graph_train__uav0000013_00000_v_00000001.pck", "t.pck"))
