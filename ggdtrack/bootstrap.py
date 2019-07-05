import torch

from ggdtrack.eval import MotMetrics, filter_out_non_roi_dets
from ggdtrack.graph_diff import GraphBatchPair, GraphBatch
from ggdtrack.klt_det_connect import estimate_intradet_iou


def find_false_positive_tracks(scene, tracks):
    metrics = MotMetrics(True)
    gt_frames = scene.ground_truth()
    frame_range = range(min(gt_frames.keys()), max(gt_frames.keys()) + 1)
    filter_out_non_roi_dets(scene, tracks)
    metrics.add(tracks, gt_frames, "tst", frame_range)

    detections = {}
    for tr in tracks:
        for det in tr:
            det.fp = False
            detections[det.frame, det.track_id] = det

    for index, row in metrics.accumulators[0].mot_events.iterrows():
        if row.Type == 'FP':
            detections[index[0], row.HId].fp = True

    for tr in tracks:
        n = sum(d.fp for d in tr)
        if n == len(tr):
            yield tr

def find_false_positive_graph_diff(scene, tracks, model, empty=torch.tensor([])):
    graph_diff = []
    for tr in find_false_positive_tracks(scene, tracks):
        detections = []
        edges = []
        prv = None
        for det in tr:
            detections.append(model.detecton_weight_feature(det))
            if prv is not None:
                edges.append(model.connection_weight_feature(prv, det))
        graph_diff.append(GraphBatchPair(GraphBatch(empty, empty, 0),
                                         GraphBatch(edges, detections, 1), 'BootstrappedFPTrack'))
    return graph_diff


if __name__ == '__main__':
    from ggdtrack.visdrone_dataset import VisDrone
    from ggdtrack.utils import load_pickle
    from ggdtrack.model import NNModelGraphresPerConnection

    find_false_positive_graph_diff(VisDrone('data').scene("val__uav0000268_05773_v"), load_pickle("cachedir/logdir_VisDrone/tracks/VisDrone_graph_val__uav0000268_05773_v_00000001.pck"), NNModelGraphresPerConnection)
