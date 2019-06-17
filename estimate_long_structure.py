from glob import glob

import torch
from tqdm import tqdm

from ggdtrack.dataset import ground_truth_tracks
from ggdtrack.duke_dataset import Duke
from ggdtrack.klt_det_connect import graph_names
from ggdtrack.lptrack import lp_track
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.utils import promote_graph


def main():
    dataset = Duke('data', cachedir="cachedir") #_mc5")
    model = NNModelGraphresPerConnection()
    logdir = dataset.logdir
    print(logdir)
    fn = sorted(glob("%s/snapshot_???.pyt" % logdir))[-1]
    model.load_state_dict(torch.load(fn)['model_state'])
    model.eval()

    gt_not_in_graph = long_connections = long_connections_within_bound = 0

    for name, cam in tqdm(graph_names(dataset, "eval"), "Estimating long structure"):
        name = name.replace("/lunarc/nobackup/projects/lu-haar/ggdtrack/", "")
        graph, detection_weight_features, connection_batch = torch.load(name + '-%s-eval_graph' % model.feature_name)
        promote_graph(graph)
        connection_weights = model.connection_batch_forward(connection_batch)
        detection_weights = model.detection_model(detection_weight_features)

        scene = dataset.scene(cam)
        gt_tracks, gt_graph_frames = ground_truth_tracks(scene.ground_truth(), graph)
        for tr in gt_tracks:
            prv = tr[0]
            for det in tr[1:]:
                prv.gt_next = det
                prv = det


        for det in graph:
            for i, nxt in zip(det.weight_index, det.next):
                if det.track_id == nxt.track_id != None and nxt.frame - det.frame > 1:
                    long_connections += 1
                    upper = get_upper_bound_from_gt(det, nxt, connection_weights, detection_weights)
                    if upper is None:
                        gt_not_in_graph += 1
                    elif 0 < connection_weights[i] < upper:
                        long_connections_within_bound += 1
                    # print ("  %s -[%4.2f]-> %s" % (det.track_id, connection_weights[i], nxt.track_id),
                    #        det.frame, nxt.frame, upper)


        # tracks = lp_track(graph, connection_batch, detection_weight_features, model)
        # print(tracks)

        print()
        print(gt_not_in_graph, long_connections, long_connections_within_bound)
        print(long_connections_within_bound / (long_connections - gt_not_in_graph))
        print()


def get_upper_bound_from_gt(det, nxt, connection_weights, detection_weights):
    upper = 0
    d = det
    while True:
        if d.gt_next not in det.next:
            return None
        upper += connection_weights[det.weight_index[det.next.index(d.gt_next)]]
        d = d.gt_next
        if d is nxt:
            break
        # upper += detection_weights[d.index]
    return upper


if __name__ == '__main__':
    main()
