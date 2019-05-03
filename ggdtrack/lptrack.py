from bisect import bisect

import cv2
from pplp import LinearProgram
from vi3o import view
import numpy as np



def hamming_weight(v, gt):
    if gt == 0.0:
        return v
    elif gt == 1.0:
        return -1.0 * v  # (1-v), but removing a constant 1 from the objective wont move the location of the maximum
    else:
        assert False


def lp_track(graph, connection_batch, detection_weight_features, model, verbose=False, add_gt_hamming=False):
    connection_weights = model.connection_batch_forward(connection_batch)
    detection_weights = model.detection_model(detection_weight_features)
    return lp_track_weights(graph, connection_weights, detection_weights, model.entry_weight, verbose, add_gt_hamming)

def lp_track_weights(graph, connection_weights, detection_weights, entry_weight, verbose=False, add_gt_hamming=False):
    if not graph:
        return []

    lp = LinearProgram(verbose)
    def Var():
        v = lp.Var()
        lp.add_constraint(0 <= v <= 1)
        return v

    for d in graph:
        d.outgoing = [Var() for n in d.next]
        d.exit = Var()
        d.entry = Var()
        d.present = Var()

    for d in graph:
        d.incomming = [p.outgoing[p.next.index(d)] for p in d.prev]

    connection_weight = 0
    for d in graph:
        lp.add_constraint(sum(d.outgoing) + d.exit - d.present == 0)
        lp.add_constraint(sum(d.incomming) + d.entry - d.present == 0)
        connection_weight += sum(connection_weights[i].item() * v for v, i in zip(d.outgoing, d.weight_index))
        if add_gt_hamming:
            connection_weight += sum(hamming_weight(v, gt) for v, gt in zip(d.outgoing, d.gt_next))


    detection_weight = sum(d.present * detection_weights[d.index] + entry_weight * d.entry for d in graph)
    if add_gt_hamming:
        detection_weight += sum(hamming_weight(d.present, d.gt_present) + hamming_weight(d.entry, d.gt_entry) for d in graph)
    lp.objective = connection_weight + detection_weight

    m = lp.maximize()

    tracks = []
    for d in graph:
        if d.entry.value:
            tr = [d]
            while not d.exit.value:
                d = d.next[max([(v.value, i) for i, v in enumerate(d.outgoing)])[1]]
                tr.append(d)
            tracks.append(tr)

    return tracks

def interpolate_missing_detections(tracks, cpy=False):
    if cpy:
        tracks = [tr[:] for tr in tracks]
    for tr in tracks:
        new_tr = []
        prv = None
        for det in tr:
            if prv is not None:
                for f in range(prv.frame+1, det.frame):
                    d = det.interpolate(prv, f)
                    if hasattr(det, 'cls'):
                        d.cls = det.cls
                    new_tr.append(d)
            prv = det
            new_tr.append(det)
        tr[:] = new_tr
    return tracks

def show_tracks(scene, tracks, frame_dets=()):
    if not tracks:
        return
    first_frame = min(tr[0].frame for tr in tracks)
    last_frame = max(tr[-1].frame for tr in tracks)
    for f in range(first_frame, last_frame+1):
        img = scene.frame(f)

        if f in frame_dets:
            for d in frame_dets[f]:
                d.draw(img, color=(255,0,0))

        for tr_id, tr in enumerate(tracks):
            if tr[0].frame <= f <= tr[-1].frame:
                det = tr[bisect([det.frame for det in tr], f)-1]
                assert det.frame == f # Did you interpolate_missing_detections(tracks)?
                if hasattr(det, 'track_id'):
                    label = '%s:%s' % (tr_id, det.track_id)
                else:
                    label = tr_id
                det.draw(img, label=label)

        cv2.polylines(img, np.array([scene.roi()]), True, (0,0,0), thickness=3)
        cv2.polylines(img, np.array([scene.roi()]), True, (255,255,255), thickness=1)


        view(img, pause=True)
        # imwrite(img, "dbg/%.8d.jpg" % f)

if __name__ == '__main__':
    from ggdtrack.duke_dataset import Duke
    from ggdtrack.visdrone_dataset import VisDrone
    from ggdtrack.utils import load_pickle
    # show_tracks(Duke('/home/hakan/src/duke').scene(3), interpolate_missing_detections(load_pickle("tracks/duke_graph_3_00190415.pck")))
    show_tracks(VisDrone('/home/hakan/src/ggdtrack/data/').scene("val__uav0000086_00000_v"), interpolate_missing_detections(load_pickle("cachedir/tracks/VisDrone_graph_val__uav0000086_00000_v_00000001.pck")))
