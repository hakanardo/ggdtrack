from random import shuffle, Random

import cv2
from collections import defaultdict
from vi3o import view
import numpy as np
import pickle
import os

from vi3o.image import imsave

from ggdtrack.utils import parallel, save_json, save_pickle, load_json, save_graph


class KltTrack:
    def __init__(self, idx, x, y):
        self.history = [(idx, x, y, None)]
        self.dets_history = []
        self.dets_history_for_post_vx = defaultdict(list)
        self.predictions = defaultdict(list)

    @property
    def idx(self):
        return self.history[-1][0]

    @property
    def x(self):
        return self.history[-1][1]

    @property
    def y(self):
        return self.history[-1][2]

    @property
    def e(self):
        return self.history[-1][3]

    def distance_to(self, x, y):
        return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

def connect(d1, d2, weight_data):
    d1.next_weight_data[d2].append(weight_data)
    d2.prev.add(d1)

def video_detections(scene, f0, frames, min_conf=None):
    if min_conf is None:
        min_conf = scene.default_min_conf
    for frame_idx in range(f0, f0 + frames):
        frame = scene.frame(frame_idx)
        detections = []
        for det in scene.detections(start_frame=frame_idx, stop_frame=frame_idx):
            if det.confidence > min_conf:
                detections.append(det)
        yield frame_idx, frame, detections


def estimate_intradet_iou(detections):
    for det in detections:
        det.max_intra_iou = 0
        det.max_intra_ioa = 0
    for i in range(len(detections)):
        for j in range(i+1, len(detections)):
            iou = detections[i].iou(detections[j])
            ioa = detections[i].ioa(detections[j])
            for det in (detections[i], detections[j]):
                det.max_intra_iou = max(det.max_intra_iou, iou)
                det.max_intra_ioa = max(det.max_intra_ioa, ioa)


def make_graph(video_detections, fps, show=False, max_connect=5):

    tracks = []
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 4,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict( maxCorners = 5000,
                           qualityLevel = 0.01,
                           minDistance = 10,
                           blockSize = 7 )

    col = (255,0,0)
    max_len = 3*fps
    min_klt_per_obj = 10
    velocity_history = fps//2
    prediction_df = fps

    graph = []
    detect = True
    for frame_idx, frame, detections in video_detections:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        height, width, _ = frame.shape

        estimate_intradet_iou(detections)
        for det in detections:
            det.next_weight_data = defaultdict(list)
            det.pre_vs = []
            det.post_vs = []
            det.prev = set()
            if show:
                det.draw(frame, label=det.id)
            graph.append(det)

        # Track klt points to next frame
        if len(tracks) > 0:
            interesting = []
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([(tr.x, tr.y) for tr in tracks]).reshape(-1, 1, 2)

            # See how the points have moved between the two frames
            p1, st, err1 = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag, e in zip(tracks, p1.reshape(-1, 2), good, err1.flat):
                if not good_flag:
                    continue
                if not (0 <= x < width and 0 <= y < height):
                    continue
                if e > 1e3:
                    continue
                tr.history.append((frame_idx, x, y, e))
                tr.history = tr.history[-max_len-1:]
                new_tracks.append(tr)
                if show:
                    cv2.circle(frame, (x, y), 2, (255-min(e*10, 255),0,0), -1)
                for prev_dets in tr.dets_history:
                    for det in prev_dets:
                        if det.id == 2860144:
                            interesting.append(tr)
            tracks = new_tracks
            interesting = tracks

            if show:
                cv2.polylines(frame, [np.int32([(x,y) for _,x,y,_ in tr.history]) for tr in interesting ],  False, col)

        # Find detections with too few klt points
        if detect:
            mask = np.zeros_like(frame_gray)
        detect = False
        min_area = float('Inf')
        for det in detections:
            cnt = 0
            for tr in tracks:
                if det.covers(tr.x, tr.y):
                    cnt += 1
            if cnt < min_klt_per_obj:
                det.update_mask(mask)
                detect = True
                min_area = min(min_area, det.area)

        # Detect new klt points
        if detect:
            feature_params['minDistance'] = int(np.sqrt(min_area / min_klt_per_obj))
            for tr in tracks:
                cv2.circle(mask, (tr.x, tr.y), feature_params['minDistance']//2, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    nt = KltTrack(frame_idx, x, y)
                    tracks.append(nt)

        # Assign detections to klt points and build detection connectivity graph
        new_tracks = []
        for tr in tracks:
            vx = vy = None

            last_dets = []
            for nxt in detections:
                if nxt.covers(tr.x, tr.y):
                    last_dets.append(nxt)
                    tr.dets_history_for_post_vx[frame_idx].append(nxt)
                    for prev_dets in tr.dets_history:
                        for prv in prev_dets:
                            df = nxt.frame - prv.frame
                            klt = tr.history[-df-1:]
                            connect(prv, nxt, ('klt', klt))
                    if vx is None and len(tr.history) > velocity_history: # Predic where the detection will be in the future
                        hist = tr.history[-velocity_history:]
                        (vx, x0), rx, _, _, _ = np.polyfit(range(len(hist)), [p[1] for p in hist], 1, full=True)
                        (vy, y0), ry, _, _, _ = np.polyfit(range(len(hist)), [p[2] for p in hist], 1, full=True)
                        klt_res = [p[3] for p in hist]
                        r = (rx[0] + ry[0], sum(klt_res), max(klt_res))
                    if vx is not None:
                        v = (vx, vy) + r
                        nxt.pre_vs.append(v)
                        for df in range(1, prediction_df):
                            d = nxt.predict(df, vx, vy)
                            d.original = nxt
                            d.prediction_v = v
                            tr.predictions[d.frame].append(d)
                        for d in tr.dets_history_for_post_vx[frame_idx - velocity_history]:
                            d.post_vs.append(v)
            if frame_idx - velocity_history in tr.dets_history_for_post_vx:
                del tr.dets_history_for_post_vx[frame_idx - velocity_history]
            if last_dets:
                tr.dets_history.append(last_dets)
            tr.dets_history = tr.dets_history[-max_connect:]
            f = frame_idx - max_len
            tr.dets_history = [last_dets for last_dets in tr.dets_history if last_dets[0].frame > f]
            if tr.dets_history:
                new_tracks.append(tr)
        tracks = new_tracks

        # Form long term connection from predicted detections
        for tr in tracks:
            for prd in tr.predictions[frame_idx]:
                for det in detections:
                    if det not in tr.dets_history[-1] and prd.iou(det) > 0.5:
                        connect(prd.original, det, ('long', prd))
            del tr.predictions[frame_idx]

        if show:
            for det in detections:
                if det.pre_vs:
                    # vx = np.median([vx for vx, vy in det.pre_vs])
                    # vy = np.median([vy for vx, vy in det.pre_vs])
                    for vx, vy, r, _, _ in det.pre_vs:
                        df = 30
                        d = det.predict(df, vx, vy)
                        cv2.arrowedLine(frame, (int(det.cx), int(det.cy)), (int(d.cx), int(d.cy)),
                                        (0,0,max(0, 255-int(r))), 1)

        prev_gray = frame_gray
        if show:
            view(frame)

    return graph


def prep_training_graphs_worker(arg):
    scene, f0, myseg, graph_name, part, params = arg
    if not os.path.exists(graph_name):
        graph = make_graph(video_detections(scene, f0, myseg), scene.fps, **params)
        save_graph(graph, graph_name)
        save_json({'first_frame': f0, 'length': myseg}, graph_name + '-meta.json')
    return part, (graph_name, scene.name)


def prep_training_graphs(dataset, cachedir, threads=None, segment_length_s=10, segment_overlap_s=1, limit=None,
                         worker=prep_training_graphs_worker, worker_params=None):
    if worker_params is None:
        worker_params = {}
    lsts = {n: [] for n in dataset.parts.keys()}
    jobs = []
    for part in lsts.keys():
        for scene_name in dataset.parts[part]:
            scene = dataset.scene(scene_name)
            segment_length = segment_length_s * scene.fps
            segment_overlap = segment_overlap_s * scene.fps
            f0 = scene.parts[part].start
            while f0 + segment_length <  scene.parts[part].stop or f0 == scene.parts[part].start:
                if f0 + 2*segment_length >  scene.parts[part].stop:
                    myseg = scene.parts[part].stop - f0
                else:
                    myseg = segment_length
                graph_name = os.path.join(cachedir, "graphs", "%s_graph_%s_%.8d.pck" % (dataset.name, scene_name, f0))
                jobs.append((scene, f0, myseg, graph_name, part, worker_params))
                f0 += myseg - segment_overlap

    jobs.sort(key=lambda j: j[3])
    Random(42).shuffle(jobs)
    if limit is not None:
        jobs = [j for j in jobs if j[4] != 'test']
        jobs = jobs[:limit]

    for part, entry in parallel(worker, jobs, threads, 'Preppping training graphs'):
        lsts[part].append(entry)
        save_json(lsts, os.path.join(cachedir, "graphs", "%s_traineval.json" % dataset.name))

def graph_names(dataset, part):
    parts = load_json(os.path.join(dataset.cachedir, "graphs", "%s_traineval.json" % dataset.name))
    if part == 'trainval':
        return parts['train'] + parts['eval']
    else:
        return parts[part]


def make_duke_test_video():
    cam = 2
    seq = []
    scene = Duke('/home/hakan/src/duke').scene(cam)
    for frame_idx, frame, detections in video_detections(scene, 54373, 10):
        fn = "test/data/duke_frame_%d_%.8d.jpg" % (cam, frame_idx)
        imsave(frame, fn)
        seq.append((frame_idx, fn.replace("test/", ""), detections))
    save_pickle(seq, "test/data/duke_test_seq_cam2_10.pck")
    gt = scene.ground_truth()
    gt = {f: gt[f] for f, _, _ in seq}
    save_pickle(gt, "test/data/duke_test_seq_cam2_10_gt.pck")


if __name__ == '__main__':
    from ggdtrack.duke_dataset import Duke
    from ggdtrack.visdrone_dataset import VisDrone
    from ggdtrack.mot16_dataset import Mot16

    # scene = Duke('/home/hakan/src/duke').scene(1)
    # make_graph(video_detections(scene, 124472, 100, 0), scene.fps, True, True)

    # prep_training_graphs(Duke('/home/hakan/src/duke'))
    # prep_training_graphs_worker((Duke('/home/hakan/src/duke').scene(2), 232034, 600, "cachedir/graphs/duke_graph_2_00232034.pck", "test"))
    # Duke('/home/hakan/src/duke').scene(7).frame(336553 + 1129-2)
    # prep_training_graphs_worker((Duke('/home/hakan/src/duke').scene(2), 232034, 600, "cachedir/graphs/duke_graph_2_00232034.pck", "test"))
    # prep_training_graphs_worker((Duke('/home/hakan/src/duke').scene(2), 54373, 600, "cachedir/graphs/duke_graph_2_00054373.pck", "??"))
    # prep_training_graphs_worker((Duke('/home/hakan/src/duke').scene(2), 54373, 10, "cachedir/graphs/tst.pck", "??"))
    # make_graph(video_detections(Duke('/home/hakan/src/duke').scene(2), 54373, 10), 60, True)
    # make_duke_test_video()

    make_graph(video_detections(VisDrone('/home/hakanad/src/ggdtrack/data/').scene("val__uav0000305_00000_v"), 160, 600), 25, True)
    # make_graph(video_detections(VisDrone('/home/hakan/src/ggdtrack/data/').scene("val__uav0000137_00458_v"), 1, 10000), 25, True)
    # make_graph(video_detections(VisDrone('/home/hakan/src/ggdtrack/data/').scene("train__uav0000279_00001_v"), 160, 10000), 25, True)
    # make_graph(video_detections(Mot16("/home/hakan/src/ggdtrack/data").scene('train__MOT16-13'), 1, 1000), 25, True)