from collections import defaultdict

from vi3o.image import imview, view, imwrite

from ggdtrack.eval import MotMetrics, filter_out_non_roi_dets


def show_gt(scene):
    gt_frames = scene.ground_truth()
    for f in sorted(gt_frames.keys()):
        img = scene.frame(f)
        for det in gt_frames[f]:
            if hasattr(det, 'cls'):
                label = '%s %d' % (scene.class_names[det.cls], det.id)
            else:
                label = str(det.id)
            det.draw(img, color=(255,0,0), label=label)
        imview(img)


def show_detections(viddet):
    for frame_idx, frame, detections in viddet:
        for det in detections:
            det.draw(frame, label=str(det.confidence))
        view(frame)

def show_gt_and_detections(scene):
    gt_frames = scene.ground_truth()
    frame_indexes = list(gt_frames.keys())
    f0, f1 = min(frame_indexes), max(frame_indexes)
    for frame_idx, img, detections in video_detections(scene, f0, f1-f0):
        for det in gt_frames[frame_idx]:
            if hasattr(det, 'cls'):
                label = '%s %d' % (scene.class_names[det.cls], det.id)
            else:

                label = str(det.id)
            det.draw(img, color=(255,0,0), label=label)
        for det in detections:
            det.draw(img, color=(0,0,255), label=str(det.confidence))
        imview(img)

def show_metrics_result(scene, tracks):
    metrics = MotMetrics(True)
    gt_frames = scene.ground_truth()
    frame_range = range(min(gt_frames.keys()), max(gt_frames.keys()) + 1)
    filter_out_non_roi_dets(scene, tracks)
    metrics.add(tracks, gt_frames, "tst", frame_range)
    print(metrics.summary())

    misses = defaultdict(set)
    matches = defaultdict(set)
    extra = defaultdict(set)
    switches = defaultdict(set)
    for index, row in metrics.accumulators[0].mot_events.iterrows():
        if row.Type == 'MISS':
            misses[index[0]].add(row.OId)
        elif row.Type == 'MATCH':
            matches[index[0]].add(row.HId)
        elif row.Type == 'FP':
            extra[index[0]].add(row.HId)
        elif row.Type == 'SWITCH':
            switches[index[0]].add(row.HId)
        else:
            print(row.Type)

    detections = {}
    for tr in tracks:
        for det in tr:
            detections[det.frame, det.track_id] = det

    for f in frame_range:
        img = scene.frame(f)
        gt_detections = {d.id: d for d in gt_frames[f]}
        for i in misses[f]:
            gt_detections[i].draw(img, (255,255,0))
        for i in matches[f]:
            detections[f, i].draw(img, (0,255,0))
        for i in extra[f]:
            detections[f, i].draw(img, (255,0,0))
        for i in switches[f]:
            detections[f, i].draw(img, (255,100,0))
        imwrite(img, "dbg/%.8d.jpg" % f)
        view(img)



if __name__ == '__main__':
    from ggdtrack.visdrone_dataset import VisDrone
    from ggdtrack.duke_dataset import Duke
    from ggdtrack.mot16_dataset import Mot16
    from ggdtrack.klt_det_connect import video_detections
    from ggdtrack.utils import load_pickle

    # show_gt(Duke("/home/hakan/src/duke").scene(1))
    # show_detections(video_detections(Duke('/home/hakan/src/duke').scene(1), 124472, 1000, 0.3))
    # show_detections(video_detections(Duke('/home/hakan/src/duke', 'openpose').scene(1), 124472, 1000, 0.3))

    # show_gt(VisDrone('/home/hakan/src/ggdtrack/data/').scene("train__uav0000323_01173_v"))
    # show_detections(video_detections(VisDrone('/home/hakan/src/ggdtrack/data/').scene("train__uav0000323_01173_v"), 1, 1000))
    # show_detections(video_detections(VisDrone('/home/hakan/src/ggdtrack/data/').scene("val__uav0000268_05773_v"), 1, 1000, 0.8))

    # show_gt(VisDrone('/home/hakan/src/ggdtrack/data/', class_set=('car','bus','truck','pedestrian','van')).scene("val__uav0000117_02622_v"))
    # show_detections(video_detections(VisDrone('/home/hakan/src/ggdtrack/data/').scene("val__uav0000117_02622_v"), 1, 1000))
    # show_gt_and_detections(VisDrone('/home/hakan/src/ggdtrack/data/', class_set=('ignored', 'car','bus','truck','pedestrian','van')).scene("val__uav0000117_02622_v"))
    # show_gt_and_detections(VisDrone('/home/hakan/src/ggdtrack/data/', class_set=('ignored', 'car','bus','truck','pedestrian','van')).scene("val__uav0000268_05773_v"))

    # show_gt(Mot16("/home/hakan/src/ggdtrack/data").scene('train__MOT16-02'))
    # show_detections(video_detections(Mot16("/home/hakan/src/ggdtrack/data").scene('train__MOT16-05'), 1, 1000))
    # show_gt(Mot16("/home/hakan/src/ggdtrack/data").scene('train__MOT16-04'))
    # show_detections(video_detections(Mot16("/home/hakan/src/ggdtrack/data").scene('train__MOT16-04'), 1, 1000))
    # show_detections(video_detections(Mot16("/home/hakan/src/ggdtrack/data").scene('train__MOT16-13'), 1, 1000))
    # show_gt(Mot16("/home/hakan/src/ggdtrack/data").scene('train__MOT16-13'))

    # show_metrics_result(VisDrone('data').scene("val__uav0000305_00000_v"), load_pickle("cachedir/logdir_VisDrone/tracks/VisDrone_graph_val__uav0000305_00000_v_00000001.pck"))
    show_metrics_result(VisDrone('data').scene("val__uav0000268_05773_v"), load_pickle("cachedir/logdir_VisDrone/tracks/VisDrone_graph_val__uav0000268_05773_v_00000001.pck"))

