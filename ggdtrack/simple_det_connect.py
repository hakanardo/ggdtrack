import os
from collections import defaultdict

from vi3o import view

from ggdtrack import klt_det_connect
from ggdtrack.klt_det_connect import video_detections, connect
from ggdtrack.utils import save_graph


def prep_training_graphs(*args, **kwargs):
    kwargs['worker'] = simple_prep_training_graphs_worker
    return klt_det_connect.prep_training_graphs(*args, **kwargs)


def simple_prep_training_graphs_worker(arg):
    scene, f0, myseg, graph_name, part = arg
    if not os.path.exists(graph_name):
        graph = make_graph(video_detections(scene, f0, myseg), scene.fps)
        save_graph(graph, graph_name)
    return part, (graph_name, scene.name)

def make_graph(video_detections, fps, show=False):

    max_temporal_distance = fps
    min_iou = 0.0

    history = []
    graph = []
    for frame_idx, frame, detections in video_detections:
        for det in detections:
            det.next_weight_data = defaultdict(list)
            det.prev = set()
            graph.append(det)
            if show:
                det.draw(frame, label=det.id)

        history = history[-max_temporal_distance:]
        for old_detections in history:
            for prv in old_detections:
                for nxt in detections:
                    if det.iou(prv) > min_iou:
                        connect(prv, nxt, None)
        history.append(detections)

        if show:
            view(frame)

    return graph

if __name__ == '__main__':
    from ggdtrack.mot16_dataset import Mot16
    make_graph(video_detections(Mot16("/home/hakan/src/ggdtrack/data").scene('train__MOT16-05'), 1, 200), 25, True)
