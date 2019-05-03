import json
import os
from collections import defaultdict
from glob import glob

from vi3o import view
from vi3o.image import imread, imview, imscale

from ggdtrack.dataset import Detection, Dataset, Scene
import numpy as np


def nms(detections):
    detections.sort(key=lambda d: d.confidence)
    selected = []
    for det in detections:
        for old in selected:
            if old.iou(det) > 0.3:
                break
        else:
            selected.append(det)
    return selected



class VisDrone(Dataset):
    name = 'VisDrone'
    class_names = ('ignored','pedestrian','person','bicycle','car','van','truck','tricycle','awning-tricyle','bus','motor', 'others')

    def __init__(self, path, detections='FasterRCNN-MOT-detections', scale=1.0, default_min_conf=0.4, class_set = ('car','bus','truck','pedestrian','van')):
        self.base_path = os.path.join(path, "VisDrone2019")
        self.detections_dir = detections
        self.scale = scale
        self.default_min_conf = default_min_conf
        self.parts = {
            'train': self._list_scenes('train'),
            'eval': self._list_scenes('val'),
            'test': self._list_scenes('test-challenge'),
        }
        self.class_indexes = set([self.class_names.index(c) for c in class_set])

    def _list_scenes(self, part):
        seqs = os.listdir(os.path.join(self.base_path, 'VisDrone2019-MOT-' + part, 'sequences'))
        return [part + '__' + s for s in seqs]

    def _path(self, scene, d):
        part, seq = scene.split('__')
        return os.path.join(self.base_path, 'VisDrone2019-MOT-' + part, d, seq)

    def scene(self, scene):
        return VisDroneScene(self, scene)

    def frame(self, scene, frame):
        fn = os.path.join(self._path(scene, 'sequences'), '%.7d.jpg' % frame)
        return imread(fn)

    def detections(self, scene, start_frame=1, stop_frame=float('Inf')):
        raise NotImplementedError

    def ground_truth(self, scene):
        frames = defaultdict(list)
        fn = self._path(scene, 'annotations') + '.txt'
        for row in np.loadtxt(fn, delimiter=',', dtype=int):
            if row[7] in self.class_indexes:
                det = Detection(row[0], row[2], row[3], row[2]+row[4], row[3]+row[5], None, row[1])
                det.cls = row[7]
                frames[det.frame].append(det)
        return frames

    def ground_truth_detections(self, camera):
        raise NotImplementedError

    def download(self):
        if not os.path.exists(self.base_path):
            raise NotImplementedError

    def prepare(self):
        pass

    def roi(self, scene):
        h, w, _ = self.frame(scene, 1).shape
        return [(0, 0), (0, h-1), (w-1, h), (w-1, 0)]


class VisDroneScene(Scene):
    fps = 25

    def __init__(self, dataset, name):
        Scene.__init__(self, dataset, name)
        n = len(os.listdir(os.path.join(dataset._path(name, 'sequences'))))
        self.parts = {part: range(1, n+1) for part in ['train', 'eval', 'test']}
        self._detections = None

    def detections(self, start_frame=1, stop_frame=np.inf):
        if self._detections is None:
            detections = defaultdict(list)
            part, seq = self.name.split('__')
            seq = os.path.join(self.dataset.base_path, self.dataset.detections_dir, part, seq + '.txt')

            did = 0
            for l in open(seq, "r").readlines():
                frow = tuple(map(float, l.split(',')))
                row = tuple(map(int, frow))
                if row[7] in self.dataset.class_indexes:
                    det = Detection(row[0], row[2], row[3], row[2]+row[4], row[3]+row[5], frow[6], did)
                    if self.dataset.scale != 1.0:
                        det = det.scale(self.dataset.scale)
                    det.cls = row[7]
                    detections[det.frame].append(det)
                    did += 1
            for dets in detections.values():
                dets[:] = nms(dets)
            self._detections = detections
            self._last_frame = max(detections.keys())

        stop_frame = min(stop_frame, self._last_frame)
        for f in range(start_frame, stop_frame + 1):
            for det in self._detections[f]:
                yield det

