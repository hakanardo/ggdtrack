import os
from collections import defaultdict

import numpy as np
from vi3o.image import imread

from ggdtrack.dataset import Dataset, Scene, Detection, nms


class Mot16(Dataset):

    def __init__(self, path, cachedir=None, logdir=None, default_min_conf=None, fold=0):
        assert 0 <= fold <=3
        self.name = 'MOT16_fold%d' % fold
        Dataset.__init__(self, cachedir, logdir)
        self.base_path = os.path.join(path, "MOT16")
        if default_min_conf is None:
            self.default_min_conf = 0
        trainval = self._list_scenes('train')
        trainval.sort()
        if fold == 3:
            eval = [trainval.pop(-1)]
        else:
            eval = [trainval.pop(2*fold), trainval.pop(2*fold)]
        self.parts = {
            'train': trainval,
            'eval': eval,
            'test': self._list_scenes('test'),
        }

    def _list_scenes(self, part):
        seqs = os.listdir(os.path.join(self.base_path, part))
        return [part + '__' + s for s in seqs]

    def _path(self, scene, d):
        part, seq = scene.split('__')
        return os.path.join(self.base_path, part, seq, d)

    def scene(self, scene):
        return Mot16Scene(self, scene)

    def ground_truth(self, scene):
        frames = defaultdict(list)
        for row in np.loadtxt(self._path(scene, "gt/gt.txt"), delimiter=','):
            if row[0] < 1 or row[6] == 0 or row[7] != 1:
                continue
            det = Detection(int(row[0]), row[2], row[3], row[2]+row[4], row[3]+row[5], None, int(row[1]))
            frames[det.frame].append(det)
        return frames

    def detections(self, scene, start_frame=1, stop_frame=float('Inf')):
        raise NotImplementedError

    def frame(self, scene, frame):
        return imread(self._path(scene, "img1/%.6d.jpg" % frame))

    def roi(self, scene):
        h, w, _ = self.frame(scene, 1).shape
        return [(0, 0), (0, h-1), (w-1, h), (w-1, 0)]

    def download(self):
        if not os.path.exists(self.base_path):
            raise NotImplementedError

    def prepare(self):
        pass


class Mot16Scene(Scene):
    def __init__(self, dataset, name):
        Scene.__init__(self, dataset, name)
        self.seqinfo = {}
        for l in open(dataset._path(name, "seqinfo.ini")).readlines():
            if '=' in l:
                k, v = l.split('=')
                self.seqinfo[k.strip()] = v.strip()
        self.fps = int(self.seqinfo['frameRate'])
        n = len(os.listdir(dataset._path(name, 'img1')))
        self.parts = {part: range(1, n+1) for part in ['train', 'eval', 'test']}
        self._detections = None

    def detections(self, start_frame=1, stop_frame=np.inf):
        if self._detections is None:
            detections = defaultdict(list)

            did = 0
            for row in np.loadtxt(self.dataset._path(self.name, "det/det.txt"), delimiter=','):
                if row[0] < 1 or row[6] == 0:
                    print(row[0], row[6], row[7])
                    continue
                det = Detection(int(row[0]), row[2], row[3], row[2]+row[4], row[3]+row[5], row[6], did)
                det.scene = self
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




if __name__ == '__main__':
    data = Mot16('data', fold=0)
    print(data.parts['train'])
    # print(data.scene('train__MOT16-02').fps)