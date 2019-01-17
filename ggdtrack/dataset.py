import json
from collections import namedtuple

import cv2
from shapely import geometry
import numpy as np


class Detection(namedtuple('Detection', ['frame', 'left', 'top', 'right', 'bottom', 'confidence', 'id'])):
    def draw(self, img, color=None, thickness=3, label=None):
        if color is None:
            color = (127,127,127) if self.confidence is None else (0,0,255)
        cv2.rectangle(img, (int(self.left), int(self.top)), (int(self.right), int(self.bottom)), color, thickness)
        if label is not None:
            label = str(label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            text_size = cv2.getTextSize(label, font, font_scale, 1)
            x, y = int(self.left), int(self.top)
            text_top = (x, y)
            text_bot = (x + text_size[0][0] + thickness*2, y - text_size[0][1] - thickness*2)
            text_pos = (x + thickness, y - thickness)
            cv2.rectangle(img, text_top, text_bot, color, -1)
            cv2.putText(img, label, text_pos, font, font_scale, (0,0,0), 1, cv2.LINE_AA)


    @property
    def cx(self):
        return (self.left + self.right) / 2

    @property
    def cy(self):
        return (self.top + self.bottom) / 2

    @property
    def width(self):
        return abs(self.right - self.left)

    @property
    def height(self):
        return abs(self.bottom - self.top)

    @property
    def area(self):
        return self.width * self.height

    @property
    def box(self):
        if not hasattr(self, '_box'):
            self._box = geometry.box(self.left, self.top, self.right, self.bottom)
        return self._box

    def iou(self, other):
        xA = max(self.left, other.left)
        xB = min(self.right, other.right)
        if xB < xA:
            return 0.0
        yA = max(self.top, other.top)
        yB = min(self.bottom, other.bottom)
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (self.right - self.left) * (self.bottom - self.top)
        boxBArea = (other.right - other.left) * (other.bottom - other.top)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def ioa(self, other):
        xA = max(self.left, other.left)
        xB = min(self.right, other.right)
        if xB < xA:
            return 0.0
        yA = max(self.top, other.top)
        yB = min(self.bottom, other.bottom)
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (self.right - self.left) * (self.bottom - self.top)
        return interArea / float(boxAArea)

    def distance_to(self, other):
        dx = self.cx - other.cx
        dy = self.cy - other.cy
        return np.sqrt(dx**2 + dy**2)

    @property
    def area(self):
        return abs(self.right - self.left) * abs(self.bottom - self.top)

    def covers(self, x, y):
        return self.box.contains(geometry.Point(x, y))

    def interpolate(self, other, f):
        dt1 = f - other.frame
        dt2 = self.frame - f
        dtot = dt2 + dt1
        left = (dt2 * other.left + dt1 * self.left) / dtot
        right = (dt2 * other.right + dt1 * self.right) / dtot
        top = (dt2 * other.top + dt1 * self.top) / dtot
        bottom = (dt2 * other.bottom + dt1 * self.bottom) / dtot
        return Detection(f, left, top, right, bottom, None, None)

    def predict(self, df, vx, vy):
        dx = vx * df
        dy = vy * df
        d = Detection(self.frame + df, self.left + dx, self.top + dy, self.right + dx, self.bottom + dy, None, None)
        return d

    def scale(self, scale):
        return Detection(self.frame, int(self.left * scale), int(self.top * scale), int(self.right * scale), int(self.bottom * scale), self.confidence, self.id)

    def move(self, dx, dy):
        return Detection(self.frame, self.left + dx, self.top + dy, self.right + dx, self.bottom + dy, self.confidence, self.id)

    def update_mask(self, mask):
        mask[self.top:self.bottom, self.left:self.right] = 255


class Dataset:
    name = 'Unknown'
    parts = {'train': None, 'eval': None, 'test': None}
    default_min_conf = 0

    def scene(self, scene):
        raise NotImplementedError

    def graph_names(self, part):
        with open("graphs/%s_traineval.json" % self.name, "r") as fd:
            parts = json.load(fd)
        if part == 'trainval':
            return parts['train'] + parts['eval']
        else:
            return parts[part]


class Scene:
    fps = None
    parts = {'train': (), 'eval': (), 'test': ()}
    name = 'Unknown'

    def frame(self, frame):
        raise NotImplementedError

    def ground_truth(self):
        raise NotImplementedError

    def detections(self, start_frame=1, stop_frame=np.inf):
        raise NotImplementedError

    def roi(self):
        raise NotImplementedError

    @property
    def default_min_conf(self):
        return self.dataset.default_min_conf

