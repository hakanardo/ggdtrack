from copy import deepcopy
from random import randrange
import numpy as np
from ggdtrack.dataset import Detection
from ggdtrack.utils import load_pickle, demote_graph, promote_graph
import os

mydir = os.path.dirname(__file__)

def slow_iou(self, other):
    b1 = self.box
    b2 = other.box
    return b1.intersection(b2).area / b1.union(b2).area

def random_detection():
    l = randrange(100)
    t = randrange(100)
    w = randrange(1,100)
    h = randrange(1,100)
    return Detection(0, l, t, l+w, t+h, 0.9, 7)

class TestData:
    def test_detetcion(self):
        d1 = Detection(0, 10, 20, 100, 200, 0.9, 7)
        assert d1.cx == 55
        assert d1.cy == 110
        assert d1.area == 16200

        d2 = Detection(2, 20, 30, 110, 220, 0.9, 7)
        assert d1.iou(d2) == d2.iou(d1) == slow_iou(d1, d2)
        assert d1.iou(d1) == d2.iou(d2) == slow_iou(d1, d1) == slow_iou(d1, d1) == 1.0

        assert d1.ioa(d2) == 0.8395061728395061
        assert d2.ioa(d1) == 0.7953216374269005

        assert d1.distance_to(d2) == d2.distance_to(d1) == 18.027756377319946
        assert d1.interpolate(d2, 1) == (1, 15, 25, 105, 210, None, None)
        assert d1.predict(2, 3, 4) == (2, 16, 28, 106, 208, None, None)
        assert d1.scale(2) == (0, 20, 40, 200, 400, 0.9, 7)
        assert d1.move(2, 3) == (0, 12, 23, 102, 203, 0.9, 7)

        mask = np.zeros((500, 500))
        d1.update_mask(mask)
        d2.update_mask(mask)
        assert mask.sum() == 5023500
        assert mask[0, 0] == mask[5, 50] == mask[50, 5] == 0
        assert mask[20, 10] == 255
        assert mask[199, 99] == 255

        for _ in range(1000):
            d1 = random_detection()
            d2 = random_detection()
            assert d1.iou(d2) == d2.iou(d1) == slow_iou(d1, d2)
            assert d1.covers(d1.cx, d1.cy)
            assert d2.covers(d2.cx, d2.cy)

            mask = np.zeros((500, 500))
            d1.update_mask(mask)
            d2.update_mask(mask)
            assert mask[int(d1.cy), int(d1.cx)] == 255
            assert mask[int(d2.cy), int(d2.cx)] == 255

    def test_demote_promote(self):
        g1 = load_pickle(os.path.join(mydir, "data", "promoted_graph.pck"))
        g2 = deepcopy(g1)
        demote_graph(g1)
        promote_graph(g1)

        for i in range(len(g1)):
            assert g1[i].prev == g2[i].prev
            assert g1[i].next_weight_data == g2[i].next_weight_data
            assert g1[i].prev.__class__ == g2[i].prev.__class__
            assert g1[i].next_weight_data.__class__ == g2[i].next_weight_data.__class__



