import os
from collections import defaultdict
from tempfile import TemporaryDirectory

import torch
from vi3o.image import imread

from ggdtrack.graph_diff import GraphDiffList, make_ggd_batch
from ggdtrack.klt_det_connect import make_graph
from ggdtrack.model import NNModelGraphresPerConnection
from ggdtrack.utils import load_pickle

mydir = os.path.dirname(__file__)

class TestHigh:
    def test_ggd_batches(self):
        graphres = torch.load(os.path.join(mydir, "data", "basic-duke_graph_3_00190415.pck"))

        with TemporaryDirectory() as tmpdir:
            model = NNModelGraphresPerConnection()
            model.load_state_dict(torch.load(os.path.join(mydir, "data", "snapshot_009.pyt"))['model_state'])
            model.eval()

            lst = GraphDiffList(tmpdir, model)

            old = []
            batch_size = 4
            n = (len(graphres) // batch_size) * batch_size
            for i in range(n):
                ex1 = graphres[i]
                old.append((model(ex1.pos) - model(ex1.neg)).item())
                lst.append(graphres[i])

            for i0 in range(0, n, batch_size):
                batch = make_ggd_batch([lst[i] for i in range(i0, i0 + batch_size)])
                l = model.ggd_batch_forward(batch)
                for i in range(i0, i0 + batch_size):
                    assert abs(l[i-i0].item() - old[i]) < 1e-3

    def make_graph(self):
        video_detections = load_pickle(os.path.join(mydir, "data", "duke_test_seq_cam2_10.pck"))
        video_detections = [(frame_idx, imread(os.path.join(mydir, frame)), detections)
                            for frame_idx, frame, detections in video_detections]
        return make_graph(video_detections, 60)

    def test_make_graph(self):
        graph = self.make_graph()
        frames = list({det.frame for det in graph})
        frames.sort()
        assert len(frames) == 10

        frame_detections = defaultdict(set)
        for det in graph:
            frame_detections[det.frame].add(det)
        for f in frames:
            assert len(frame_detections[f]) == 2
        for f in frames[:-1]:
            for det in frame_detections[f]:
                ok = False
                for nxt in det.next_weight_data.keys():
                    assert nxt.frame > f
                    if nxt.frame == f + 1:
                        ok = True
                assert ok

