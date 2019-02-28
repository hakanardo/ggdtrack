import os
from tempfile import TemporaryDirectory

import torch

from ggdtrack.graph_diff import GraphDiffList, make_ggd_batch
from ggdtrack.model import NNModelGraphresPerConnection

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

