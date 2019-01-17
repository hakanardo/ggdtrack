from tempfile import TemporaryDirectory

import torch

from ggdtrack.graph_diff import GraphDiffList
from ggdtrack.model import NNModelGraphresPerConnection


class TestHigh:
    def test_ggd_batches(self):
        graphres = torch.load("test_data/duke_graph_1_00180640.pck")

        with TemporaryDirectory() as tmpdir:
            lst = GraphDiffList(tmpdir)

            model = NNModelGraphresPerConnection()
            model.load_state_dict(torch.load("test_data/duke_graphres_model_000.pyt"))
            model.eval()

            old = []
            batch_size = 4
            n = (len(graphres) // batch_size) * batch_size
            for i in range(n):
                ex1 = graphres[i]
                old.append((model(ex1.pos) - model(ex1.neg)).item())
                lst.append(graphres[i])

            for i0 in range(0, n, batch_size):
                l = model.ggd_batch_forward(make_ggd_batch([lst[i] for i in range(i0, i0 + batch_size)]))
                for i in range(i0, i0 + batch_size):
                    assert abs(l[i-i0].item() - old[i]) < 1e-3
