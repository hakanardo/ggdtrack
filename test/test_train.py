from ggdtrack.train import NormStats
import numpy as np

class TestTrain:
    def test_norm_stats(self):
        s = NormStats()
        a = np.arange(10)
        s.extend(a)
        assert s.mean == np.mean(a)
        assert s.var == np.var(a)
