from tempfile import NamedTemporaryFile, TemporaryDirectory

import torch

from ggdtrack.mmap_array import VarHMatrixList, ScalarList, VectorList
import numpy as np
import sys


class TestMmap:

    def test_ScalarList(self):
        with TemporaryDirectory() as tmpdir:
            lst = ScalarList(tmpdir, "d", int)
            assert len(lst) == 0
            lst.append(7)
            lst.append(42)
            assert len(lst) == 2
            assert lst[0] == 7
            assert lst[1] == 42
            lst.extend([])
            lst.extend([4,5,6])
            assert lst[0] == 7
            assert lst[1] == 42
            assert lst[2] == 4
            assert lst[3] == 5
            assert lst[4] == 6


    def test_VectorList(self):
        with TemporaryDirectory() as tmpdir:
            lst = VectorList(tmpdir, "d", 3, int)
            a1 = [1, 2, 3]
            a2 = [7,8,9]
            lst.append(a1)
            lst.append(a2)
            lst.append(a1)

            assert np.all(lst[0] == a1)
            assert np.all(lst[1] == a2)
            assert np.all(lst[2] == a1)
            assert len(lst) == 3

            lst.extend([a1, a2])
            assert np.all(lst[3] == a1)
            assert np.all(lst[4] == a2)
            assert len(lst) == 5

            lst.extend([])

            lst.append(a1)
            assert np.all(lst[4] == a2)
            assert np.all(lst[5] == a1)

    def test_VarHMatrixList(self):
        with TemporaryDirectory() as tmpdir:
            lst = VarHMatrixList(tmpdir, "d", "i", 3)
            a1 = [[1, 2, 3], [4, 5, 6]]
            a2 = [[7,8,9]]
            lst.append(a1)
            lst.append(a2)
            lst.append(a1)
            lst.append(a1 + a2)
            assert np.all(lst[0] == a1)
            assert np.all(lst[1] == a2)
            assert np.all(lst[2] == a1)
            assert np.all(lst[3] == a1 + a2)
            assert len(lst) == 4

            lst.extend([a1, a2])
            assert np.all(lst[4] == a1)
            assert np.all(lst[5] == a2)
            assert len(lst) == 6

            lst.extend([])

            lst.append(np.empty((0, 3)))
            lst.append(a1)
            assert np.all(lst[5] == a2)
            assert lst[6].shape == (0, 3)
            assert np.all(lst[7] == a1)

