import numpy as np
import os

def as_database(db, mode=None):
    if db[-6:] != "_mmaps":
        db += "_mmaps"
    if not os.path.exists(db):
        os.mkdir(db)
    return db


class ScalarList:
    def __init__(self, db, data_name, dtype=float):
        db = as_database(db)
        self.file_name = os.path.join(db, data_name)
        self.dtype = dtype
        self._data = None

    @property
    def data(self):
        if self._data is None:
            if os.path.exists(self.file_name):
                self._data = np.memmap(self.file_name, self.dtype, mode='r')
            else:
                self._data = np.empty(0, self.dtype)
        return self._data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def append(self, val):
        self.extend([val])

    def extend(self, vals):
        if len(vals) > 0:
            a = np.asarray(vals, self.dtype)
            assert len(a.shape) == 1
            self._data = None
            with open(self.file_name, 'ab') as fd:
                fd.write(a.tobytes())


class VectorList:
    def __init__(self, db, data_name, len, dtype=float):
        db = as_database(db)
        self.file_name = os.path.join(db, data_name)
        self.dtype = dtype
        self.len = len
        self._data = None

    @property
    def data(self):
        if self._data is None:
            if os.path.exists(self.file_name):
                self._data = np.memmap(self.file_name, self.dtype, mode='r').reshape((-1, self.len))
            else:
                self._data = np.empty((0, self.len), self.dtype)
        return self._data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def append(self, val):
        self.extend([val])

    def extend(self, vals):
        if len(vals) > 0:
            a = np.asarray(vals, self.dtype)
            assert len(a.shape) == 2
            assert a.shape[1] == self.len
            self._data = None
            with open(self.file_name, 'ab') as fd:
                fd.write(a.tobytes())


class VarHMatrixList:
    def __init__(self, db, data_name, index_name, width, dtype=np.float32):
        db = as_database(db)
        self.file_name = os.path.join(db, data_name)
        self.index = ScalarList(db, index_name, int)
        if len(self.index) == 0:
            self.index.append(0)
        self.width = width
        self.dtype = dtype
        self._data = None

    def __len__(self):
        return len(self.index) - 1

    @property
    def data(self):
        if self._data is None:
            if os.path.exists(self.file_name):
                self._data = np.memmap(self.file_name, self.dtype, mode='r').reshape((-1, self.width))
            else:
                self._data = np.empty((0, self.width), self.dtype)
        return self._data

    def __getitem__(self, item):
        if isinstance(item, slice):
            assert item.step == None
            i1 = self.index[item.start]
            i2 = self.index[item.stop]
            idx = self.index[item.start:item.stop+1]
            idx = [i - idx[0] for i in idx]
            return idx, self.data[i1:i2]
        else:
            i1 = self.index[item]
            i2 = self.index[item + 1]
            return self.data[i1:i2]

    def append(self, a):
        self.extend([a])

    def extend(self, arrays):
        if len(arrays) > 0:
            di = len(self.data)
            self._data = None
            idx = []
            for i, a in enumerate(arrays):
                if len(a) > 0:
                    with open(self.file_name, 'ab') as fd:
                        fd.write(np.asarray(a, dtype=self.dtype).tobytes())
                di += len(a)
                idx.append(di)
            self.index.extend(idx)


