import os
import zipfile
from tempfile import TemporaryDirectory

from ggdtrack.dataset import Dataset
from ggdtrack.utils import download_file


class Mot19(Dataset):

    def __init__(self, path, cachedir=None, logdir=None, default_min_conf=None, fold=0):
        assert 0 <= fold <=3
        self.name = 'MOT16_fold%d' % fold
        Dataset.__init__(self, cachedir, logdir)
        self.base_path = os.path.join(path, "MOT19")
        if default_min_conf is None:
            self.default_min_conf = 0
        self.download()

        # trainval = self._list_scenes('train')
        # trainval.sort()
        # if fold == 3:
        #     eval = [trainval.pop(-1)]
        # else:
        #     eval = [trainval.pop(2*fold), trainval.pop(2*fold)]
        # self.parts = {
        #     'train': trainval,
        #     'eval': eval,
        #     'test': self._list_scenes('test'),
        # }

    def download(self):
        if not os.path.exists(os.path.join(self.base_path, "train")):
            with TemporaryDirectory() as tmp:
                download_file("https://motchallenge.net/data/CVPR19.zip", tmp)
                zip = zipfile.ZipFile(os.path.join(tmp, "MOT19.zip"), 'r')
                zip.extractall(self.base_path)
                zip.close()

if __name__ == '__main__':
    data = Mot19('data', fold=0)
