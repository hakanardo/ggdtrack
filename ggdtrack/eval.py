import os
from collections import namedtuple
from tempfile import TemporaryDirectory

import torch

from ggdtrack.klt_det_connect import graph_names
from ggdtrack.mmap_array import VarHMatrixList
from ggdtrack.utils import load_pickle, save_torch, parallel


class ConnectionBatch(namedtuple('ConnectionBatch', ['klt_idx', 'klt_data', 'long_idx', 'long_data'])):
    def to(self, device):
        return ConnectionBatch(*[x.to(device) for x in self])

def prep_eval_graph_worker(args):
    model, graph_name = args
    ofn = graph_name + '-%s-eval_graph' % model.feature_name
    if os.path.exists(ofn):
        return ofn
    graph = load_pickle(graph_name)

    with TemporaryDirectory() as tmpdir:
        detection_weight_features = []
        edge_weight_features_klt = VarHMatrixList(tmpdir, 'klt_data', 'klt_index', model.klt_feature_length)
        edge_weight_features_long = VarHMatrixList(tmpdir, 'long_data', 'long_index', model.long_feature_length)
        for idx, d in enumerate(graph):
            d.index = idx
            d.next = []
            d.weight_index = []
            for n, weight_data in d.next_weight_data.items():
                d.next.append(n)
                klt, long = model.connection_weight_feature(d, n)
                d.weight_index.append(len(edge_weight_features_klt))
                edge_weight_features_klt.append(klt)
                edge_weight_features_long.append(long)
            d.prev = list(d.prev)
            detection_weight_features.append(model.detecton_weight_feature(d))
            del d.max_intra_iou
            del d.max_intra_ioa
            del d.post_vs
            del d.pre_vs
            del d.next_weight_data
        assert len(edge_weight_features_klt) == len(edge_weight_features_long)
        detection_weight_features = torch.tensor(detection_weight_features)
        connection_batch = ConnectionBatch(torch.tensor(edge_weight_features_klt.index.data),
                                           torch.tensor(edge_weight_features_klt.data),
                                           torch.tensor(edge_weight_features_long.index.data),
                                           torch.tensor(edge_weight_features_long.data))
    save_torch((graph, detection_weight_features, connection_batch), ofn)
    return ofn

def prep_eval_graphs(dataset, model, threads=6):
    jobs = [(model, name) for name, cam in graph_names(dataset, "eval")]
    for i, n in enumerate(parallel(prep_eval_graph_worker, jobs, threads)):
        print('%d/%d: %s'% (i + 1, len(jobs), n))

if __name__ == '__main__':
    from ggdtrack.duke_dataset import Duke
    from ggdtrack.model import NNModelGraphresPerConnection
    prep_eval_graphs(Duke('/home/hakan/src/duke'), NNModelGraphresPerConnection)

