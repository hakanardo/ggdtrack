import numpy as np
import torch
from torch import nn


class DetectionModel(nn.Module):
    def __init__(self, features):
        nn.Module.__init__(self)
        self.net = nn.Sequential(nn.Linear(features, 32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.ReLU(),
                                 nn.Linear(32, 1))
        self.register_buffer('mean', torch.zeros(features))
        self.register_buffer('std', torch.ones(features))

    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros((0,), device=x.device)
        return self.net((x - self.mean) / self.std)

class KltEdgeModel(nn.Module):
    def __init__(self, features):
        nn.Module.__init__(self)
        self.net = nn.Sequential(nn.Linear(features, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU())
        self.register_buffer('mean', torch.zeros(features))
        self.register_buffer('std', torch.ones(features))

    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros((0,), device=x.device)
        return self.net((x - self.mean) / self.std)

class LongEdgeModel(nn.Module):
    def __init__(self, features):
        nn.Module.__init__(self)
        self.net = nn.Sequential(nn.Linear(features, 16), nn.ReLU(),
                                 nn.Linear(16, 16), nn.ReLU(),
                                 nn.Linear(16, 16), nn.ReLU(),
                                 nn.Linear(16, 16), nn.ReLU(),
                                 nn.Linear(16, 16), nn.ReLU(),
                                 nn.Linear(16, 16), nn.ReLU(),
                                 nn.Linear(16, 16), nn.ReLU())
                                 # nn.Linear(64, 1))
        self.register_buffer('mean', torch.zeros(features))
        self.register_buffer('std', torch.ones(features))

    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros((0,), device=x.device)
        return self.net((x - self.mean) / self.std)

class LongKltEdgeModelMean(nn.Module):
    def __init__(self, klt_feature_length, long_feature_length):
        nn.Module.__init__(self)
        self.klt_model = KltEdgeModel(klt_feature_length)
        self.long_model = LongEdgeModel(long_feature_length)
        self.combine_model = nn.Sequential(nn.Linear(82, 256), nn.ReLU(),
                                           nn.Linear(256, 256), nn.ReLU(),
                                           nn.Linear(256, 256), nn.ReLU(),
                                           nn.Linear(256, 1))

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def forward(self, all_x):
        klt_x, long_x = all_x
        if len(klt_x) == 0:
            klt = torch.zeros(64).to(klt_x.device)
        else:
            klt = self.klt_model(klt_x).mean(0)
        if len(long_x) == 0:
            long = torch.zeros(16).to(klt_x.device)
        else:
            long = self.long_model(long_x).mean(0)
        n_klt = torch.tensor(float(len(klt_x)), device=klt_x.device)
        n_long = torch.tensor(float(len(long_x)), device=klt_x.device)
        f = torch.cat([n_klt.reshape(1), klt, n_long.reshape(1), long])
        return self.combine_model(f)


class NNModel(nn.Module):
    def score(self, tracks):
        s = 0
        for tr in tracks:
            prv = None
            for det in tr:
                s += self.detection_weight_function(det)
                if prv is None:
                    s += self.entry_weight
                else:
                    s += self.connection_weight_function(prv, det, prv.next_weight_data[det])
                prv = det
        return s

    def detection_weight_function(self, det):
        f = self.detecton_weight_feature(det)
        return self.detection_model(f).item()

    def connection_weight_function(self, det, nxt):
        f = self.connection_weight_feature(det, nxt)
        return self.edge_model(f).item()


class NNModelGraphresPerConnection(NNModel):
    detecton_feature_length = 3
    klt_feature_length = 21
    long_feature_length = 6
    feature_name = 'basic'

    def __init__(self):
        nn.Module.__init__(self)
        self.detection_model = DetectionModel(self.detecton_feature_length)
        self.edge_model = LongKltEdgeModelMean(self.klt_feature_length, self.long_feature_length)
        self.entry_weight_parameter = nn.Parameter(torch.Tensor([0]))

    def forward(self, batch):
        s = self.entry_weight_parameter * batch.entries
        if batch.detection.nelement() > 0:
            s += self.detection_model(batch.detection).sum()
        for e in batch.edge:
            s += self.edge_model(e)
        return s

    def ggd_batch_forward(self, batch):
        edge_scores = self.connection_batch_forward(batch) * batch.edge_signs
        edge_scores = idx_sum(edge_scores, batch.edge_idx, 1)

        if batch.detection_signs.size == 0:
            detection_scores = torch.zeros((len(batch.detection_idx) - 1, 1))
        else:
            detection_scores = batch.detection_signs * self.detection_model(batch.detections)
            detection_scores = idx_sum(detection_scores, batch.detection_idx, 1)

        entry_score = batch.entry_diffs * self.entry_weight_parameter
        return edge_scores + detection_scores + entry_score.reshape(-1,1)


    def connection_batch_forward(self, batch):
        klt_scores = self.edge_model.klt_model(batch.klt_data)
        klt_scores, klt_n = idx_mean(klt_scores, batch.klt_idx, 64)

        long_scores = self.edge_model.long_model(batch.long_data)
        long_scores, long_n = idx_mean(long_scores, batch.long_idx, 16)

        f = torch.cat((klt_n.reshape(-1, 1), klt_scores, long_n.reshape(-1, 1), long_scores), 1)
        return self.edge_model.combine_model(f)


    def eval(self):
        nn.Module.eval(self)
        self.entry_weight = self.entry_weight_parameter.item()

    @staticmethod
    def detecton_weight_feature(det):
        return (det.confidence, det.max_intra_iou, det.max_intra_ioa)


    @staticmethod
    def connection_weight_feature(det, nxt):
        klt_lst = []
        long_lst = []
        for kind, val in det.next_weight_data[nxt]:
            if kind == 'klt':
                df = nxt.frame - det.frame
                conf = min(-c for _,_,_,c in val if c is not None)
                t = np.linspace(0, len(val)-1, 10)
                xx = np.interp(t, range(len(val)), [p[1] for p in val])
                yy = np.interp(t, range(len(val)), [p[2] for p in val])
                _, x0, y0, _ = val[0]
                _, x1, y1, _ = val[-1]
                iou = det.move(x1-x0, y1-y0).iou(nxt)

                xx -= xx[0]
                yy -= yy[0]
                if False:
                    if xx[-1] != 0:
                        xx /= xx[-1]
                    if yy[-1] != 0:
                        yy /= yy[-1]

                features = np.hstack([np.array([df, conf, iou]), xx[1:], yy[1:]]).astype(np.float32)
                assert NNModelGraphresPerConnection.klt_feature_length == len(features)
                klt_lst.append(features)
            elif kind == 'long':
                df = nxt.frame - det.frame
                iou = val.iou(nxt)
                pre_v = tuple(val.prediction_v[:2])
                if nxt.post_vs:
                    post_v = tuple(np.median(np.array(nxt.post_vs)[:,:2], 0))
                else:
                    post_v = (0, 0)
                features = np.array((df, iou) + pre_v + post_v).astype(np.float32)
                if NNModelGraphresPerConnection.long_feature_length != len(features):
                    print(len(features))
                assert NNModelGraphresPerConnection.long_feature_length == len(features)
                long_lst.append(features)
            else:
                assert False
        if klt_lst:
            klt = np.vstack(klt_lst)
        else:
            klt = np.zeros((0, NNModelGraphresPerConnection.klt_feature_length))
        if long_lst:
            long = np.vstack(long_lst)
        else:
            long = np.zeros((0, NNModelGraphresPerConnection.long_feature_length))
        return klt, long

def idx_mean(scores, idx, width):
    scores = torch.cat((torch.zeros((1, width), device=scores.device), scores))
    scores = torch.cumsum(scores, 0)
    n = (idx[1:] - idx[:-1]).to(dtype=torch.float)
    scores = scores[idx]
    mean = (scores[1:] - scores[:-1]) / n[:, None]
    mean = torch.where(n.reshape(-1,1)==0, torch.zeros_like(mean), mean)
    return mean, n

def idx_sum(scores, idx, width):
    scores = torch.cat((torch.zeros((1, width), device=scores.device), scores))
    scores = torch.cumsum(scores, 0)
    scores = scores[idx]
    return scores[1:] - scores[:-1]
