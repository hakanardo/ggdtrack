import numpy as np

class NNModelGraphresPerConnection:
    detecton_feature_length = 3
    klt_feature_length = 21
    long_feature_length = 6
    feature_name = 'basic'

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
