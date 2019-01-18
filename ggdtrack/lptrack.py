from pplp import LinearProgram


def lp_track(graph, connection_batch, detection_weight_features, model, verbose=False):
    if not graph:
        return []

    lp = LinearProgram(verbose)
    def Var():
        v = lp.Var()
        lp.add_constraint(0 <= v <= 1)
        return v

    for d in graph:
        d.outgoing = [Var() for n in d.next]
        d.exit = Var()
        d.entry = Var()
        d.present = Var()

    for d in graph:
        d.incomming = [p.outgoing[p.next.index(d)] for p in d.prev]

    connection_weights = model.connection_batch_forward(connection_batch)
    connection_weight = 0
    for d in graph:
        lp.add_constraint(sum(d.outgoing) + d.exit - d.present == 0)
        lp.add_constraint(sum(d.incomming) + d.entry - d.present == 0)
        connection_weight += sum(connection_weights[i].item() * v for v, i in zip(d.outgoing, d.weight_index))

    detection_weights = model.detection_model(detection_weight_features)
    detection_weight = sum(d.present * detection_weights[d.index] + model.entry_weight * d.entry for d in graph)
    lp.objective = connection_weight + detection_weight

    m = lp.maximize()

    tracks = []
    for d in graph:
        if d.entry.value:
            tr = [d]
            while not d.exit.value:
                d = d.next[max([(v.value, i) for i, v in enumerate(d.outgoing)])[1]]
                tr.append(d)
            tracks.append(tr)

    return tracks

def interpolate_missing_detections(tracks, cpy=False):
    if cpy:
        tracks = [tr[:] for tr in tracks]
    for tr in tracks:
        new_tr = []
        prv = None
        for det in tr:
            if prv is not None:
                for f in range(prv.frame+1, det.frame):
                    d = det.interpolate(prv, f)
                    if hasattr(det, 'cls'):
                        d.cls = det.cls
                    new_tr.append(d)
            prv = det
            new_tr.append(det)
        tr[:] = new_tr
    return tracks
