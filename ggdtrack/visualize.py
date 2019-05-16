from vi3o.image import imview, view



def show_gt(scene):
    gt_frames = scene.ground_truth()
    for f in sorted(gt_frames.keys()):
        img = scene.frame(f)
        for det in gt_frames[f]:
            if hasattr(det, 'cls'):
                label = '%s %d' % (scene.class_names[det.cls], det.id)
            else:
                label = str(det.id)
            det.draw(img, color=(255,0,0), label=label)
        imview(img)


def show_detections(viddet):
    for frame_idx, frame, detections in viddet:
        for det in detections:
            det.draw(frame, label=str(det.confidence))
        view(frame)


if __name__ == '__main__':
    from ggdtrack.visdrone_dataset import VisDrone
    from ggdtrack.duke_dataset import Duke
    from ggdtrack.klt_det_connect import video_detections

    # show_gt(Duke("/home/hakan/src/duke").scene(1))
    # show_detections(video_detections(Duke('/home/hakan/src/duke').scene(1), 124472, 1000, 0.3))
    # show_detections(video_detections(Duke('/home/hakan/src/duke', 'openpose').scene(1), 124472, 1000, 0.3))

    # show_gt(VisDrone('/home/hakan/src/ggdtrack/data/').scene("train__uav0000323_01173_v"))
    show_detections(video_detections(VisDrone('/home/hakan/src/ggdtrack/data/').scene("train__uav0000323_01173_v"), 1, 1000))
