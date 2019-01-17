import json
from bisect import bisect
from collections import defaultdict
from collections import namedtuple

import cv2
import numpy as np
import os
import h5py
from shapely import geometry
from scipy import io
import pickle
from ggdtrack.dataset import Dataset, Detection, Scene


class Duke(Dataset):
    name = "duke"

    def __init__(self, path, detections='dpm', scale=1.0):
        self.path = path
        self.video_reader = DukeVideoReader(path)
        self.scale = scale
        self.detections = {
            'dpm': self.dpm_detections,
            'openpose': self.openpose_detections,
        }[detections]

    def scene(self, scene):
        return DukeScene(self, scene)

    def frame(self, camera, frame):
        img = self.video_reader.getFrame(camera, frame)
        if self.scale != 1.0:
            h, w, _ = img.shape
            imsize = (self.scale * w, self.scale *h)
            img = cv2.resize(frame, imsize)
        return img

    def dpm_detections(self, camera, start_frame=1, stop_frame=float('Inf')):
        detections = h5py.File(self.path + '/detections/DPM/camera%d.mat' % camera, "r")['detections']
        i0 = bisect(RowView(detections, 1), start_frame) - 1
        while detections[1, i0] == start_frame and i0 >= 0:
            i0 -= 1
        i0 += 1

        for ind in range(i0, detections.shape[1]):
            det = detections[:, ind]
            camera = int(det[0])
            frame  = int(det[1])
            left, top, right, bottom = map(lambda x: int(self.scale*x), det[2:6])
            confidence = det[-1]
            if frame > stop_frame:
                break
            yield Detection(frame, left, top, right, bottom, confidence, ind)

    def openpose_detections(self, camera, start_frame=1, stop_frame=float('Inf')):
        detections = h5py.File(self.path + '/detections/openpose/camera%d.mat' % camera)['detections']
        i0 = bisect(RowView(detections, 1), start_frame) - 1
        while detections[1, i0] == start_frame and i0 >= 0:
            i0 -= 1
        i0 += 1

        for ind in range(i0, detections.shape[1]):
            det = detections[:, ind]
            camera = int(det[0])
            frame  = int(det[1])
            left, top, width, height = pose2bb(det[2:])
            left, top, right, bottom = map(lambda x: int(self.scale*x), [left, top, left+width, top+height])
            confidence = det[-1]
            if frame > stop_frame:
                break
            yield Detection(frame, left, top, right, bottom, confidence, ind)

    def ground_truth_detections(self, camera):
        print(self.path + '/ground_truth/trainval.mat')
        gt = io.loadmat(self.path + '/ground_truth/trainval.mat')['trainData']
        gt = gt[gt[:,2].argsort()]
        for cam, track_id, frame, left, top, width, height, worldX, worldY, feetX, feetyY in gt:
            if cam != camera:
                continue
            det = Detection(frame, left, top, left+width, top+height, None, int(track_id))
            if self.scale != 1.0:
                det = det.scale(self.scale)
            yield det

    def ground_truth(self, camera):
        with open(self.path + '/ground_truth/camera%d.pck' % camera, "rb") as fd:
            gt_frames = pickle.load(fd)
        return gt_frames

    def convert_ground_truth(self):
        for cam in range(1, 9):
            print(cam)
            gt = defaultdict(list)
            for det in self.ground_truth_detections(cam):
                gt[det.frame].append(det)
            with open(self.path + '/ground_truth/camera%d.pck' % cam, "wb") as fd:
                pickle.dump(gt, fd, -1)

    def convert_dpm_detections(self):
        for cam in range(1, 9):
            print(cam)
            detections = defaultdict(list)
            for i, det in enumerate(self.dpm_detections(cam)):
                if i % 10000 == 0:
                    print('  ', i)
                detections[det.frame].append(det)
            with open(self.path + '/detections/DPM/frame_detections_camera%d.pck' % cam, "wb") as fd:
                pickle.dump(detections, fd, -1)

    _rois = {
        1: [[214, 553], [1904, 411], [1897, 1055], [216, 1051]],
        2: [[35, 423], [574, 413], [624, 300], [1075, 284], [1150, 341], [1153, 396], [1260, 393], [1610, 492], [1608, 460], [1614, 446], [1771, 440], [1777, 485], [1894, 483], [1894, 1048], [29, 1051]],
        3: [[65, 659], [926, 662], [798, 592], [827, 574], [1201, 573], [1500, 671], [1888, 673], [1882, 1047], [68, 1044]],
        4: [[1189, 1043], [1664, 1043], [1434, 240], [1267, 240]],
        5: [[28, 1054], [1897, 1054], [1893, 202], [410, 211], [397, 313], [311, 320], [104, 369], [71, 449], [107, 567], [27, 750]],
        6: [[154, 646], [957, 626], [1863, 886], [1866, 1050], [40, 1059], [56, 775], [103, 682]],
        7: [[40, 751], [40, 1049], [1862, 1050], [1862, 875], [862, 817], [1004, 730], [667, 710], [445, 779]],
        8: [[1867, 637], [1806, 1066], [53, 1047], [455, 807], [457, 729], [824, 531]],
    }

    def roi(self, camera):
        return self._rois[camera]

    def download(self):
        os.system('''
            mkdir %s
            pushd %s
            mkdir DukeMTMC
            pushd DukeMTMC
            mkdir -p ground_truth calibration detections frames masks videos detections/DPM detections/openpose
            
            pushd ground_truth
            wget http://vision.cs.duke.edu/DukeMTMC/data/ground_truth/trainval.mat
            wget http://vision.cs.duke.edu/DukeMTMC/data/ground_truth/trainvalRaw.mat
            popd
            
            pushd calibration
            wget http://vision.cs.duke.edu/DukeMTMC/data/calibration/calibration.txt
            wget http://vision.cs.duke.edu/DukeMTMC/data/calibration/camera_position.txt
            wget http://vision.cs.duke.edu/DukeMTMC/data/calibration/ROIs.txt
            popd
            
            pushd detections/openpose
            for i in `seq 8`; do
                wget http://vision.cs.duke.edu/DukeMTMC/data/detections/openpose/camera$i.mat
            done
            popd
            
            pushd detections/DPM
            for c in `seq 8`; do
                wget http://vision.cs.duke.edu/DukeMTMC/data/detections/DPM/camera$c.mat
            done
            popd
            
            for c in `seq 8`; do
                mkdir videos/camera$c
                pushd videos/camera$c
                for p in `seq 0 9`; do
                    wget http://vision.cs.duke.edu/DukeMTMC/data/videos/camera$c/`printf '%%.5d' $p`.MTS
                done
                popd
            done
            popd
        ''' % (self.path, self.path))

    def prepare(self):
        self.convert_ground_truth()
        self.convert_dpm_detections()


class DukeScene(Scene):
    def __init__(self, dataset, camera):
        self.dataset = dataset
        self.camera = camera

    def frame(self, frame):
        return self.dataset.frame(self.camera, frame)

    def detections(self, start_frame=1, stop_frame=np.inf):
        return self.dataset.detections(self.camera, start_frame, stop_frame)

    def ground_truth(self):
        return self.dataset.ground_truth(self.camera)

    def roi(self):
        return self.dataset.roi(self.camera)



class DukeVideoReader:
    def __init__(self, dataset_path):
        self.NumCameras = 8
        self.NumFrames = [359580, 360720, 355380, 374850, 366390, 344400, 337680, 353220]
        self.PartMaxFrame = 38370
        self.MaxPart = [9, 9, 9, 9, 9, 8, 8, 9]
        self.PartFrames = []
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 14250])
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 15390])
        self.PartFrames.append([38400, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 10020])
        self.PartFrames.append([38670, 38670, 38670, 38670, 38670, 38700, 38670, 38670, 38670, 26790])
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 21060])
        self.PartFrames.append([38400, 38370, 38370, 38370, 38400, 38400, 38370, 38370, 37350, 0])
        self.PartFrames.append([38790, 38640, 38460, 38610, 38760, 38760, 38790, 38490, 28380, 0])
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 7890])
        self.DatasetPath = dataset_path
        self.CurrentCamera = None
        self.CurrentPart = None
        self.PrevCamera = 1
        self.PrevFrame = -1
        self.PrevPart = 0
        self.Video = None #cv2.VideoCapture('{:s}/videos/camera{:d}/{:05d}.MTS'.format(self.DatasetPath, self.CurrentCamera, self.CurrentPart), cv2.CAP_FFMPEG)

    def getFrame(self, iCam, iFrame):
        # iFrame should be 1-indexed
        assert iFrame > 0 and iFrame <= self.NumFrames[iCam-1], 'Frame out of range'
        #print('Frame: {0}'.format(iFrame))
        # Cam 4 77311
        #Coompute current frame and in which video part the frame belongs
        ksum = 0
        for k in range(10):
            ksumprev = ksum
            ksum += self.PartFrames[iCam-1][k]
            if iFrame <= ksum:
                currentFrame = iFrame - 1 - ksumprev
                iPart = k
                break
        # Update VideoCapture object if we are reading from a different camera or video part
        if iPart != self.CurrentPart or iCam != self.CurrentCamera:
            self.CurrentCamera = iCam
            self.CurrentPart = iPart
            self.PrevFrame = -1
            fn = '{:s}videos/camera{:d}/{:05d}.MTS'.format(self.DatasetPath, self.CurrentCamera, self.CurrentPart)
            if not os.path.exists(fn):
                raise IOError("File not found: '%s'" % fn)
            self.Video = cv2.VideoCapture(fn, cv2.CAP_FFMPEG)
        # Update time only if reading non-consecutive frames
        if not currentFrame == self.PrevFrame + 1:

            #if iCam == self.PrevCamera and iPart == self.PrevPart and currentFrame - self.PrevFrame < 30:
            #    # Skip consecutive images if less than 30 frames difference
            #    back_frame = max(self.PrevFrame, 0)
            #else:
            #    # Seek, but make sure a keyframe is read before decoding
            back_frame = max(currentFrame - 31, 0) # Keyframes every 30 frames
            #print('back_frame set to: {0}'.format(back_frame))
            self.Video.set(cv2.CAP_PROP_POS_FRAMES, back_frame)
            if not self.Video.get(cv2.CAP_PROP_POS_FRAMES) == back_frame:
                print('Warning: OpenCV has failed to set back_frame to {0}. OpenCV value is {1}. Target value is {2}'.format(back_frame, self.Video.get(cv2.CAP_PROP_POS_FRAMES), currentFrame))

            back_frame = self.Video.get(cv2.CAP_PROP_POS_FRAMES)
            #print('back_frame is: {0}'.format(back_frame))
            while back_frame < currentFrame:
                self.Video.read()
                back_frame += 1
        #print('currentFrame: {0}'.format(currentFrame))
        #print('current position: {0}'.format(self.Video.get(cv2.CAP_PROP_POS_FRAMES)))
        assert self.Video.get(cv2.CAP_PROP_POS_FRAMES) == currentFrame, 'Frame position error'
        result, img = self.Video.read()
        if result is False:
            print('-Could not read frame, trying again')
            back_frame = max(currentFrame - 61, 0)
            self.Video.set(cv2.CAP_PROP_POS_FRAMES, back_frame)
            if not self.Video.get(cv2.CAP_PROP_POS_FRAMES) == back_frame:
                print('-Warning: OpenCV has failed to set back_frame to {0}. OpenCV value is {1}. Target value is {2}'.format(back_frame, self.Video.get(cv2.CAP_PROP_POS_FRAMES), currentFrame))
            back_frame = self.Video.get(cv2.CAP_PROP_POS_FRAMES)
            #print('-back_frame is: {0}'.format(back_frame))
            while back_frame < currentFrame:
                self.Video.read()
                back_frame += 1
            result, img = self.Video.read()

        # img = img[:, :, ::-1]  # bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Update
        self.PrevFrame = currentFrame
        self.PrevCamera = iCam
        self.PrevPart = iPart
        return img

class RowView:
    def __init__(self, array, row):
        self.array = array
        self.row = row

    def __getitem__(self, item):
        return self.array[self.row, item]

    def __len__(self):
        return self.array.shape[1]


def pose2bb(pose):

    renderThreshold = 0.05
    ref_pose = np.array([[0.,   0.], #nose
       [0.,   23.], # neck
       [28.,   23.], # rshoulder
       [39.,   66.], #relbow
       [45.,  108.], #rwrist
       [-28.,   23.], # lshoulder
       [-39.,   66.], #lelbow
       [-45.,  108.], #lwrist
       [20., 106.], #rhip
       [20.,  169.], #rknee
       [20.,  231.], #rankle
       [-20.,  106.], #lhip
       [-20.,  169.], #lknee
       [-20.,  231.], #lankle
       [5.,   -7.], #reye
       [11.,   -8.], #rear
       [-5.,  -7.], #leye
       [-11., -8.], #lear
       ])


    # Template bounding box
    ref_bb = np.array([[-50., -15.], #left top
                [50., 240.]])  # right bottom

    pose = np.reshape(pose,(18,3))
    valid = np.logical_and(np.logical_and(pose[:,0]!=0,pose[:,1]!=0), pose[:,2] >= renderThreshold)

    if np.sum(valid) < 2:
        bb = np.array([0, 0, 0, 0])
        print('got an invalid box')
        print(pose)
        return bb

    points_det = pose[valid,0:2]
    points_reference = ref_pose[valid,:]

    # 1a) Compute minimum enclosing rectangle

    base_left = min(points_det[:,0])
    base_top = min(points_det[:,1])
    base_right = max(points_det[:,0])
    base_bottom = max(points_det[:,1])

    # 1b) Fit pose to template
    # Find transformation parameters
    M = points_det.shape[0]
    B = points_det.flatten('F')
    A = np.vstack((np.column_stack((points_reference[:,0], np.zeros((M)), np.ones((M)),  np.zeros((M)))),
         np.column_stack((np.zeros((M)),  points_reference[:,1], np.zeros((M)),  np.ones((M))) )))


    params = np.linalg.lstsq(A,B)
    params = params[0]
    M = 2
    A2 = np.vstack((  np.column_stack( (ref_bb[:,0], np.zeros((M)), np.ones((M)),  np.zeros((M)))),
         np.column_stack( (np.zeros((M)),  ref_bb[:,1], np.zeros((M)),  np.ones((M)))) ))

    result = np.matmul(A2,params)

    fit_left = min(result[0:2])
    fit_top = min(result[2:4])
    fit_right = max(result[0:2])
    fit_bottom = max(result[2:4])

    # 2. Fuse bounding boxes
    left = min(base_left,fit_left)
    top = min(base_top,fit_top)
    right = max(base_right,fit_right)
    bottom = max(base_bottom,fit_bottom)

    left = left*1920
    top = top*1080
    right = right*1920
    bottom = bottom*1080

    height = bottom - top + 1
    width = right - left + 1

    bb = np.array([left, top, width, height])
    return bb

if __name__ == '__main__':
    Duke("/tmp/tst").download()