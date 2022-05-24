import detectron2, torchtext
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from sklearn.metrics.pairwise import cosine_similarity

import sys


def l2_norm(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def max_images_similarity(a, b_array):
    tmp_sim = cosine_similarity(a, b_array)[0]
    return tmp_sim


def convert_box2array(img, box, size_x=100, size_y=200):
    tmp = cv2.resize(img[box[1]: (box[1] + box[3]) // 2, box[0]: box[2]], dsize=(size_x, size_y)).reshape(-1,).tolist()
    return tmp

def isin(box, coord):
  if coord[0] > box[0] and coord[0] < box[2] and coord[1] > box[1] and coord[1] < box[3]:
    return True 
  return False

class PersDict:
    def __init__(self, threshold=0.6, max_idx=30, max_frame_diff=30, imgbias=0.2, fps=30, direction_boost=1.3, matchpoint_boost=1.005):
        self.entrDict = {}
        self.max_idx = max_idx
        self.threshold = threshold
        self.max_frame_diff = max_frame_diff
        self.imgbias = imgbias
        self.fps = fps
        self.direction_boost = direction_boost
        self.matchpoint_boost = matchpoint_boost

    def update(self, img, humanbox, i, matchpoints):
        human_center1 = [(humanbox[0] + humanbox[2]) / 2,
                         (humanbox[1] + humanbox[3]) / 2]
        human_center2 = [(3 * humanbox[0] + humanbox[2]) /
                         4, (humanbox[1] + humanbox[3]) / 2]
        matchpoints_ = [x for x in matchpoints if isin(humanbox, x['now'])]
        width, height = img.shape[0], img.shape[1]
        humanarray = convert_box2array(img, humanbox)
        idxs = [idx for idx in self.entrDict if self.entrDict[idx]['i'] != i
                and abs(self.entrDict[idx]['hc1'][0] - human_center1[0]) < width*self.imgbias
                and abs(self.entrDict[idx]['hc1'][1] - human_center1[1]) < height*self.imgbias and abs(self.entrDict[idx]['hc2'][0] - human_center2[0]) < width*self.imgbias
                and abs(self.entrDict[idx]['hc2'][1] - human_center2[1]) < height*self.imgbias and i - self.entrDict[idx]['i'] < self.fps/3]
        candidarray = [[self.entrDict[idx]['humanarray'], (self.entrDict[idx]['hc1'][0] - self.entrDict[idx]['prevhc1'][0], self.entrDict[idx]['hc1'][1] - self.entrDict[idx]['prevhc1'][1])]
                       for idx in self.entrDict if self.entrDict[idx]['i'] != i
                       and abs(self.entrDict[idx]['hc1'][0] - human_center1[0]) < width*self.imgbias
                       and abs(self.entrDict[idx]['hc1'][1] - human_center1[1]) < height*self.imgbias and abs(self.entrDict[idx]['hc2'][0] - human_center2[0]) < width*self.imgbias
                       and abs(self.entrDict[idx]['hc2'][1] - human_center2[1]) < height*self.imgbias and i - self.entrDict[idx]['i'] < self.fps/3]

        if len(idxs) == 0:
            idx = self.get_newidx()
        else:
            sims = max_images_similarity(np.array(humanarray).reshape(1, -1), np.array([x[0] for x in candidarray]))
            if np.max(sims) <= self.threshold:
                idx = self.get_newidx()
            else:
                for cnt, idx in enumerate(idxs):
                    if (human_center1[0] - self.entrDict[idx]['hc1'][0]) * candidarray[cnt][1][0] > 0 and \
                            (human_center1[1] - self.entrDict[idx]['hc1'][1]) * candidarray[cnt][1][1] > 0:
                        sims[cnt] *= self.direction_boost
                    for matchpoint in matchpoints_:
                      if isin(self.entrDict[idx]['boxcoord'], matchpoint['before']):
                        sims[cnt] *= self.matchpoint_boost
                idx = np.argmax(sims)
                idx = idxs[idx]

        if idx in self.entrDict:
            self.entrDict[idx] = {'hc1': human_center1,
                                  'hc2': human_center2,
                                  'humanarray': humanarray,
                                  'i': i,
                                  'prevhc1': self.entrDict[idx]['hc1'],
                                  'boxcoord': humanbox
                                  }
        else:
            self.entrDict[idx] = {'hc1': human_center1, 'hc2': human_center2, 'humanarray': humanarray, 'i': i, 'prevhc1': human_center1, 'boxcoord': humanbox}
        return idx

    def get_newidx(self):
        for i in range(self.max_idx):
            if i not in self.entrDict:
                return i
        return -1

    def clean_old(self, i):
        idxs_delete = []
        for idx in self.entrDict:
            if abs(i - self.entrDict[idx]['i']) > self.max_frame_diff:
                idxs_delete.append(idx)
        for idx in idxs_delete:
            self.entrDict.pop(idx)
            
            
class VideoProcesser:
  def __init__(self, threshold=0.6, max_idx=30, max_frame_diff=30, imgbias=0.2, direction_boost=1.3, matchpoint_boost=1.005):
    self.threshold = threshold
    self.max_idx = max_idx
    self.max_frame_diff = max_frame_diff
    self.imgbias = imgbias
    self.direction_boost = direction_boost
    self.matchpoint_boost = matchpoint_boost
    self.cfg = get_cfg()
    self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    self.predictor = DefaultPredictor(cfg)


  def process_video(self, videopath, fps=30, use_cv2tracker=False, use_matchpoints=True, save_video=True, out_name='movie.mp4'):
    persons_dict = PersDict(threshold=self.threshold, max_idx=self.max_idx,
                            max_frame_diff=self.max_frame_diff, 
                            imgbias=self.imgbias, 
                            direction_boost=self.direction_boost, 
                            matchpoint_boost=self.matchpoint_boost,
                            fps=fps)
    person_boxes = []
    if use_matchpoints:
      orb = cv2.ORB_create()
      bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    imgs = []
    i = -1
    if use_cv2tracker:
      multi_tracker = cv2.MultiTracker_create()
    cap = cv2.VideoCapture(videopath)
    if save_video:
      os.makedirs('./pictures/', exist_ok=True)
    while cap.isOpened():
      ret, img = cap.read()
      imgs.append(img)
      if ret:
        i += 1
        if not use_cv2tracker or i == 0:
          outputs = self.predictor(img)
          person_boxes = [tuple([int(y) for y in box.detach().cpu().numpy()]) for i, box in zip(outputs["instances"].pred_classes, outputs["instances"].pred_boxes) if i == 0]
        matchpoints = []
        if i > 1:
          if use_matchpoints:
            kp1, des1 = orb.detectAndCompute(imgs[-1],None)
            kp2, des2 = orb.detectAndCompute(imgs[-2],None)
            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)
            for match in matches:
              matchpoints.append({'now': kp1[match.queryIdx].pt, 'before': kp2[match.trainIdx].pt})
          if use_cv2tracker:
            success, bboxes = multi_tracker.update(img)
            if success:
              for idx,box in enumerate(bboxes):
                cv2.rectangle(img,(box[0], box[1]),(box[2], box[3]),(0,255,0),3)
                cv2.putText(img, str(idx),(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA)
        for j, box in enumerate(person_boxes):
            if i == 1 and use_cv2tracker:
              multi_tracker.add(cv2.TrackerCSRT_create(), img, box)
            elif not use_cv2tracker:
              idx = persons_dict.update(img, box, i, matchpoints)
              cv2.rectangle(img,(box[0], box[1]),(box[2], box[3]),(0,255,0),3)
              cv2.putText(img,str(idx),(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA)
        if save_video:
          cv2.imwrite(f'./pictures/img{i}.png', img)
        persons_dict.clean_old(i)
      else:
        break
    cap.release()
    os.system(f"ffmpeg -r {fps} -i ./pictures/img%01d.png -vcodec mpeg4 -y {out_name}")
    
    
    
if __name__ == '__main__':
    vp = VideoProcesser()
    vp.process_video(sys.argv[1], sys.argv[2], sys.argv[3])
