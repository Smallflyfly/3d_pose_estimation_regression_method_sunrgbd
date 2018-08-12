# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from model.nms_wrapper import nms
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

import torch

import xml.etree.ElementTree as ET

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  _, scores, bbox_pred, rois, depth, rotation = net.test_image(blobs['data'], blobs['im_info'])
  
  boxes = rois[:, 1:5] / im_scales[0]
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  depth_pred = np.reshape(depth, [depth.shape[0], -1])
  rotation_pred = np.reshape(rotation, [rotation.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
    depth_pred = depth_pred * 0.1
    rotation_pred = rotation_pred * 0.5
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes, depth_pred, rotation_pred

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(torch.from_numpy(dets), thresh).numpy()
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def loadlabel(xmlname):
    tree = ET.parse(xmlname)
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.int32)
    # gt_classes = np.zeros((num_objs, 100), dtype=np.str)
    gt_classes = np.chararray(num_objs, 100)
    # gt_grid  = np.zeros((num_objs), dtype=np.int32) #fang
    # gt_ax = np.zeros((num_objs), dtype=np.float32) #fang
    # gt_ay = np.zeros((num_objs), dtype=np.float32) #fang
    gt_rotation = np.zeros((num_objs)).astype(float)
    gt_depth = np.zeros((num_objs)).astype(float)
    pi = 3.141593
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        position = obj.find('position')
        yp = float(position.find('yp').text)
        cls = obj.find('name').text
        if cls == 'desk':
          cls = 'table'
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls

        rotation = obj.find('orientation')
        ax = float(rotation.find('ax').text)
        ay = float(rotation.find('ay').text)
        axt1 = round(math.acos(ax) * 180.0 / pi)
        if axt1 < 0 or axt1 > 180.0:
          print('error')
          fang[-1]
        # ayt1 = int(math.acos(ay) * 180.0 / pi)

        if ax >= 0:
          if ay >= 0:
            xangle =  axt1
          else:
            xangle = 360 - axt1
        else:
          if ay >= 0:
            xangle = axt1
          else:
            xangle = 360 - axt1
        
        xangle = xangle + 90
        if xangle >= 360:
          xangle = xangle - 360
        gt_rotation[ix] = (xangle / 180.0) * pi 
        gt_depth[ix] = yp

    return boxes, gt_classes, gt_depth, gt_rotation

def cal_overlap(gt_box, cls_det):
  xmin = cls_det[0]
  ymin = cls_det[1]
  xmax = cls_det[2]
  ymax = cls_det[3]

  gtxmin = gt_box[0]
  gtymin = gt_box[1]
  gtxmax = gt_box[2]
  gtymax = gt_box[3]

  # print(xmin, ymin, xmax, ymax)
  # print(gtxmin, gtymin, gtxmax, gtymax)
  # fang[-1]

  if(xmin>=gtxmax or ymin>=gtymax or xmax<=gtxmin or ymax<=gtymin):
    overlap = 0.0

  else:
    x = []
    y = []
    x.append(gtxmin)
    x.append(gtxmax)
    x.append(xmin)
    x.append(xmax)
    x.sort()
    
    y.append(gtymin)
    y.append(gtymax)
    y.append(ymin)
    y.append(ymax)
    y.sort()

    # print(x)
    # print(y)
    # print(xmin, xmax, ymin, ymax)
    # print(gtxmin, gtxmax, gtymin, gtymax)
    gt_s = (gtxmax - gtxmin + 1.0) * (gtymax - gtymin + 1.0) * 1.0
    # print(gt_s)
    # print((x[2]-x[1]) * (y[2]- y[1]))
    overlap = (x[2] - x[1] + 1.0) * (y[2]- y[1] + 1.0) / gt_s
    # print(overlap)
    # fang[-1]
  
  return overlap

def evl(cls_dets, gt_box):
  all_overlap = np.zeros((cls_dets.shape[0], len(gt_box))).astype(float)
  for i in range(cls_dets.shape[0]):
    for j in range(len(gt_box)):
      overlap = cal_overlap(gt_box[j], cls_dets[i, :])
      all_overlap[i][j] = overlap
  return all_overlap

def visual_predit(cls_dets, gt_boxes, gt_classes, im):
  # print(cls_dets)
  # print(gt_boxes)
  # print(gt_classes)
  # fang[-1]
  for i in range(cls_dets.shape[0]):
    cv2.rectangle(im, (cls_dets[i, 0], cls_dets[i, 1]), (cls_dets[i, 2], cls_dets[i, 3]), (255, 255, 0), 1)
  for i in range(gt_boxes.shape[0]):
    cv2.rectangle(im, (gt_boxes[i, 0], gt_boxes[i, 1]), (gt_boxes[i, 2], gt_boxes[i, 3]), (255, 0, 0), 1)
  cv2.imshow('img', im)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  # fang[-1]


def test_net_fang(net, imdb, weights_filename, max_per_image=100, thresh=0.7):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)

  # print(imdb.image_index)
  # fang[-1]
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
    
  cls_num = [0, 0, 0, 0, 0, 0, 0]
  pcls_num = [0, 0, 0, 0, 0, 0, 0]
  tpcls_num = [0, 0, 0, 0, 0, 0, 0]
  med_depth = [[],[],[],[],[],[],[]]
  med_rotation = [[],[],[],[],[],[],[]]
  # pi_rotation = [[],[],[],[],[],[],[]]
  # tpgrid_num = [0, 0, 0, 0, 0, 0, 0]
  totalnum = 0
  all_class = ('__background__',  # always index 0
                    'chair', 'table', 'sofa', 'bed', 'shelf', 'cabinet')
  pi = 3.141593
  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))
    # print(imdb.image_path_at(i))
    # fang[-1]

    _t['im_detect'].tic()
    scores, boxes, depth, rotation = im_detect(net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # print(scores.shape)
    # print(boxes.shape)
    # print(grid_scores)
    # print(grid_scores.shape)
    # fang[-1]
    
    # print(grids_inds)
    filename = imdb.image_index[i]
    xmlfile = '/media/smallflyfly/Software/sunrgbd-position-rotation-regression-150k/data/VOCdevkit2007/VOC2007/Annotations/' + filename + '.xml'
    gt_boxes, gt_classes, gt_depth, gt_rotation = loadlabel(xmlfile)
    totalnum += len(gt_classes)
    all_boxes = [[] for _ in range(imdb.num_classes)]
    all_depth = [[] for _ in range(imdb.num_classes)]
    all_rotation = [[] for _ in range(imdb.num_classes)]
    # print(all_boxes)
    # fang[-1]
    for k in range(len(gt_classes)):
      cls = gt_classes[k]
      cls_i = all_class.index(cls)
      # print(cls_i)
      # fang[-1]
      all_boxes[cls_i].append(gt_boxes[k])
      all_depth[cls_i].append(gt_depth[k])
      all_rotation[cls_i].append(gt_rotation[k])
      cls_num[cls_i] += 1
    # print(all_boxes)
    # print(all_depth)
    # print(all_rotation)
    # fang[-1]
    # print(filename)
    # skip j = 0, because it's the background class
    # grids_inds = grid_scores.argmax(1)
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      
      cls_depth = depth[inds, j*1:(j+1)*1]
      cls_rotation = rotation[inds, j*1:(j+1)*1]

      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(torch.from_numpy(cls_dets), cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
      cls_dets = cls_dets[keep, :]
      depth_dets = cls_depth[keep, :]
      rotation_dets = cls_rotation[keep, :]
      # after nms
      # then choose prob >= 0.5 to left
      # print(cls_dets)
      # print(depth_dets)
      # print(rotation_dets)
      # fang[-1]


      if len(cls_dets) == 0:
        continue
      pcls_num[j] += len(keep)
      if len(all_boxes[j]) == 0:
        continue
      # print('**********************', cls_dets)
      # fang[-1]

      # visual_predit(cls_dets, gt_boxes, gt_classes, im)
      # print(all_boxes[j])
      # print(len(cls_dets), len(all_boxes[j]))
      all_overlap = evl(cls_dets, all_boxes[j])
      # print(all_overlap)
      # fang[-1]
      max_overlap, ind= all_overlap.max(1), all_overlap.argmax(1) # max overlap num
      # print(max_overlap, ind)
      # ids = np.where(max_overlap > 0.5)
      # # print(ids)
      # ind = ind[ids]
      # max_overlap = max_overlap[ids]
      # print(ind)
      for ii in range(len(max_overlap)):
        if max_overlap[ii] < 0.5:
          ind[ii] = -1
          continue
        for jj in range(ii+1, len(ind)):
          if ind[jj] == ind[ii]:
            if max_overlap[ii] < max_overlap[jj]:
              ind[ii] = -1
              break
            else:
              ind[jj] = -1
      # print(ind)
      # fang[-1]

      # for ii in range()
      # gtgrid = list(all_grids[j])
      gtdepth = list(all_depth[j])
      gtrotation = list(all_rotation[j])
      # print(gt_depth)
      # print(gt_rotation)
      # fang[-1]
      # print(gtgrid) # gt grid
      # print(grid_dets) # predict grid
      # print(ind)
      p_depth = []
      # pi_rotation = []
      p_rotation = []
      for indd in ind:
        if indd != -1:
          p_depth.append(gtdepth[indd])
          p_rotation.append(gtrotation[indd])
        else:
          p_depth.append(-1)
          p_rotation.append(-1)
      # print(p_depth)
      # print(p_rotation)
      # print(p_grid)
      # fang[-1]
      # print(max_overlap)
      ids = np.where(ind > -1)
      ind = ind[ids]
      # print(ind)
      # fang[-1]
      # print(p_rotation)
      # print(rotation_dets)
      tpcls_num[j] += len(ind)
      for index, pp in enumerate(p_depth):
        if pp == -1:
          continue
        else:
          detal_depth = abs(pp - depth_dets[index])
          med_depth[j].append(detal_depth)
      
      for index, qq in enumerate(p_rotation):
        if qq == -1:
          continue
        else:
          # if qq == 0.0:
          #   print(filename)
          #   qq = 0.01
          #   continue
          # if rotation_dets[index] == 0.0:
          #   rotation_dets[index] == 0.01
          #   continue
          # if rotation_dets[index] < 0:
          #   continue
          # print(qq, rotation_dets[index])
          # qq = qq * 180 / pi
          # rotation_dets[index] = rotation_dets[index] * 180 / pi
            tt = rotation_dets[index]
            detal_rotation1 = abs(qq - tt)
            detal_rotation2 = 2.0 * pi - detal_rotation1
            detal_rotation = min(detal_rotation1, detal_rotation2)
            med_rotation[j].append(detal_rotation) 
          #log---------------------------------------------------------
          # print(qq)
          # tt = rotation_dets[index]
          # print(tt)

          # gtR_det = np.zeros((3,3)).astype(float)
          # pR_det = np.zeros((3,3)).astype(float)
          # print(gtR_det)
          # print(pR_det)
          # gtR_det[0, 0] = math.cos(qq)
          # gtR_det[0, 1] = math.sin(qq)
          # gtR_det[1, 0] = -1 * math.sin(qq)
          # gtR_det[1, 1] = math.cos(qq)
          # gtR_det[2, 2] = 1.0
          # print(gtR_det)
          # pR_det[0, 0] = math.cos(tt)
          # pR_det[0, 1] = -1.0 * math.sin(tt)
          # pR_det[1, 0] = math.sin(tt)
          # pR_det[1, 1] = math.cos(tt)
          # pR_det[2, 2] = 1.0
          # print(pR_det)
          # # fang[-1]
          # dotR = np.dot(gtR_det, pR_det)
          # print(dotR)
          # dotR[0, 0] = math.log(dotR[0, 0], 10)
          # dotR[0, 0] = math.log(dotR[0, 1], 10)
          # dotR[1, 0] = math.log(dotR[1, 0], 10)
          # dotR[1, 1] = math.log(dotR[1, 1], 10)
          # dotR[2, 2] = math.log(dotR[2, 2], 10)
          # print(dotR)
          # fang[-1]
          #log---------------------------------------------------------


          # detal_rotation = math.log(qq*rotation_dets[index], 10)
          # print(qq)
          # print(rotation_dets[index])
          # print(detal_rotation)
          # fang[-1]
          # med_rotation[j].append(detal_rotation)
      # fang[-1]
      # print(med_depth)
      # fang[-1]
    print('{:d} / 500'.format(i+1))
  depth_result = [[],[],[],[],[],[],[]]
  rotation_result = [[],[],[],[],[],[],[]]
  depth_rotation_result = [[],[],[],[],[],[],[]]
  rotation_num = [[],[],[],[],[],[],[]]
  pi_rotation = [[],[],[],[],[],[],[]]
  for ii in range(1, 7):
    tnum = 0
    print(len(med_depth[ii]))
    rotation_num[ii].append(len(med_depth[ii]))
    for jj in range(len(med_depth[ii])):
      p_angle = med_rotation[ii][jj] * 180.0 / pi
      if med_depth[ii][jj] <= 0.5 and p_angle <= 30.0:
        tnum += 1
    depth_rotation_result[ii].append(tnum)
  
  print(depth_rotation_result)
  ans_rotation = np.array(depth_rotation_result[1:]).astype(float) / np.array(rotation_num[1:]).astype(float)
  print(ans_rotation)
  # fang[-1]
  rotation_20metric = [[],[],[],[],[],[],[]]
  for ii in range(1, 7):
    tnum = 0
    t20num = 0
    for jj in range(len(med_rotation[ii])):
      p_angle = med_rotation[ii][jj] * 180.0 / pi
      if p_angle <= 30.0:
        tnum += 1
      if p_angle <= 10.0:
        t20num += 1
    pi_rotation[ii].append(tnum)
    rotation_20metric[ii].append(t20num)
  print(pi_rotation)
  print(rotation_20metric)
  ans_pi_rotation = np.array(pi_rotation[1:]).astype(float) / np.array(rotation_num[1:]).astype(float)
  # print(ans_pi_rotation)
  ans_20_m_rotation = np.array(rotation_20metric[1:]).astype(float) / np.array(rotation_num[1:]).astype(float)
  print(ans_20_m_rotation)
  fang[-1]


  med_result_rotation = [[],[],[],[],[],[],[]]
  for ii in range(1, 7):
    tmp = np.array(med_rotation[ii]).astype(float)
    tmp = sorted(tmp)
    half = len(tmp) // 2
    med_rotation_result = (tmp[half] + tmp[~half]) / 2
    med_rotation_result = med_rotation_result * 180.0 / pi
    med_result_rotation[ii].append(med_rotation_result)
  print(med_result_rotation)
  
  med_result_depth = [[],[],[],[],[],[],[]]
  for ii in range(1, 7):
    tmp = np.array(med_depth[ii]).astype(float)
    
    tmp = sorted(tmp)
    # print(tmp)
    half = len(tmp) // 2
    med_depth_result = (tmp[half] + tmp[~half]) / 2
    med_result_depth[ii].append(med_depth_result)
  print(med_result_depth)
  # print(med_depth[1])
  # for ip in range(1,7):
  #   for iq in range(len(med_depth[ip])):
  #     med_depth[ip][iq] = abs(med_depth[ip][iq])
  #     if med_depth[ip][iq] < 0:
  #       print('error')
  #       fang[-1]
  #   # print(med_depth[1])
  #   tmp = np.array(med_depth[ip]).astype(float)
  #   tmp = tmp / math.sqrt(2.0)
  #   tmp = sorted(tmp)
  #   half = len(tmp) // 2
  #   med_depth_result = (tmp[half] + tmp[~half]) / 2
  #   depth_result[ip].append(med_depth_result)
  # print(depth_result)
  # fang[-1]

  # tpi_rotation = [[],[],[],[],[],[],[]]
  # for jp in range(1,7):
  #   pi_rotation[jp].append(len(med_rotation[jp]))
  #   for jq in range(len(med_rotation[jp])):
  #     med_rotation[jp][jq] = abs(med_rotation[jp][jq])
  #     if med_rotation[jp][jq] < 0:
  #       print('error')
  #       fang[-1]
  #   tmp = np.array(med_rotation[jp]).astype(float)
  #   tmp = tmp / math.sqrt(2.0)
  #   tmp = sorted(tmp)
  #   print(tmp)
  #   half = len(tmp) // 2
  #   med_rotation_result = (tmp[half] + tmp[~half]) / 2
  #   rotation_result[jp].append(med_rotation_result)
  #   tmp2 = list(tmp)
  #   t_num = 0
  #   for kk in tmp2:
  #     if kk <= pi / 6.0:
  #       t_num += 1
  #   tpi_rotation[jp].append(t_num)
  # print(pi_rotation)
  # print(tpi_rotation)

  # print(rotation_result)


  cls_num_det = np.array(cls_num[1:]).astype(float)
  pcls_num_det = np.array(pcls_num[1:]).astype(float)
  tpcls_num_det = np.array(tpcls_num[1:]).astype(float)
  # tpgrid_num_det = np.array(tpgrid_num[1:]).astype(float)

  cls_pre = tpcls_num_det / pcls_num_det
  cls_mpre = np.mean(cls_pre[0:5])
  print(cls_pre)
  print(cls_mpre)
  # grid_pre = tpgrid_num_det / pcls_num_det
  # grid_mpre = np.mean(grid_pre[:-1])
  # print(grid_pre)
  # print(grid_mpre)
  fang[-1]
      
      


      
      # print(cls_dets)
      # print(grid_dets)
      # fang[-1]
      

    # print(all_boxes)
    # print(all_boxes.shape)
    # fang[-1]

    # Limit to max_per_image detections *over all classes*
    # if max_per_image > 0:
    #   image_scores = np.hstack([all_boxes[j][i][:, -1]
    #                 for j in range(1, imdb.num_classes)])
    #   if len(image_scores) > max_per_image:
    #     image_thresh = np.sort(image_scores)[-max_per_image]
    #     for j in range(1, imdb.num_classes):
    #       keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
    #       all_boxes[j][i] = all_boxes[j][i][keep, :]
    # _t['misc'].toc()

    # print(all_boxes)
    # fang[-1]


    # print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
    #     .format(i + 1, num_images, _t['im_detect'].average_time(),
    #         _t['misc'].average_time()))

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

def test_net(net, imdb, weights_filename, max_per_image=100, thresh=0.):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    scores, boxes, grid_scores= im_detect(net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(torch.from_numpy(cls_dets), cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # print(all_boxes)
    # print(all_boxes.shape)
    # fang[-1]

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    # print(all_boxes)
    # fang[-1]


    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time(),
            _t['misc'].average_time()))

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

