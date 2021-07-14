# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import ray
import multiprocessing

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

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

def get_state_dict(checksession, checkepoch, checkpoint):
  import os.path as osp
  import sys

  def add_path(path):
      if path not in sys.path:
          sys.path.insert(0, path)

  this_dir = osp.dirname(__file__)

  # Add lib to PYTHONPATH
  lib_path = osp.join(this_dir, 'lib')
  add_path(lib_path)

  coco_path = osp.join(this_dir, 'data', 'coco', 'PythonAPI')
  add_path(coco_path)

  lr = cfg.TRAIN.LEARNING_RATE
  momentum = cfg.TRAIN.MOMENTUM
  weight_decay = cfg.TRAIN.WEIGHT_DECAY

  cfg_from_file("detection/cfgs/res101.yml")

  cfg.USE_GPU_NMS = True
  np.random.seed(cfg.RNG_SEED)

  # load model
  model_dir = "models/res101_handobj_100K/pascal_voc"
  if not os.path.exists(model_dir):
    raise Exception('There is no input directory for loading network from ' + model_dir)
  load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(checksession, checkepoch, checkpoint))
  checkpoint = torch.load(load_name)

  model_id = ray.put(checkpoint['model'])
  
  return model_id

def detection_init(model_id):

  pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
  set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 

  # Init Network
  gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
  RCNNActor = ray.remote(num_gpus=1, num_cpus=int(float(multiprocessing.cpu_count())/float(len(gpus)))-1)(resnet)
  fasterRCNN = RCNNActor.remote(pascal_classes, 101, pretrained=False, class_agnostic=False)

  ray.wait([fasterRCNN.create_architecture.remote()])
  
  ray.wait([fasterRCNN.load_state_dict.remote(model_id)])

  print('Load model successfully!')

  return fasterRCNN


def detection(result):
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, loss_list, im_info, im_scales, im = result

    thresh_hand = 0.5 
    thresh_obj = 0.5

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    # extact predicted params
    contact_vector = loss_list[0][0] # hand contact state info
    offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
    lr_vector = loss_list[2][0].detach() # hand side info (left/right)

    # get hand contact 
    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

    # get hand side 
    lr = torch.sigmoid(lr_vector) > 0.5
    lr = lr.squeeze(0).float()

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                      + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    det_toc = time.time()
    misc_tic = time.time()
    im2show = np.copy(im)
    obj_dets, hand_dets = None, None
    for j in xrange(1, len(pascal_classes)):
        # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
        if pascal_classes[j] == 'hand':
          inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
        elif pascal_classes[j] == 'targetobject':
          inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)

        # if there is det
        if inds.numel() > 0:
          cls_scores = scores[:,j][inds]
          _, order = torch.sort(cls_scores, 0, True)
          cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
          
          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
          cls_dets = cls_dets[order]
          keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
          cls_dets = cls_dets[keep.view(-1).long()]
          if pascal_classes[j] == 'targetobject':
            obj_dets = cls_dets.cpu().numpy()
          if pascal_classes[j] == 'hand':
            hand_dets = cls_dets.cpu().numpy()
          
    im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)

    misc_toc = time.time()
    nms_time = misc_toc - misc_tic
    im2show = im2show.convert('RGB')
    open_cv_image = np.array(im2show)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

    # total_toc = time.time()
    # total_time = total_toc - total_tic
    # frame_rate = 1 / total_time
    # print('Pure Det Frame rate:', frame_rate)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    return (hand_dets, open_cv_image)