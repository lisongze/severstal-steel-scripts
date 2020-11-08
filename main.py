#!/usr/bin/env python
# coding: utf-8

import csv
import os
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
from utils import *


data_root = './data/severstal-steel-defect-detection'   # set your own severstal steel data root path
train_images_path = os.path.join(data_root, 'train_images')
test_images_path = os.path.join(data_root, 'test_images')
train_label_file = os.path.join(data_root, 'train.csv')
train_labels_path = os.path.join(data_root, 'train_labels')
color_labels_path = os.path.join(data_root, 'color_labels')
vis_labels_path = os.path.join(data_root, 'vis_labels')

# read label csv and parse it
images, cls_ids, segments = read_train_label(train_label_file)

# generate label which used to train
generate_label_image(train_images_path, train_labels_path, images, cls_ids, segments, False)

# generate label with color to view
generate_label_image(train_images_path, color_labels_path, images, cls_ids, segments, True)

# draw label in source image to have a check
vis_label_in_src_image(train_images_path, color_labels_path, vis_labels_path)