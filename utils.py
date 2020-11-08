#!/usr/bin/env python
# coding: utf-8

import csv
import os
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt


SEVERSTAL_STEEL_IMAGE_WIDTH = 1600
SEVERSTAL_STEEL_IMAGE_HEIGHT = 256


def parse_segment(segment):
    """
    @brief parse EncodedPixels in train.csv
    
    @param segment EncodedPixels string, one row
    """
    segment = segment.split(' ')
    segment = [int(s) for s in segment]
    
    pixels = []
    for i in range(0, len(segment), 2):
        pixel = [segment[i], segment[i+1]]
        pixels.append(pixel)
    
    pixels = np.array(pixels, dtype=np.int)
    return pixels

def read_train_label(label_file):
    """
    @brief read severstal steel train label, train.csv
    
    @param label_file train.csv file path
    """
    images = []
    cls_ids = []
    segments = []
    with open(label_file, 'r') as f:
        data = csv.reader(f)
        
        idx = 0
        for row in data:
            # remove row 0, ImageId ClassId EncodedPixels
            if idx == 0:
                idx = idx + 1
                continue
            images.append(row[0])
            cls_ids.append(int(row[1]))
            segments.append(parse_segment(row[2]))
        
        return images, cls_ids, segments

def index2uv(index, width=SEVERSTAL_STEEL_IMAGE_WIDTH, height=SEVERSTAL_STEEL_IMAGE_HEIGHT):
    """
    @brief parse EncodedPixels index to (u, v) coordinate
    @details severstal steel EncodedPixels is column based
    
    @param index EncodedPixels
    @param width severstal steel data image width
    @param height severstal steel data image height
    """
    
    # data is column base
    v = index // height
    u = index % height
    
    return u, v

def draw_segment_label(img, cls_id, segment, with_color = False):
    """
    @brief draw segment label in label image
    
    @param img image
    @param cls_id label id
    @param segment segmentation pixels
    """
    color_map = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255]
    ]
    for seg in segment:
        for i in range(seg[0], seg[0]+seg[1], 1):
            u, v = index2uv(i-1)
            if with_color == True:
                img[u, v] = color_map[cls_id]
            else:
                img[u, v] = [cls_id] * 3

def generate_label_image(image_root, label_root, image_files, cls_ids, segments, with_color = False):
    """
    @brief generate label image
    
    @param image_root image root path
    @param label_root output label image root path
    @param image_files image file names read from label csv file
    @param cls_ids class ids of each segment, read from label csv file
    @param segments segment pixels
    @param with_color draw visible colorful label when it is true, otherwise draw label for train
    """
    
    for i in range(len(image_files)):
        image_file_name = image_files[i]
        image_file = os.path.join(image_root, image_file_name)
        label_file = os.path.join(label_root, image_file_name)
        
        src_image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        
        label_image = None
        if os.path.exists(label_file):
            label_image = cv2.imread(label_file, cv2.IMREAD_COLOR)
        else:
            label_image = np.zeros(src_image.shape, dtype=src_image.dtype)
        
        draw_segment_label(label_image, cls_ids[i], segments[i], with_color)
        
        cv2.imwrite(label_file, label_image)
        
        print('processed ', image_file_name, ', ', i, ' / ', len(image_files))

def vis_label_in_src_image(image_root, label_root, output_root):
    """
    @brief visualize colorful label in src image
    
    @param image_root src image root path
    @param label_root colorful label root path
    @param output_root output root path
    """
    image_files = os.listdir(image_root)
    
    for i in range(len(image_files)):
        image_file = image_files[i]
        image_file_path = os.path.join(image_root, image_file)
        label_file_path = os.path.join(label_root, image_file)
        output_file_path = os.path.join(output_root, image_file)
        
        src_image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        if not os.path.exists(label_file_path):
            # cv2.imwrite(output_file_path, src_image)
            continue
        else:
            label_image = cv2.imread(label_file_path, cv2.IMREAD_COLOR)
            output_image = cv2.addWeighted(src_image, 1.0, label_image, 0.5, 0)
            cv2.imwrite(output_file_path, output_image)
        
        print('processed ', image_file, ', ', i, ' / ', len(image_files))