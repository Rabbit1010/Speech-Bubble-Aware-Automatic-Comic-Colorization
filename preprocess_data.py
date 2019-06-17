# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:09:59 2019

@author: Wei-Hsiang, Shen
"""

import os
import numpy as np
import cv2

# get all data path
file_list = []
in_dir_name = './comic_img/'
for filename in os.listdir(in_dir_name):
    if filename.lower().endswith('.png'):
        file_list.append(os.path.join(in_dir_name, filename))

# random shuffle and train/test split
VAL_SPLIT = 0.8
np.random.shuffle(file_list)
file_list_train = file_list[0:int(round(len(file_list)*VAL_SPLIT))]
file_list_val = file_list[int(round(len(file_list)*VAL_SPLIT)):]

# for training image
out_sketch_dir_name = './data/comic_sketch/train/'
out_img_dir_name = './data/comic_img/train/'
for file_path in file_list_train:
    filename = file_path.split('/')[-1]
    img_path = os.path.join(in_dir_name, filename)

    img = cv2.imread(img_path) # load input image
    out_path = os.path.join(out_img_dir_name, filename)
    cv2.imwrite(out_path, img)

    """Hand-sketch comic image generationg"""
    # Canny edge detection
    img_canny = cv2.Canny(img, 100, 300, apertureSize=3)
    img_canny[img_canny==0] = 5 # make the edge black
    img_canny[img_canny==255] = 0
    img_canny[img_canny==5] = 255

    # binarize the image
    img_black_part = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_black_part[img_black_part<=50] = 0
    img_black_part[img_black_part!=0] = 255

    # combine the edges and the black parts
    img_sketch = np.zeros_like(img)
    img_sketch[np.logical_and(img_black_part, img_canny)] = 255

    out_path = os.path.join(out_sketch_dir_name, filename)
    cv2.imwrite(out_path, img_sketch)

# for validaion image
out_sketch_dir_name = './data/comic_sketch/validation/'
out_img_dir_name = './data/comic_img/validation/'
for file_path in file_list_val:
    filename = file_path.split('/')[-1]
    img_path = os.path.join(in_dir_name, filename)

    img = cv2.imread(img_path) # load input image
    out_path = os.path.join(out_img_dir_name, filename)
    cv2.imwrite(out_path, img)

    """Hand-sketch comic image generationg"""
    # Canny edge detection
    img_canny = cv2.Canny(img, 100, 300, apertureSize=3)
    img_canny[img_canny==0] = 5 # make the edge black
    img_canny[img_canny==255] = 0
    img_canny[img_canny==5] = 255

    # binarize the image
    img_black_part = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_black_part[img_black_part<=50] = 0
    img_black_part[img_black_part!=0] = 255

    # combine the edges and the black parts
    img_sketch = np.zeros_like(img)
    img_sketch[np.logical_and(img_black_part, img_canny)] = 255

    out_path = os.path.join(out_sketch_dir_name, filename)
    cv2.imwrite(out_path, img_sketch)