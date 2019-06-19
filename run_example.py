# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 22:51:51 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import os
import cv2

import tensorflow as tf
from model import Low_res_colorizer, Polishing_network_small
from speech_bubble_segmentation import Speech_Bubble_segmentation
from data_generator import load_and_preprocess_image

if os.path.isdir("results")==False:
    os.mkdir('results')

# Load models and its pretrained weights
model_low_res_colorizer = Low_res_colorizer()
model_low_res_colorizer.load_weights('./checkpoints/low_res_colorizer_weights.h5')
model_polishing_network = Polishing_network_small()
model_polishing_network.load_weights('./checkpoints/polishing_network_small_weights.h5')

# Read all the files into a list
image_dir = "./examples/"
image_list = []
for filename in os.listdir(image_dir):
    if filename.lower().endswith("sketch.png"):
        image_list.append(os.path.join(image_dir, filename))

# Construct input tensor dataset
input_tensors = tf.data.Dataset.from_tensor_slices(image_list)
# Preprocess the images
input_tensors = input_tensors.map(lambda x: load_and_preprocess_image(x, shape=[256, 256], gray=True))
input_tensors = input_tensors.batch(4)

# Model feed forward (inference)
for full_res_gray_img in input_tensors.take(1):
    low_res_color_img = model_low_res_colorizer(full_res_gray_img)
    high_res_color_img = model_polishing_network((full_res_gray_img, low_res_color_img))

    # For each image
    for i_img in range(4):
        # Plot and save image
        img = cv2.cvtColor(full_res_gray_img.numpy()[i_img], cv2.COLOR_BGR2RGB)
        cv2.imshow("Input full-resolution gray image", img)
        cv2.waitKey(0)
        cv2.imwrite('./results/Input full-resolution gray image {}.png'.format(i_img), img*255)

        img = cv2.cvtColor(low_res_color_img.numpy()[i_img], cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256,256))
        cv2.imshow("Output low-resolution color image", img)
        cv2.waitKey(0)
        cv2.imwrite('./results/Output low-resolution color image {}.png'.format(i_img), img*255)

        img = cv2.cvtColor(high_res_color_img.numpy()[i_img], cv2.COLOR_BGR2RGB)
        cv2.imshow("Output high-resolution color image", img)
        cv2.waitKey(0)
        cv2.imwrite('./results/Output high-resolution color image {}.png'.format(i_img), img*255)

        # Speech bubble segmentation
        speech_bubble_segmentation, text_detected_image = Speech_Bubble_segmentation(img_path='./results/Output high-resolution color image {}.png'.format(i_img))

        # Plot and save image
        cv2.imshow("Output text-detected image", text_detected_image)
        cv2.waitKey(0)
        cv2.imwrite('./results/Output text-detected image {}.png'.format(i_img), text_detected_image)

        cv2.imshow("Output speech bubble segmentation image", speech_bubble_segmentation)
        cv2.waitKey(0)
        cv2.imwrite('./results/Output speech bubble segmentation image {}.png'.format(i_img), speech_bubble_segmentation)

        # Filter-out color in speech bubbles
        final_image = high_res_color_img.numpy()[i_img]
        gray_image = full_res_gray_img.numpy()[i_img]
        mask = np.array(speech_bubble_segmentation, dtype='bool')
        final_image[mask] = gray_image[mask]

        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Final output image", final_image)
        cv2.waitKey(0)
        cv2.imwrite('./results/Final output image {}.png'.format(i_img), final_image*255)