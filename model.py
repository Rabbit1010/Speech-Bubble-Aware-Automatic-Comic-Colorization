# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 00:39:56 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from tensorflow.keras import layers

class IdentityBlock(object):
    """
    Identity block in ResNet
    No conv layer at shortcut (skip connection)
    """
    def __init__(self, num_feature):
        self.num_feature = num_feature

    def __call__(self, x):
        shortcut = x

        x = layers.Conv2D(self.num_feature, kernel_size=(1,1), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(self.num_feature, kernel_size=(3,3), use_bias=False, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(self.num_feature, kernel_size=(1,1), use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)

        return x

class ConvBlock(object):
    """
    Convolution block in ResNet
    Contain conv layer at shortcut (projection connection)
    """
    def __init__(self, num_feature_in, num_feature_out, strides=(1,1)):
        self.num_feature_in = num_feature_in
        self.num_feature_out = num_feature_out
        self.strides = strides

    def __call__(self, x):
        shortcut = x

        x = layers.Conv2D(self.num_feature_in, kernel_size=(1,1), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(self.num_feature_in, kernel_size=(3,3), use_bias=False, strides=self.strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(self.num_feature_out, kernel_size=(1,1), use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        # projection connection with 1x1 convolution layer and the same strides
        shortcut = layers.Conv2D(self.num_feature_out, kernel_size=(1, 1), use_bias=False, strides=self.strides)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x

class Conv_Batch_LReLu(object):
    def __init__(self, num_features, strides=(1,1)):
        self.num_features = num_features
        self.strides = strides

    def __call__(self, x):
        x = layers.Conv2D(self.num_features, kernel_size=(3,3), strides=self.strides, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        return x

class Conv_Batch_ReLu(object):
    def __init__(self, num_features, strides=(1,1)):
        self.num_features = num_features
        self.strides = strides

    def __call__(self, x):
        x = layers.Conv2D(self.num_features, kernel_size=(3,3), strides=self.strides, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

def Low_res_colorizer_new():
    """
    Input:  Gray-scale image in full resolution
    Output: Colorized image in low resolution
    """
    inputs = tf.keras.Input(shape=(256, 256, 1))

    x = inputs

    # Scaling down
    x = Conv_Batch_ReLu(64)(x)
    x = ConvBlock(64, 128, (2,2))(x)
    x = IdentityBlock(128)(x)
    x = IdentityBlock(128)(x)
    x = ConvBlock(128, 256, (2,2))(x)
    x = IdentityBlock(256)(x)
    x = IdentityBlock(256)(x)
    x = ConvBlock(256, 512, (2,2))(x)
    x = IdentityBlock(512)(x)
    x = IdentityBlock(512)(x)

    # Colorization network
    x = ConvBlock(512, 256)(x)
    x = IdentityBlock(256)(x)
    x = ConvBlock(256, 128)(x)
    x = IdentityBlock(128)(x)
    x = ConvBlock(128, 64)(x)
    x = IdentityBlock(64)(x)
    x = ConvBlock(64, 32)(x)
    x = IdentityBlock(32)(x)

    x = layers.Conv2D(3, (3,3), padding='same', activation='relu')(x)

    outputs = x

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name='low_res_colorizer')

    return model

def Colorizer():
    """
    Input:  Gray-scale image in full resolution
    Output: Colorized image in full resolution
    """
    inputs = tf.keras.Input(shape=(256, 256, 1))

    x = inputs

    # Low-level features network
    x = Conv_Batch_LReLu(64)(x)
    x = ConvBlock(64, 128, (2,2))(x)

    x = Conv_Batch_LReLu(64, (2,2))(x)
    x = Conv_Batch_LReLu(128)(x)
    x = Conv_Batch_LReLu(128, (2,2))(x)
    x = Conv_Batch_LReLu(256)(x)
    x = Conv_Batch_LReLu(256, (2,2))(x)
    x = Conv_Batch_LReLu(512)(x)

    # Mid-Level features network
    x = Conv_Batch_LReLu(512)(x)
    x = Conv_Batch_LReLu(256)(x)

    # Colorization network
    x = Conv_Batch_LReLu(128)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = Conv_Batch_LReLu(64)(x)
    x = Conv_Batch_LReLu(64)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = Conv_Batch_LReLu(32)(x)
    x = Conv_Batch_LReLu(32)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = Conv_Batch_LReLu(16)(x)
    x = Conv_Batch_LReLu(3)(x)

    outputs = x
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name='low_res_colorizer')

    return model

def Low_res_colorizer():
    """
    Input:  Gray-scale image in full resolution
    Output: Colorized image in low resolution
    """
    inputs = tf.keras.Input(shape=(256, 256, 1))

    # Low-level features network
    x = inputs

    x = Conv_Batch_ReLu(64, (2,2))(x)
    x = Conv_Batch_ReLu(128)(x)
    x = Conv_Batch_ReLu(128, (2,2))(x)
    x = Conv_Batch_ReLu(256)(x)
    x = Conv_Batch_ReLu(256, (2,2))(x)
    x = Conv_Batch_ReLu(512)(x)

    # Colorization network
    x = Conv_Batch_ReLu(512)(x)
    x = Conv_Batch_ReLu(256)(x)
    x = Conv_Batch_ReLu(128)(x)
    x = Conv_Batch_ReLu(64)(x)
    x = Conv_Batch_ReLu(32)(x)
    x = layers.Conv2D(3, (3,3), padding='same', activation='relu')(x)

    outputs = x

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name='low_res_colorizer')

    return model

def Polishing_network():
    """
    Input:  Gray-scale image in full resolution (256,256,1)
            Low-resolution color image in full resolution (32,32,3)
    Output: Colorized image in full resolution (256,256,3)
    """
    inputs_gray = tf.keras.Input(shape=(256, 256, 1))
    inputs_color = tf.keras.Input(shape=(32, 32, 3))
    inputs_color_up = tf.image.resize(inputs_color, size=(256,256))

    x = tf.concat([inputs_gray, inputs_color_up], axis=-1)

    # encoder
    for num_filters in list([64, 128, 256, 512]):
        x = layers.Conv2D(num_filters, kernel_size=(3,3), strides=(2,2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    for _ in range(5):
        x = layers.Conv2D(512, kernel_size=(3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    # decoder
    for num_filters in list([512, 256, 128, 64]):
        x = layers.UpSampling2D(size=(2,2))(x)
        x = layers.Conv2D(num_filters, kernel_size=(3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.Conv2D(3, kernel_size=(3,3), padding='same')(x)
    x = layers.ReLU()(x)

    outputs = x

    model = tf.keras.Model(inputs=[inputs_gray, inputs_color], outputs=[outputs], name='polishing_network')
    return model

def Polishing_network_small():
    """
    Input:  Gray-scale image in full resolution (256,256,1)
            Low-resolution color image in full resolution (32,32,3)
    Output: Colorized image in full resolution (256,256,3)
    """
    inputs_gray = tf.keras.Input(shape=(256, 256, 1))
    inputs_color = tf.keras.Input(shape=(32, 32, 3))
    inputs_color_up = tf.image.resize(inputs_color, size=(256,256))

    x = tf.concat([inputs_gray, inputs_color_up], axis=-1)

    # encoder
    for _features in list([64, 128, 256]):
        x = Conv_Batch_LReLu(64, (2,2))(x)
        x = Conv_Batch_LReLu(_features*2)(x)

    for _ in range(5):
        x = Conv_Batch_LReLu(512)(x)

    # decoder
    for _features in list([256, 128]):
        x = layers.UpSampling2D(size=(2,2))(x)
        x = Conv_Batch_ReLu(_features*2)(x)
        x = Conv_Batch_ReLu(_features)(x)

    x = layers.UpSampling2D(size=(2,2))(x)
    x = tf.concat([x, inputs_gray], axis=-1)
    x = Conv_Batch_ReLu(128)(x)
    x = Conv_Batch_ReLu(64)(x)

    x = layers.Conv2D(3, kernel_size=(3,3), padding='same', activation='relu')(x)

    outputs = x

    model = tf.keras.Model(inputs=[inputs_gray, inputs_color], outputs=[outputs], name='polishing_network')
    return model
