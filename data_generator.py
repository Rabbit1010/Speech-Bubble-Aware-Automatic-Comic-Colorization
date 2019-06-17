# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:08:21 2019

@author: Wei-Hsiang, Shen
"""
import tensorflow as tf
import os


def load_and_preprocess_image(path, shape, gray=False, RGB=False):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3) # exclude alpha channel
    image = tf.image.resize(image, shape)
    image /= 255.0

    if gray==True:
        image = tf.image.rgb_to_grayscale(image)
        return image

    if RGB==True:
        return image

def Get_filepath_list():
    # Get all file path
    sketch_list_train = []
    _dir = "./data/comic_sketch/train/"
    for filename in os.listdir(_dir):
        if filename.lower().endswith(".png"):
            sketch_list_train.append(os.path.join(_dir, filename))

    sketch_list_val = []
    _dir = "./data/comic_sketch/validation/"
    for filename in os.listdir(_dir):
        if filename.lower().endswith(".png"):
            sketch_list_val.append(os.path.join(_dir, filename))

    img_list_train = []
    _dir = "./data/comic_img/train/"
    for filename in os.listdir(_dir):
        if filename.lower().endswith(".png"):
            img_list_train.append(os.path.join(_dir, filename))

    img_list_val = []
    _dir = "./data/comic_img/validation/"
    for filename in os.listdir(_dir):
        if filename.lower().endswith(".png"):
            img_list_val.append(os.path.join(_dir, filename))

    assert len(sketch_list_train)==len(img_list_train)
    assert len(sketch_list_val)==len(img_list_val)

    return sketch_list_train, img_list_train, sketch_list_val, img_list_val

def Generate_dataset_colorizer(BATCH_SIZE=32):
    sketch_list_train, img_list_train, sketch_list_val, img_list_val = Get_filepath_list()

    # make tf.dataset
    X_train_path_ds = tf.data.Dataset.from_tensor_slices(sketch_list_train)
    y_train_path_ds = tf.data.Dataset.from_tensor_slices(img_list_train)
    X_val_path_ds = tf.data.Dataset.from_tensor_slices(sketch_list_val)
    y_val_path_ds = tf.data.Dataset.from_tensor_slices(img_list_val)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    X_train_ds = X_train_path_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256], gray=True), num_parallel_calls=AUTOTUNE)
    y_train_ds = y_train_path_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256], RGB=True), num_parallel_calls=AUTOTUNE)
    X_val_ds = X_val_path_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256], gray=True), num_parallel_calls=AUTOTUNE)
    y_val_ds = y_val_path_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256], RGB=True), num_parallel_calls=AUTOTUNE)

    train_ds = tf.data.Dataset.zip((X_train_ds, y_train_ds))
    val_ds = tf.data.Dataset.zip((X_val_ds, y_val_ds))

    # Image augmentation
    @tf.function
    def Augmentation(img1, img2):
        if tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)<=0.5:
            img1 = tf.image.flip_left_right(img1)
            img2 = tf.image.flip_left_right(img2)
        return img1, img2
    train_ds = train_ds.map(Augmentation, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(Augmentation, num_parallel_calls=AUTOTUNE)

    print("Generated tf.data with BATCH_SIZE = {} ".format(BATCH_SIZE))
    print("Training data count: {}".format(len(sketch_list_train)))
    print("Validation data count: {}".format(len(sketch_list_val)))
    print("train_ds:", train_ds)
    print("valn_ds:", val_ds)

    # Add dataset settings using tf.data API
    train_ds = train_ds.shuffle(buffer_size=len(sketch_list_train)) # buffer size as larage as the datset ensures that the data is completely shuffled
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) # lets the dataset fetch batches in the background whilte the model is training

    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def Generate_dataset_polishing_network(BATCH_SIZE=32):
    """
    X1 : hand sketch image (256,256,1)
    X2 : low-res color image (32,32,3)
    y : full-res color image (256,256,3)
    """
    sketch_list_train, img_list_train, sketch_list_val, img_list_val = Get_filepath_list()

    # make tf.dataset
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    X1_train_ds = tf.data.Dataset.from_tensor_slices(sketch_list_train)
    X1_train_ds = X1_train_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256], gray=True), num_parallel_calls=AUTOTUNE)
    X1_val_ds = tf.data.Dataset.from_tensor_slices(sketch_list_val)
    X1_val_ds = X1_val_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256], gray=True), num_parallel_calls=AUTOTUNE)

    X2_train_ds = tf.data.Dataset.from_tensor_slices(img_list_train)
    X2_train_ds = X2_train_ds.map(lambda x: load_and_preprocess_image(x, shape=[32, 32], RGB=True), num_parallel_calls=AUTOTUNE)
    X2_val_ds = tf.data.Dataset.from_tensor_slices(img_list_val)
    X2_val_ds = X2_val_ds.map(lambda x: load_and_preprocess_image(x, shape=[32, 32], RGB=True), num_parallel_calls=AUTOTUNE)

    y_train_ds = tf.data.Dataset.from_tensor_slices(img_list_train)
    y_train_ds = y_train_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256], RGB=True), num_parallel_calls=AUTOTUNE)
    y_val_ds = tf.data.Dataset.from_tensor_slices(img_list_val)
    y_val_ds = y_val_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256], RGB=True), num_parallel_calls=AUTOTUNE)

    # Image augmentation
    @tf.function
    def Flip(img1):
        img1 = tf.image.flip_left_right(img1)
        return img1

    if tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)<=0.5:
        X1_train_ds = X1_train_ds.map(Flip, num_parallel_calls=AUTOTUNE)
        X2_train_ds = X2_train_ds.map(Flip, num_parallel_calls=AUTOTUNE)
        y_train_ds = y_train_ds.map(Flip, num_parallel_calls=AUTOTUNE)

    # combine data and label dataset
    train_ds = tf.data.Dataset.zip((X1_train_ds, X2_train_ds))
    train_ds = tf.data.Dataset.zip((train_ds, y_train_ds))
    val_ds = tf.data.Dataset.zip((X1_val_ds, X2_val_ds))
    val_ds = tf.data.Dataset.zip((val_ds, y_val_ds))

    print("Generated tf.data with BATCH_SIZE = {} ".format(BATCH_SIZE))
    print("Training data count: {}".format(len(sketch_list_train)))
    print("Validation data count: {}".format(len(sketch_list_val)))
    print("train_ds:", train_ds)
    print("valn_ds:", val_ds)

    # Add dataset settings using tf.data API
    train_ds = train_ds.shuffle(buffer_size=len(sketch_list_train)) # buffer size as larage as the datset ensures that the data is completely shuffled
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) # lets the dataset fetch batches in the background whilte the model is training

    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def Generate_dataset_low_res_colorizer(BATCH_SIZE=32):
    sketch_list_train, img_list_train, sketch_list_val, img_list_val = Get_filepath_list()

    # make tf.dataset
    X_train_path_ds = tf.data.Dataset.from_tensor_slices(sketch_list_train)
    y_train_path_ds = tf.data.Dataset.from_tensor_slices(img_list_train)
    X_val_path_ds = tf.data.Dataset.from_tensor_slices(sketch_list_val)
    y_val_path_ds = tf.data.Dataset.from_tensor_slices(img_list_val)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    X_train_ds = X_train_path_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256], gray=True), num_parallel_calls=AUTOTUNE)
    y_train_ds = y_train_path_ds.map(lambda x: load_and_preprocess_image(x, shape=[32, 32], RGB=True), num_parallel_calls=AUTOTUNE)
    X_val_ds = X_val_path_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256], gray=True), num_parallel_calls=AUTOTUNE)
    y_val_ds = y_val_path_ds.map(lambda x: load_and_preprocess_image(x, shape=[32, 32], RGB=True), num_parallel_calls=AUTOTUNE)

    train_ds = tf.data.Dataset.zip((X_train_ds, y_train_ds))
    val_ds = tf.data.Dataset.zip((X_val_ds, y_val_ds))

    # Image augmentation
    @tf.function
    def Augmentation(img1, img2):
        if tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)<=0.5:
            img1 = tf.image.flip_left_right(img1)
            img2 = tf.image.flip_left_right(img2)
        return img1, img2

    train_ds = train_ds.map(Augmentation, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(Augmentation, num_parallel_calls=AUTOTUNE)

    print("Generated tf.data with BATCH_SIZE = {} ".format(BATCH_SIZE))
    print("Training data count: {}".format(len(sketch_list_train)))
    print("Validation data count: {}".format(len(sketch_list_val)))
    print("train_ds:", train_ds)
    print("valn_ds:", val_ds)

    # Add dataset settings using tf.data API
    train_ds = train_ds.shuffle(buffer_size=len(sketch_list_train)) # buffer size as larage as the datset ensures that the data is completely shuffled
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) # lets the dataset fetch batches in the background whilte the model is training

    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds