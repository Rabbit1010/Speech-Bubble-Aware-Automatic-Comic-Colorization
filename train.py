# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:29:45 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
import os

from data_generator import Generate_dataset_low_res_colorizer, Generate_dataset_polishing_network, Generate_dataset_colorizer
from model import Low_res_colorizer, Polishing_network, Colorizer, Polishing_network_small, Low_res_colorizer_new

def Get_data_count():
    # Count how many training data
    training_data_count = 0
    for filename in os.listdir("./data/comic_img/train/"):
        if filename.lower().endswith('.png'):
            training_data_count += 1
    val_data_count = 0
    for filename in os.listdir("./data/comic_img/validation/"):
        if filename.lower().endswith('.png'):
            val_data_count += 1

    return training_data_count, val_data_count

def Train_colorizer(BATCH_SIZE=32, EPOCH=100):
    training_data_count, val_data_count = Get_data_count()
    train_ds, val_ds = Generate_dataset_colorizer(BATCH_SIZE=BATCH_SIZE)

    # Compile the model
    model = Colorizer()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mse'])

    # Train model on dataset
    checkpoint_path = "./checkpoints/colorizer_checkpoint_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.h5"
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True,
                                                         period=1)

    model.fit(x=train_ds, validation_data=val_ds,
              epochs=EPOCH,  verbose=1,
              steps_per_epoch=tf.math.ceil(training_data_count/BATCH_SIZE).numpy(),
              validation_steps=tf.math.ceil(val_data_count/BATCH_SIZE).numpy(),
              callbacks=[save_checkpoint])

    return model

def Train_low_res_colorizer_new(BATCH_SIZE=16, EPOCH=100):
    training_data_count, val_data_count = Get_data_count()
    train_ds, val_ds = Generate_dataset_low_res_colorizer(BATCH_SIZE=BATCH_SIZE)

    # Compile the model
    model = Low_res_colorizer_new()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mse'])

    # Train model on dataset
    checkpoint_path = "./checkpoints/low_res_colorizer_new_checkpoint_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.h5"
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True,
                                                         period=1)

    model.fit(x=train_ds, validation_data=val_ds,
              epochs=EPOCH, verbose=1,
              steps_per_epoch=tf.math.ceil(training_data_count/BATCH_SIZE).numpy(),
              validation_steps=tf.math.ceil(val_data_count/BATCH_SIZE).numpy(),
              callbacks=[save_checkpoint])

    return model

def Train_low_res_colorizer(BATCH_SIZE=32, EPOCH=100):
    training_data_count, val_data_count = Get_data_count()
    train_ds, val_ds = Generate_dataset_low_res_colorizer(BATCH_SIZE=BATCH_SIZE)

    # Compile the model
    model = Low_res_colorizer()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mse'])

    # Train model on dataset
    checkpoint_path = "./checkpoints/low_res_colorizer_checkpoint_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.h5"
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True,
                                                         period=1)

    model.fit(x=train_ds, validation_data=val_ds,
              epochs=EPOCH, verbose=1,
              steps_per_epoch=tf.math.ceil(training_data_count/BATCH_SIZE).numpy(),
              validation_steps=tf.math.ceil(val_data_count/BATCH_SIZE).numpy(),
              callbacks=[save_checkpoint])

    return model

def Train_polishing_network(BATCH_SIZE=32, EPOCH=100):
    training_data_count, val_data_count = Get_data_count()
    train_ds, val_ds = Generate_dataset_polishing_network(BATCH_SIZE=BATCH_SIZE)

    # Compile the model
    model = Polishing_network()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mse'])

    # Train model on dataset
    checkpoint_path = "./checkpoints/polishing_network_checkpoint_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.h5"
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True,
                                                         period=1)

    model.fit(x=train_ds, validation_data=val_ds,
              epochs=EPOCH, verbose=1,
              steps_per_epoch=tf.math.ceil(training_data_count/BATCH_SIZE).numpy(),
              validation_steps=tf.math.ceil(val_data_count/BATCH_SIZE).numpy(),
              callbacks=[save_checkpoint])

    return model

def Train_polishing_network_small(BATCH_SIZE=32, EPOCH=100):
    training_data_count, val_data_count = Get_data_count()
    train_ds, val_ds = Generate_dataset_polishing_network(BATCH_SIZE=BATCH_SIZE)

    # Compile the model
    model = Polishing_network_small()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mse'])

    # Train model on dataset
    checkpoint_path = "./checkpoints/polishing_network_small_checkpoint_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.h5"
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True,
                                                         period=1)

    model.fit(x=train_ds, validation_data=val_ds,
              epochs=EPOCH, verbose=1,
              steps_per_epoch=tf.math.ceil(training_data_count/BATCH_SIZE).numpy(),
              validation_steps=tf.math.ceil(val_data_count/BATCH_SIZE).numpy(),
              callbacks=[save_checkpoint])

    return model

if __name__ == '__main__':
    Train_low_res_colorizer()

    Train_polishing_network_small()

    Train_colorizer()