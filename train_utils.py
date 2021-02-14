import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
from tqdm import tqdm
import random
import cv2


def normalize(input_slice):
    if input_slice.max() != 0:
        input_slice = input_slice*(255.0/input_slice.max())
        input_slice = input_slice/255.0
        input_slice -= 0.5
    return input_slice


def crop_center(img, cropx, cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def make_small(img, width, height):
    new_image = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    return new_image

#One image
def get_one_image(direc, slices):
    single_slice = []
    for image in listdir(direc):
        try:
            data = pydicom.dcmread(direc + "/" + image)
            add = normalize(make_small(data.pixel_array, 128, 128).reshape((128, 128))).tolist()
            single_slice += [add]
        except:
            single_slice = None
            break
    if single_slice != None:
        single_slice = np.array(single_slice).reshape((128, 128, int(np.array(single_slice).size/128/128)))[:,:,(int(len(single_slice)/2)-int(slices/2)):(int(len(single_slice)/2)+int(slices/2))]
        if np.array(single_slice).shape == (128, 128, slices):
            return single_slice
    
    return single_slice


def get_autoencoder_loss_object():
    return tf.keras.losses.MeanSquaredError()


def get_simple_classifier_loss_object():
    return tf.keras.losses.BinaryCrossentropy()


def get_simple_comparison_loss_object():
    return tf.keras.losses.BinaryCrossentropy()


def autoencoder_loss(model, x, y, training):
    y_ = model(x, training=training)
    return get_autoencoder_loss_object()(y_true=y, y_pred=y_)


def simple_classifier_loss(model, x, y, training):
    y_ = model(x, training=training)
    return get_simple_classifier_loss_object()(y_true=y, y_pred=y_)


def simple_comparison_loss(model, x, y, training):
    y_ = model(x, training=training)
    return get_simple_comparison_loss_object()(y_true=y, y_pred=y_)


def compute_autoencoder_gradients(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = autoencoder_loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def compute_simple_classifier_gradients(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = simple_classifier_loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def compute_simple_comparison_gradients(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = simple_comparison_loss(model, inputs, targets, training=True)
        loss_value += sum(model.losses)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.0001)