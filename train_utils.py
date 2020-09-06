import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
from tqdm import tqdm
import random

'''
def one_image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def convert_to_tfrecords():
    record_file = 'train1.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        for filename, label in image_labels.items():
            image_string = open(filename, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())'''


def normalize(input_slice):
    input_slice = input_slice / 300.0
    input_slice -= 0.5
    return input_slice


#Temporary loader
def get_train_data(AD, CN):
    ad_image_names = listdir(AD)
    cn_image_names = listdir(CN)

    ad_images = []
    cn_images = []

    print("Phase 1...")

    for i in tqdm(ad_image_names):
        for j in listdir(AD + "/" + i):
            single_slice = []
            for k in listdir(AD + "/" + i + "/" + j):
                try:
                    data = pydicom.dcmread(AD + "/" + i + "/" + j + "/" + k)
                    single_slice += [normalize(data.pixel_array.reshape((128, 128))).tolist()]
                except:
                    pass
            if np.array(single_slice).shape == (188, 128, 128):
                ad_images += [[single_slice]]
    
    print("Phase 2...")

    for i in tqdm(cn_image_names):
        for j in listdir(CN + "/" + i):
            single_slice = []
            for k in listdir(CN + "/" + i + "/" + j):
                try:
                    data = pydicom.dcmread(CN + "/" + i + "/" + j + "/" + k)
                    single_slice += [normalize(data.pixel_array.reshape((128, 128)))]
                except:
                    pass
            if np.array(single_slice).shape == (188, 128, 128):
                cn_images += [[single_slice]]

    print("Phase 3...")

    ad_images = np.array(ad_images).reshape((len(ad_images), 188, 128, 128))
    cn_images = np.array(cn_images).reshape((len(cn_images), 188, 128, 128))

    ad_images = ad_images[:len(cn_images)]

    print(cn_images.shape)

    '''
    train_x = []
    train_y = []
    for i in range(num_of_each):
        train_x += [[ad_images[i], cn_images[i]]]
        train_y += [1]
        train_x += [[cn_images[i+4], cn_images[i]]]
        train_y += [0]
    
    train_x = np.array(train_x).reshape((1, len(train_x), 2, 360, 128, 128))
    train_y = np.array(train_y).reshape((len(train_y), 1))'''
    
    return cn_images, ad_images


#Temporary preparer
def create_trainable_data(train_x_control, train_x_ad):
    train_x = np.array([[train_x_ad[:16], train_x_control[:16]]])
    train_y = np.ones((1, 16))
    add_on = np.array([[train_x_control[16:32], train_x_control[:16]]])
    train_x = np.concatenate((train_x, add_on))
    train_y = np.concatenate((train_y, np.zeros((1, 16))))
    
    train_x = train_x.reshape((2, 32, 188, 128, 128))
    train_y = train_y.reshape((1, 32))

    print("Shuffling...")

    shuffler = list(zip(train_x.tolist(), train_y.tolist()))
    random.shuffle(shuffler)

    train_x[:], train_y[:] = zip(*shuffler)

    train_x = np.array(train_x).reshape((2, 32, 188, 128, 128))
    train_y = np.array(train_y).reshape((1, 32))

    return train_x, train_y


def get_loss_object():
    return tf.keras.losses.BinaryCrossentropy()


def get_optimizer():
    return tf.keras.optimizers.Adam()

#get_train_data("", "", 100)