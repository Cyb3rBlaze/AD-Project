import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
from tqdm import tqdm

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
    input_slice = input_slice / 255.0
    input_slice -= 0.5
    return input_slice


#Temporary loader
def get_train_data(AD, CN, num_of_each):
    ad_image_names = listdir(AD)
    cn_image_names = listdir(CN)

    ad_images = []
    cn_images = []

    print("Phase 1...")

    for i in tqdm(ad_image_names):
        for j in listdir(AD + "/" + i):
            single_slice = []
            for k in listdir(AD + "/" + i + "/" + j):
                data = pydicom.dcmread(AD + "/" + i + "/" + j + "/" + k)
                single_slice += [data.pixel_array.reshape((16384, 1))]
            ad_images += [[single_slice]]
    
    print("Phase 2...")

    for i in tqdm(cn_image_names):
        for j in listdir(CN + "/" + i):
            single_slice = []
            for k in listdir(CN + "/" + i + "/" + j):
                data = pydicom.dcmread(CN + "/" + i + "/" + j + "/" + k)
                single_slice += [data.pixel_array.reshape((16384, 1))]
            cn_images += [[single_slice]]

    print("Phase 3...")

    ad_images = np.array(ad_images)
    cn_images = np.array(cn_images)

    train_x = []
    train_y = []

    for i in range(num_of_each):
        train_x += [[ad_images[i], cn_images[i]]]
        train_y += [1]
        train_x += [[cn_images[i+4], cn_images[i]]]
        train_y += [0]
    
    return train_x, train_y

def get_loss_object():
    return tf.keras.losses.BinaryCrossentropy()


def get_optimizer():
    return tf.keras.optimizers.Adam()

#get_train_data("", "", 100)