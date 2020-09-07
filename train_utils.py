import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
from tqdm import tqdm
import random
import cv2

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


#Temporary loader
def get_train_data(AD, CN, slices, num_of_examples):
    ad_image_sets = listdir(AD)
    cn_image_sets = listdir(CN)

    ad_images = []
    cn_images = []

    print("Phase 1...")
    
    for i in tqdm(ad_image_sets):
        for p in listdir(AD + "/" + i):
            for h in listdir(AD + "/" + i + "/" + p):
                for j in listdir(AD + "/" + i + "/" + p + "/" + h):
                    single_slice = []
                    for k in listdir(AD + "/" + i + "/" + p + "/" + h + "/" + j):
                        try:
                            data = pydicom.dcmread(AD + "/" + i + "/" + p + "/" + h + "/" + j + "/" + k)
                            single_slice += [normalize(make_small(data.pixel_array, 128, 128).reshape((128, 128))).tolist()]
                        except:
                            single_slice = None
                            break
                    if single_slice != None:
                        single_slice = single_slice[(int(len(single_slice)/2)-int(slices/2)):(int(len(single_slice)/2)+int(slices/2))]
                        if np.array(single_slice).shape == (slices, 128, 128):
                            ad_images += [[single_slice]]
        if len(ad_images) >= num_of_examples:
            break
    
    ad_images = np.array(ad_images).reshape((len(ad_images), slices, 128, 128))
    ad_images = ad_images[:num_of_examples]
    
    print("Phase 2...")

    for i in tqdm(cn_image_sets):
        for p in listdir(CN + "/" + i):
            for h in listdir(CN + "/" + i + "/" + p):
                for j in listdir(CN + "/" + i + "/" + p + "/" + h):
                    single_slice = []
                    for k in listdir(CN + "/" + i + "/" + p + "/" + h + "/" + j):
                        try:
                            data = pydicom.dcmread(CN + "/" + i + "/" + p + "/" + h + "/" + j + "/" + k)
                            single_slice += [normalize(make_small(data.pixel_array, 128, 128).reshape((128, 128))).tolist()]
                        except:
                            single_slice = None
                            break
                    if single_slice != None:
                        single_slice = single_slice[(int(len(single_slice)/2)-int(slices/2)):(int(len(single_slice)/2)+int(slices/2))]
                        if np.array(single_slice).shape == (slices, 128, 128):
                            cn_images += [[single_slice]]
        if len(cn_images) >= num_of_examples:
            break

    cn_images = np.array(cn_images).reshape((len(cn_images), slices, 128, 128))
    cn_images = cn_images[:num_of_examples]

    print("Phase 3...")
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

    print(ad_images.shape)
    print(cn_images.shape)
    
    return cn_images, ad_images


#Temporary preparer
def create_trainable_data(train_x_control, train_x_ad, slices, num_of_examples):
    print("Shuffling...")
    
    control_input = train_x_control[0].reshape((1, slices, 128, 128))
    for i in range(int(num_of_examples/2)-1):
        control_input = np.concatenate((control_input, train_x_control[0].reshape((1, slices, 128, 128))))
    train_x = np.array([np.concatenate((train_x_ad[:int(num_of_examples/2)], control_input)), 
        np.concatenate((train_x_control[:int(num_of_examples/2)], control_input))])
    train_y = np.concatenate((np.ones((1, int(num_of_examples/2))), np.zeros((1, int(num_of_examples/2))))).reshape((1, num_of_examples))
    
    '''train_x = np.array([np.concatenate((train_x_ad[:int(num_of_examples/2)], train_x_control[:int(num_of_examples/2)])), np.concatenate((train_x_control[int(num_of_examples/2):num_of_examples], train_x_control[:int(num_of_examples/2)]))])
    train_y = np.concatenate((np.ones((1, int(num_of_examples/2))), np.zeros((1, int(num_of_examples/2))))).reshape((1, num_of_examples))'''

    random.seed(0)
    random.shuffle(train_x[0])
    random.shuffle(train_x[1])
    random.shuffle(train_y[0])

    return train_x, train_y


def get_loss_object():
    return tf.keras.losses.BinaryCrossentropy()


def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.0001)

#get_train_data("", "", 100)