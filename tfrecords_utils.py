import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import listdir
import os as os
import pydicom
import cv2
import sys
import copy


def normalize(input_img):
    input_img = input_img/input_img.max()
    input_img -= input_img.min()
    input_img = input_img/input_img.max()

    return input_img



def crop_center(img, cropx, cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def make_small(img, width, height):
    new_image = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    return new_image


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def one_image_example(image_np, label):
    image_data = tf.io.serialize_tensor(image_np)
    label = int(label)
    print(label)

    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_data),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_tfrecords(filenames, data_x, labels):
    print("Combining Data...")
    
    #train_x = np.concatenate((av45, av1451, fdg))
    train_x = data_x
    #train_y = np.concatenate((label_av45, label_av1451, label_fdg))
    train_y = labels

    '''combined = list(zip(train_x, train_y))

    random.seed(0)
    random.shuffle(combined)
    
    train_x, train_y = zip(*combined)'''

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    print(train_y.shape)

    print("Writing...")

    start = 0
    for file_ in filenames:
        with tf.io.TFRecordWriter(file_) as writer:
            for i in tqdm(range(int(train_y.shape[0]/len(filenames)))):
                tf_example = one_image_example(train_x[i+start], train_y[i+start])
                writer.write(tf_example.SerializeToString())


def get_autoencoder_data(AV45, slices, num_of_examples, starting_index):
    av45_image_sets = listdir(AV45)

    av45_images = []

    #CN:0
    label_av45 = np.zeros(num_of_examples)
    #MCI:1
    label_av45 = np.concatenate((label_av45, np.ones(num_of_examples)))
    #AD:2
    label_av45 = np.concatenate((label_av45, np.full(shape=num_of_examples, fill_value=2, dtype=np.int)))

    print("Phase 1...")

    #collections of collections of collections of images (i:Classifier_Train_AV45_CN)
    for i in av45_image_sets:
        type_change = False
        break_loop = False
        curr_index = 0
        print(i)
        #collections of collections of images (m:002_S_0413)
        for m in tqdm(listdir(AV45 + "/" + i)):
            #collections of images parent folder (p:ADNI_Brain_PET__Raw_AV45)
            for p in listdir(AV45 + "/" + i + "/" + m):
                #collections of images (h:2011-06-20_16_16_17.0)
                for h in listdir(AV45 + "/" + i + "/" + m + "/" + p):
                    #whole image samples (j:I214559)
                    for j in listdir(AV45 + "/" + i + "/" + m + "/" + p + "/" + h):
                        if curr_index >= starting_index and type_change == False:
                            single_image = np.array([])
                            count = 0
                            for k in listdir(AV45 + "/" + i + "/" + m + "/" + p + "/" + h + "/" + j):
                                new_path = AV45 + "/" + i + "/" + m + "/" + p + "/" + h + "/" + j + "/" + str(k)[97:99] + ".dcm"
                                os.rename(AV45 + "/" + i + "/" + m + "/" + p + "/" + h + "/" + j + "/" + str(k), new_path)

                            for k in listdir(AV45 + "/" + i + "/" + m + "/" + p + "/" + h + "/" + j):
                                new_path = AV45 + "/" + i + "/" + m + "/" + p + "/" + h + "/" + j + "/" + str(k)[0:2] + ".dcm"
                                try:
                                    if k[1:2] == "_":
                                        new_path = AV45 + "/" + i + "/" + m + "/" + p + "/" + h + "/" + j + "/" + str(k)[0:1] + ".dcm"
                                except:
                                    pass
                                os.rename(AV45 + "/" + i + "/" + m + "/" + p + "/" + h + "/" + j + "/" + k, new_path)
                            
                            parse_dir_initial_strings = listdir(AV45 + "/" + i + "/" + m + "/" + p + "/" + h + "/" + j)
                            parse_dir_initial_ints = []
                            for e in parse_dir_initial_strings:
                                e = e[0:2]
                                if e[1:2] == ".":
                                    e = e[0:1]
                                parse_dir_initial_ints += [int(e)]
                            
                            #individual dcm files (k:test.dcm)
                            for k in sorted(parse_dir_initial_ints):
                                print(k)
                                try:
                                    data = pydicom.dcmread(AV45 + "/" + i + "/" + m + "/" + p + "/" + h + "/" + j + "/" + str(k) + ".dcm")
                                    single_slice = make_small(data.pixel_array, 128, 128).reshape((128, 128, 1))
                                    if single_image.shape[0] >= 1:
                                        single_image = np.concatenate((single_image, single_slice), axis=2)
                                    else:
                                        single_image = single_slice
                                except:
                                    break
                            try:
                                single_image = normalize(single_image)
                                av45_images += [single_image]
                            except:
                                pass
                        curr_index += 1
                    if np.array(av45_images).shape[0] % num_of_examples == 0 and np.array(av45_images).shape[0] > 0 and curr_index >= starting_index+1:
                        type_change = True
                        break_loop = True
                        break
                if break_loop:
                    break
            if break_loop:
                break
    
    av45_images = np.array(av45_images).reshape((len(av45_images), 128, 128, slices))/1.0
    av45_images = av45_images[:num_of_examples*3]

    print(av45_images.shape)
    print(label_av45.shape)
    
    return av45_images, label_av45


def get_classifier_data(AV45, slices, num_of_examples, starting_index):
    av45_image_sets = listdir(AV45)

    av45_images = []

    #CN:0
    label_av45 = np.zeros(num_of_examples)
    #AD:1
    label_av45 = np.concatenate((label_av45, np.ones(num_of_examples)))

    print("Phase 1...")

    #collections of collections of collections of images (i:Classifier_Train_AV45_CN)
    for i in av45_image_sets:
        type_change = False
        break_loop = False
        curr_index = 0
        print(i)
        if i != "Classifier_Train_AV45_MCI":
            #collections of collections of images (m:002_S_0413)
            for m in tqdm(listdir(AV45 + "/" + i)):
                #collections of images parent folder (p:ADNI_Brain_PET__Raw_AV45)
                for p in listdir(AV45 + "/" + i + "/" + m):
                    #collections of images (h:2011-06-20_16_16_17.0)
                    for h in listdir(AV45 + "/" + i + "/" + m + "/" + p):
                        #whole image samples (j:I214559)
                        for j in listdir(AV45 + "/" + i + "/" + m + "/" + p + "/" + h):
                            if curr_index >= starting_index and type_change == False:
                                single_image = np.array([])
                                count = 0
                                
                                parse_dir_initial_strings = listdir(AV45 + "/" + i + "/" + m + "/" + p + "/" + h + "/" + j)
                                parse_dir_initial_ints = []
                                for e in parse_dir_initial_strings:
                                    e = e[0:2]
                                    if e[1:2] == ".":
                                        e = e[0:1]
                                    parse_dir_initial_ints += [int(e)]
                                
                                #individual dcm files (k:test.dcm)
                                for k in sorted(parse_dir_initial_ints):
                                    try:
                                        data = pydicom.dcmread(AV45 + "/" + i + "/" + m + "/" + p + "/" + h + "/" + j + "/" + str(k) + ".dcm")
                                        single_slice = make_small(data.pixel_array, 128, 128).reshape((128, 128, 1))
                                        if single_image.shape[0] >= 1:
                                            single_image = np.concatenate((single_image, single_slice), axis=2)
                                        else:
                                            single_image = single_slice
                                    except:
                                        break
                                try:
                                    single_image = normalize(single_image)
                                    av45_images += [single_image]
                                except:
                                    pass
                            curr_index += 1
                        if np.array(av45_images).shape[0] % num_of_examples == 0 and np.array(av45_images).shape[0] > 0 and curr_index >= starting_index+1:
                            type_change = True
                            break_loop = True
                            break
                    if break_loop:
                        break
                if break_loop:
                    break
    
    av45_images = np.array(av45_images).reshape((len(av45_images), 128, 128, slices))/1.0
    av45_images = av45_images[:num_of_examples*3]

    print(av45_images.shape)
    print(label_av45.shape)
    
    return av45_images, label_av45


def _parse_image_function(example_proto):
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    
    feature1 = example['label']
    feature2 = tf.io.parse_tensor(example['image_raw'], out_type=tf.float64)
    
    return feature1, feature2


#Not fit for shape (128, 128, slices)
def save_tfrecord_image(np_array):
    plt.imsave("tfrecord.jpg", np_array[100])


def open_tfrecords(filenames):
    raw_image_dataset = tf.data.TFRecordDataset(filenames)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    return parsed_image_dataset

#First example is corrupted

#av45, label_av45 = get_autoencoder_data("./data/AV45/Train", 96, 20, 0)

#generate_autoencoder_tfrecords(['./data/tfrecords/Autoencoder_Data_Train/data1.tfrecord'], av45, label_av45)

#av45, label_av45 = get_classifier_data("./data/AV45/Train", 96, 3, 1)

#generate_tfrecords(['./data/tfrecords/Reference/data1.tfrecord'], av45, label_av45)