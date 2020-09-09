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
                            add = normalize(make_small(data.pixel_array, 128, 128).reshape((128, 128))).tolist()
                            single_slice += [add]
                        except:
                            single_slice = None
                            break
                    if single_slice != None:
                        single_slice = np.array(single_slice).reshape((128, 128, int(np.array(single_slice).size/128/128)))[:,:,(int(len(single_slice)/2)-int(slices/2)):(int(len(single_slice)/2)+int(slices/2))]
                        if np.array(single_slice).shape == (128, 128, slices):
                            ad_images += [[single_slice]]
        if len(ad_images) >= num_of_examples:
            break
    
    ad_images = np.array(ad_images).reshape((len(ad_images), 128, 128, slices))
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
                            add = normalize(make_small(data.pixel_array, 128, 128).reshape((128, 128))).tolist()
                            single_slice += [add]
                        except:
                            single_slice = None
                            break
                    if single_slice != None:
                        single_slice = np.array(single_slice).reshape((128, 128, int(np.array(single_slice).size/128/128)))[:,:,(int(len(single_slice)/2)-int(slices/2)):(int(len(single_slice)/2)+int(slices/2))]
                        if np.array(single_slice).shape == (128, 128, slices):
                            cn_images += [[single_slice]]
        if len(cn_images) >= num_of_examples:
            break

    cn_images = np.array(cn_images).reshape((len(cn_images), 128, 128, slices))
    cn_images = cn_images[:num_of_examples]

    print(ad_images.shape)
    print(cn_images.shape)
    
    return cn_images, ad_images


#Temporary preparer
def create_trainable_data(train_x_control, train_x_ad, slices, num_of_examples):
    print("Shuffling...")
    
    control_input = train_x_control[0].reshape((1, 128, 128, slices))
    for i in range(int(num_of_examples/2)-1):
        control_input = np.concatenate((control_input, train_x_control[0].reshape((1, 128, 128, slices))))
    train_x = np.array([np.concatenate((train_x_ad[:int(num_of_examples/2)], control_input)), 
        np.concatenate((train_x_control[:int(num_of_examples/2)], control_input))])
    train_y = np.concatenate((np.ones((1, int(num_of_examples/2))), np.zeros((1, int(num_of_examples/2))))).reshape((1, num_of_examples))

    random.seed(0)
    random.shuffle(train_x[0])
    random.shuffle(train_x[1])
    random.shuffle(train_y[0])

    return train_x, train_y


def create_classifier_data(train_x_control, train_x_ad, num_of_examples):
    train_x = np.array(np.concatenate((train_x_control, train_x_ad)))
    train_y = np.array(np.concatenate((np.zeros((num_of_examples)), np.ones((num_of_examples)))))
    train_y = train_y.reshape((num_of_examples*2, 1))

    combined = list(zip(train_x, train_y))

    random.seed(0)
    random.shuffle(combined)
    
    train_x, train_y = zip(*combined)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    return train_x, train_y


def get_loss_object():
    return tf.keras.losses.BinaryCrossentropy()


def loss(model, x, y, training):
    y_ = model(x, training=training)
    return get_loss_object()(y_true=y, y_pred=y_)


def compute_gradients(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.0001)