import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from keras.utils.generic_utils import get_custom_objects

l2 = tf.keras.regularizers.L2(0.001)
l1 = tf.keras.regularizers.l1(0.1)


def down_sample(input_layer, num_filters, kernel_size, strides, padding, max_pool):
    skip_input = input_layer

    intermediate_layer = tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding=padding, activity_regularizer=l1)(input_layer)
    intermediate_layer = tf.keras.layers.MaxPool2D(max_pool)(intermediate_layer)
    intermediate_layer = tf.keras.layers.BatchNormalization()(intermediate_layer)
    intermediate_layer = tf.keras.layers.LeakyReLU()(intermediate_layer)

    return intermediate_layer


def up_sample(input_layer, num_filters, kernel_size, strides, padding, max_pool):
    skip_input = input_layer

    intermediate_layer = tf.keras.layers.UpSampling2D()(input_layer)
    intermediate_layer = tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding=padding)(intermediate_layer)
    intermediate_layer = tf.keras.layers.BatchNormalization()(intermediate_layer)
    intermediate_layer = tf.keras.layers.LeakyReLU()(intermediate_layer)

    return intermediate_layer


def create_autoencoder(input_dims, num_filters, kernel_size, strides, padding, max_pool):
    input_layer = tf.keras.layers.Input(shape=input_dims)
    intermediate_layer = input_layer

    for i in range(len(num_filters)):
        intermediate_layer = down_sample(intermediate_layer, num_filters[i], kernel_size[i], strides[i], padding[i], max_pool[i])
    
    for i in reversed(range(len(num_filters))):
        intermediate_layer = up_sample(intermediate_layer, num_filters[i], kernel_size[i], strides[i], padding[i], max_pool[i])

    final_layer = tf.keras.layers.Conv2D(input_dims[2], kernel_size[0], strides=1, padding=padding[0], activation="sigmoid")(intermediate_layer)

    return tf.keras.Model(inputs=input_layer, outputs=final_layer)


def create_simple_classifier():
    model = tf.keras.models.load_model('saved_model/my_model')

    input_layer = tf.keras.Input(shape=(128, 128, 96))

    encoder = tf.keras.Model(model.input, model.layers[12].output)
    encoder.trainable = False

    intermediate = encoder(input_layer, training=False)
    
    intermediate = tf.keras.layers.Flatten()(intermediate)
    final_activation = tf.keras.layers.Dense(128, activation="tanh")(intermediate)
    final_activation = tf.keras.layers.Dense(1, activation="sigmoid")(final_activation)

    return tf.keras.Model(inputs=input_layer, outputs=final_activation)


def euclidean_distance(inputs):
	(x1, x2) = inputs
	sumSquared = K.sum(K.square(x1 - x2), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def create_simple_comparison_model(input_num):
    model = tf.keras.models.load_model('saved_model/my_model')

    ad_input = tf.keras.Input(shape=(128, 128, 96))

    unknown_input = tf.keras.Input(shape=(128, 128, 96))

    encoder = tf.keras.Model(model.input, model.layers[12].output)

    intermediate_ad = encoder(ad_input)
    intermediate_ad = tf.keras.layers.Flatten()(intermediate_ad)
    intermediate_ad = tf.keras.layers.Dense(128)(intermediate_ad)

    intermediate_unknown = encoder(unknown_input)
    intermediate_unknown = tf.keras.layers.Flatten()(intermediate_unknown)
    intermediate_unknown = tf.keras.layers.Dense(128)(intermediate_unknown)
    
    ad_unknown = tf.keras.layers.Lambda(euclidean_distance)([intermediate_ad, intermediate_unknown])

    final_activation = tf.keras.layers.Dense(1, activation="sigmoid")(ad_unknown)

    return tf.keras.Model(inputs=[ad_input, unknown_input], outputs=final_activation)