import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from keras.utils.generic_utils import get_custom_objects

l2 = tf.keras.regularizers.L2(0.2)


def down_sample(input_layer, num_filters, kernel_size, padding, max_pool):
    intermediate_layer = tf.keras.layers.Conv3D(num_filters, kernel_size=kernel_size, padding=padding, activity_regularizer=l2)(input_layer)
    intermediate_layer = tf.keras.layers.Conv3D(num_filters, kernel_size=kernel_size, padding=padding, activity_regularizer=l2)(intermediate_layer)
    intermediate_layer = tf.keras.layers.MaxPool3D(max_pool)(intermediate_layer)
    intermediate_layer = tf.keras.layers.BatchNormalization()(intermediate_layer)
    intermediate_layer = tf.keras.layers.LeakyReLU()(intermediate_layer)

    return intermediate_layer


def up_sample(input_layer, num_filters, kernel_size, padding, max_pool):
    intermediate_layer = tf.keras.layers.UpSampling3D()(input_layer)
    intermediate_layer = tf.keras.layers.Conv3D(num_filters, kernel_size=kernel_size, padding=padding)(intermediate_layer)
    intermediate_layer = tf.keras.layers.Conv3D(num_filters, kernel_size=kernel_size, padding=padding)(intermediate_layer)
    intermediate_layer = tf.keras.layers.BatchNormalization()(intermediate_layer)
    intermediate_layer = tf.keras.layers.LeakyReLU()(intermediate_layer)

    return intermediate_layer


def create_autoencoder(input_dims, num_filters, kernel_size, padding, max_pool):
    input_layer = tf.keras.layers.Input(shape=input_dims)

    #encoder temp
    intermediate_layer = tf.keras.layers.Conv3D(4, kernel_size=(3, 3, 3), padding="same", activation="tanh")(input_layer)
    intermediate_layer = tf.keras.layers.MaxPool3D((2, 2, 2))(intermediate_layer)
    intermediate_layer = tf.keras.layers.Conv3D(8, kernel_size=(3, 3, 3), padding="same", activation="tanh")(intermediate_layer)
    intermediate_layer = tf.keras.layers.MaxPool3D((2, 2, 2))(intermediate_layer)

    #decoder temp
    intermediate_layer = tf.keras.layers.Conv3DTranspose(8, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same", activation="tanh")(intermediate_layer)
    intermediate_layer = tf.keras.layers.Conv3DTranspose(4, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same", activation="tanh")(intermediate_layer)

    final_layer = tf.keras.layers.Conv3D(1, (3, 3, 3), padding="same", activation="sigmoid")(intermediate_layer)

    return tf.keras.Model(inputs=input_layer, outputs=final_layer)


def create_encoder(input_layer, num_filters, kernel_size, padding, max_pool):
    intermediate_layer = input_layer

    for i in range(len(num_filters)):
        intermediate_layer = down_sample(intermediate_layer, num_filters[i], kernel_size[i], padding[i], max_pool[i])
    
    final_layer = intermediate_layer

    return final_layer


def create_decoder(inputs, num_filters, kernel_size, padding, max_pool):
    intermediate_layer = inputs

    for i in reversed(range(len(num_filters))):
        intermediate_layer = up_sample(intermediate_layer, num_filters[i], kernel_size[i], padding[i], max_pool[i])

    final_layer = tf.keras.layers.Conv3D(1, kernel_size[0], strides=1, padding=padding[0], activation="sigmoid")(intermediate_layer)

    return final_layer


def create_simple_classifier(input_dims, num_filters, kernel_size, strides, padding, max_pool):
    input_layer = tf.keras.layers.Input(shape=input_dims)

    encoder = create_encoder(input_layer, num_filters, kernel_size, strides, padding, max_pool)
    decoder = create_decoder(encoder, num_filters, kernel_size, strides, padding, max_pool)
    
    intermediate = tf.keras.layers.Flatten()(encoder)
    final_activation = tf.keras.layers.Dense(1, activation="sigmoid")(intermediate)

    return tf.keras.Model(inputs=input_layer, outputs=[decoder, final_activation])


def euclidean_distance(inputs):
	(x1, x2) = inputs
	sumSquared = K.sum(K.square(x1 - x2), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def subtract(inputs):
	(x1, x2) = inputs
	return x1-x2


def create_simple_comparison_model(input_dims, num_filters, kernel_size, strides, padding, max_pool):
    ad_input = tf.keras.layers.Input(shape=(96, 128, 128, 1))
    unknown_input = tf.keras.Input(shape=(96, 128, 128, 1))
    cn_input = tf.keras.Input(shape=(96, 128, 128, 1))

    #AD
    encoder_ad = create_encoder(ad_input, num_filters, kernel_size, strides, padding, max_pool)
    decoder_ad = create_decoder(encoder_ad, num_filters, kernel_size, strides, padding, max_pool)

    #Unknown
    encoder_unknown = create_encoder(unknown_input, num_filters, kernel_size, strides, padding, max_pool)
    decoder_unknown = create_decoder(encoder_unknown, num_filters, kernel_size, strides, padding, max_pool)

    #CN
    encoder_cn = create_encoder(cn_input, num_filters, kernel_size, strides, padding, max_pool)
    decoder_cn = create_decoder(encoder_cn, num_filters, kernel_size, strides, padding, max_pool)
    
    #Euclidian distance
    ad_unknown = tf.keras.layers.Lambda(euclidean_distance)([encoder_ad, encoder_unknown])
    cn_unknown = tf.keras.layers.Lambda(euclidean_distance)([encoder_cn, encoder_unknown])
    
    final_difference = tf.keras.layers.Lambda(subtract)([ad_unknown, cn_unknown])

    flatten_difference = tf.keras.layers.Flatten()(final_difference)

    final_activation = tf.keras.layers.Dense(1, activation="sigmoid")(flatten_difference)

    return tf.keras.Model(inputs=[ad_input, unknown_input, cn_input], outputs=[decoder_ad, decoder_unknown, decoder_cn, final_activation])


def create_av45_fdg_comparison_model(input_dims, num_filters, kernel_size, padding, max_pool):
    ad_input_av45 = tf.keras.layers.Input(shape=(96, 128, 128, 1))
    unknown_input_av45 = tf.keras.Input(shape=(96, 128, 128, 1))
    cn_input_av45 = tf.keras.Input(shape=(96, 128, 128, 1))

    ad_input_fdg = tf.keras.layers.Input(shape=(96, 128, 128, 1))
    unknown_input_fdg = tf.keras.Input(shape=(96, 128, 128, 1))
    cn_input_fdg = tf.keras.Input(shape=(96, 128, 128, 1))

    #AD
    encoder_ad_av45 = create_encoder(ad_input_av45, num_filters, kernel_size, padding, max_pool)
    decoder_ad_av45 = create_decoder(encoder_ad_av45, num_filters, kernel_size, padding, max_pool)

    encoder_ad_fdg = create_encoder(ad_input_fdg, num_filters, kernel_size, padding, max_pool)
    decoder_ad_fdg = create_decoder(encoder_ad_fdg, num_filters, kernel_size, padding, max_pool)

    #Unknown
    encoder_unknown_av45 = create_encoder(unknown_input_av45, num_filters, kernel_size, padding, max_pool)
    decoder_unknown_av45 = create_decoder(encoder_unknown_av45, num_filters, kernel_size, padding, max_pool)

    encoder_unknown_fdg = create_encoder(unknown_input_fdg, num_filters, kernel_size, padding, max_pool)
    decoder_unknown_fdg = create_decoder(encoder_unknown_fdg, num_filters, kernel_size, padding, max_pool)

    #CN
    encoder_cn_av45 = create_encoder(cn_input_av45, num_filters, kernel_size, padding, max_pool)
    decoder_cn_av45 = create_decoder(encoder_cn_av45, num_filters, kernel_size, padding, max_pool)

    encoder_cn_fdg = create_encoder(cn_input_fdg, num_filters, kernel_size, padding, max_pool)
    decoder_cn_fdg = create_decoder(encoder_cn_fdg, num_filters, kernel_size, padding, max_pool)
    
    #Euclidian distance
    ad_unknown_av45 = tf.keras.layers.Lambda(euclidean_distance)([encoder_ad_av45, encoder_unknown_av45])
    cn_unknown_av45 = tf.keras.layers.Lambda(euclidean_distance)([encoder_cn_av45, encoder_unknown_av45])

    ad_unknown_fdg = tf.keras.layers.Lambda(euclidean_distance)([encoder_ad_fdg, encoder_unknown_fdg])
    cn_unknown_fdg = tf.keras.layers.Lambda(euclidean_distance)([encoder_cn_fdg, encoder_unknown_fdg])
    
    final_difference_av45 = tf.keras.layers.Lambda(subtract)([ad_unknown_av45, cn_unknown_av45])

    flatten_av45 = tf.keras.layers.Flatten(activity_regularizer=l2)(final_difference_av45)

    final_difference_fdg = tf.keras.layers.Lambda(subtract)([ad_unknown_fdg, cn_unknown_fdg])

    flatten_fdg = tf.keras.layers.Flatten(activity_regularizer=l2)(final_difference_fdg)

    concacted = tf.keras.layers.concatenate([flatten_av45, flatten_fdg])

    final_activation = tf.keras.layers.Dense(1, activation="sigmoid")(concacted)

    return tf.keras.Model(inputs=[ad_input_av45, ad_input_fdg, unknown_input_av45, unknown_input_fdg, cn_input_av45, cn_input_fdg], outputs=[decoder_ad_av45, decoder_ad_fdg, decoder_unknown_av45, decoder_unknown_fdg, decoder_cn_av45, decoder_cn_fdg, final_activation])