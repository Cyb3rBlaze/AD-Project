import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

def model_block(input_layer, num_filters, kernel_size, strides, padding, max_pool):
    conv1 = tf.keras.layers.Conv2D(num_filters, kernel_size, strides=strides, padding=padding)(input_layer)
    conv2 = tf.keras.layers.Conv2D(num_filters, kernel_size, strides=strides, padding=padding)(conv1)
    if max_pool != False:
        pool1 = tf.keras.layers.MaxPool2D(pool_size=max_pool)(conv2)
        return pool1
    else:
        return conv2

def create_model(input_dims, num_filters, kernel_size, strides, padding, max_pool):
    patient_input_layer = tf.keras.layers.Input(shape=input_dims)
    control_input_layer = tf.keras.layers.Input(shape=input_dims)

    patient_intermediate_layer = patient_input_layer
    control_intermediate_layer = control_input_layer

    for i in range(len(num_filters)):
        patient_intermediate_layer = model_block(patient_intermediate_layer, num_filters[i], kernel_size[i], strides[i], padding[i], max_pool[i])

    for i in range(len(num_filters)):
        control_intermediate_layer = model_block(control_intermediate_layer, num_filters[i], kernel_size[i], strides[i], padding[i], max_pool[i])

    patient_flatten = tf.keras.layers.Flatten()(patient_intermediate_layer)
    control_flatten = tf.keras.layers.Flatten()(control_intermediate_layer)

    distance_euclid = tf.keras.layers.Lambda(lambda tensors : K.abs(tensors[0] - tensors[1]))([patient_flatten , control_flatten])
    
    final_layer = tf.keras.layers.Dense(1, activation="sigmoid")(distance_euclid)

    return tf.keras.Model(inputs=[patient_input_layer, control_input_layer], outputs=final_layer)

'''model = create_model([256, 256, 3], [16, 32, 64], [3, 3, 3], [1, 2, 2], ["same", "same", "same"], [2, 2, 2])
model.summary()
print(model.predict([np.random.rand(1, 256, 256, 3), np.random.rand(1, 256, 256, 3)]))'''