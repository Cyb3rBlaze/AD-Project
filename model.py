import tensorflow as tf
import numpy as np

def model_block(input_layer, num_filters, kernel_size, strides, padding, max_pool):
    conv1 = tf.keras.layers.Conv2D(num_filters, kernel_size, strides=strides, padding=padding)(input_layer)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=max_pool)(conv1)

    return pool1

def create_model(input_dims, num_filters, kernel_size, strides, padding, max_pool):
    patient_input_layer = tf.keras.layers.Input(shape=input_dims)
    control_input_layer = tf.keras.layers.Input(shape=input_dims)

    patient_intermediate_layer = patient_input_layer
    control_intermediate_layer = control_input_layer

    for i in range(len(num_filters)):
        patient_intermediate_layer = model_block(patient_intermediate_layer, num_filters[i], kernel_size[i], strides[i], padding[i], max_pool[i])

    for i in range(len(num_filters)):
        control_intermediate_layer = model_block(control_intermediate_layer, num_filters[i], kernel_size[i], strides[i], padding[i], max_pool[i])

    combined = tf.keras.layers.concatenate([patient_intermediate_layer, control_intermediate_layer])

    flatten = tf.keras.layers.Flatten()(combined)
    final_layer = tf.keras.layers.Dense(1, activation="sigmoid")(flatten)

    return tf.keras.Model(inputs=[patient_input_layer, control_input_layer], outputs=final_layer)

'''model = create_model([256, 256, 3], [16, 32, 64], [3, 3, 3], [1, 2, 2], ["same", "same", "same"], [2, 2, 2])
model.summary()
print(model.predict([np.random.rand(1, 256, 256, 3), np.random.rand(1, 256, 256, 3)]))'''