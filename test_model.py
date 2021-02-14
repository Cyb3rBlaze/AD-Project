import tensorflow as tf

model = tf.keras.models.load_model('saved_model/my_model')

model.summary()