import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_data),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_classifier_tfrecords(filenames, train_x_control, train_x_ad, examples):
    train_x = np.array(np.concatenate((train_x_control, train_x_ad)))
    train_y = np.array(np.concatenate((np.zeros((train_x_control.shape[0])), np.ones((train_x_ad.shape[0])))))
    train_y = train_y.reshape(((train_x_control.shape[0]+train_x_ad.shape[0]), 1))

    print("Augmenting...")

    train_x = np.concatenate((train_x, np.flip(train_x, axis=1)))
    train_y = np.concatenate((train_y, train_y))

    combined = list(zip(train_x, train_y))

    random.seed(0)
    random.shuffle(combined)
    
    train_x, train_y = zip(*combined)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    print("Writing...")

    start = 0
    for file_ in filenames:
        with tf.io.TFRecordWriter(file_) as writer:
            for i in tqdm(range(int(train_y.shape[0]/len(filenames)))):
                tf_example = one_image_example(train_x[i+start], train_y[i+start])
                writer.write(tf_example.SerializeToString())
        start += int(examples/len(filenames))


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


def open_tfrecords(filenames, examples, slices):
    raw_image_dataset = tf.data.TFRecordDataset(filenames)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    
    '''
    train_x = []
    train_y = []
    for image_features in tqdm(parsed_image_dataset):
        train_x += [image_features[1].numpy()]
        train_y += [image_features[0].numpy()]

    train_x = np.array(train_x).reshape((examples*4, 128, 128, slices))
    train_y = np.array(train_y).reshape((examples*4, 1))

    print(train_x.shape)
    print(train_y.shape)'''

    return parsed_image_dataset
        

#open_tfrecords()