from model import create_classifier
from train_utils import normalize, get_train_data, loss, get_optimizer, create_trainable_data, create_classifier_data, compute_gradients
from tfrecords_utils import generate_classifier_tfrecords, open_tfrecords

import tensorflow as tf
from tqdm import tqdm
import numpy as np

EXAMPLES = 150
SLICES = 100
BATCH_SIZE = 8
EPOCHS = 200

def main():
    print("Creating model...")
    model = create_classifier([128, 128, SLICES], [128, 128, 256], [3, 3, 3], [1, 1, 1], ["same", "same", "same"], [2, 2, 2])
    model.summary()

    print("Creating training objects...")
    optimizer = get_optimizer()

    print("Gathering data...")
    #train_x_control, train_x_ad = get_train_data("./data/Alzheimer's/AV45(Amyloid)", "./data/Control/AV45(Amyloid)", SLICES, EXAMPLES)
    
    print("Generating tfrecords...")
    #generate_classifier_tfrecords(['./data/tfrecords/data1.tfrecord', './data/tfrecords/data2.tfrecord', './data/tfrecords/data3.tfrecord'], train_x_control, train_x_ad, EXAMPLES)

    print("Pulling tfrecords...")
    dataset = open_tfrecords(["./data/tfrecords/data1.tfrecord", "./data/tfrecords/data2.tfrecord", "./data/tfrecords/data3.tfrecord"], EXAMPLES, SLICES)

    print("Training...")

    for epoch in range(EPOCHS):
        epoch_loss = None
        curr_batch_x = None
        curr_batch_y = None
        batch_sample = 0
        for sample in tqdm(dataset):
            if batch_sample == 0:
                curr_batch_x = sample[1].numpy().reshape((1, 128, 128, SLICES))
                curr_batch_y = sample[0].numpy().reshape((1, 1))
                batch_sample += 1
            elif batch_sample % BATCH_SIZE == 0:
                loss_value, grads = compute_gradients(model, curr_batch_x, curr_batch_y)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss = loss_value
                batch_sample = 0
            else:
                curr_batch_x = np.concatenate((curr_batch_x, sample[1].numpy().reshape((1, 128, 128, SLICES))))
                curr_batch_y = np.concatenate((curr_batch_y, sample[0].numpy().reshape((1, 1))))
                batch_sample += 1

        print(epoch_loss)

main()