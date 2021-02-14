from model import create_autoencoder, create_simple_classifier, create_simple_comparison_model
from train_utils import get_optimizer, compute_autoencoder_gradients, compute_simple_classifier_gradients, compute_simple_comparison_gradients, get_one_image
from tfrecords import open_tfrecords

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from vtk.util import numpy_support
import vtk
from PIL import Image
import nibabel as nib

EXAMPLES = 525
STARTING_INDEX = 0
SLICES = 96
BATCH_SIZE = 8
EPOCHS = 100


def save_image(data, filePath):
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(filePath)


def plot3d(data, filePath):
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, filePath)


def train_autoencoder():
    print("Creating model...")
    model = create_autoencoder([128, 128, SLICES], [64, 64, 64], [3, 3, 3], [1, 1, 1], ["same", "same", "same"], [2, 2, 2])
    model.summary()

    print("Creating training objects...")
    optimizer = get_optimizer()

    print("Pulling tfrecords...")
    dataset = open_tfrecords(["./data/tfrecords/Classifier_Data_Train/data1.tfrecord",
        "./data/tfrecords/Autoencoder_Data_Train/data2.tfrecord",
        "./data/tfrecords/Autoencoder_Data_Train/data3.tfrecord",
        "./data/tfrecords/Autoencoder_Data_Train/data4.tfrecord",
        "./data/tfrecords/Autoencoder_Data_Train/data5.tfrecord",
        "./data/tfrecords/Autoencoder_Data_Train/data6.tfrecord",
        "./data/tfrecords/Autoencoder_Data_Train/data7.tfrecord",
        "./data/tfrecords/Autoencoder_Data_Train/data8.tfrecord",
        "./data/tfrecords/Autoencoder_Data_Train/data9.tfrecord"])
    
    dataset_test = open_tfrecords(["./data/tfrecords/Autoencoder_Data_Test/data1.tfrecord",
        "./data/tfrecords/Autoencoder_Data_Test/data2.tfrecord"])

    print("Training...")

    curr_batch_x = None
    curr_batch_y = None

    test_data_x = None
    test_data_y = None

    train_loss_results = []

    for epoch in range(EPOCHS):
        epoch_loss_avg = tf.keras.metrics.Mean()
        batch_sample = 0
        save_image_input = None
        saved = False
        for sample in tqdm(dataset):
            if batch_sample == 0 and saved == True:
                curr_batch_x = sample[1].numpy().reshape((1, 128, 128, SLICES))
                curr_batch_y = sample[1].numpy().reshape((1, 128, 128, SLICES))
                batch_sample += 1
            elif batch_sample % BATCH_SIZE == 0 and saved == True:
                loss_value, grads = compute_autoencoder_gradients(model, curr_batch_x, curr_batch_y)
                epoch_loss_avg.update_state(loss_value)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                print("Train loss: " + str(loss_value))
                batch_sample = 0
            elif saved == True:
                curr_batch_x = np.concatenate((curr_batch_x, sample[1].numpy().reshape((1, 128, 128, SLICES))))
                curr_batch_y = np.concatenate((curr_batch_y, sample[1].numpy().reshape((1, 128, 128, SLICES))))
                batch_sample += 1
            if saved == False:
                save_image_input = sample[1].numpy().reshape((128, 128, SLICES))
                plot3d(sample[1].numpy().reshape((128, 128, SLICES)), "3d/input.nii.gz")
                save_image(sample[1].numpy().reshape((128, 128, SLICES))[:, :, 10], "images/input.jpg")
                saved = True
        
        batch_sample = 0
        for sample in tqdm(dataset_test):
            if batch_sample == 0:
                curr_batch_x = sample[1].numpy().reshape((1, 128, 128, SLICES))
                curr_batch_y = sample[1].numpy().reshape((1, 128, 128, SLICES))
                batch_sample += 1
            else:
                curr_batch_x = np.concatenate((curr_batch_x, sample[1].numpy().reshape((1, 128, 128, SLICES))))
                curr_batch_y = np.concatenate((curr_batch_y, sample[1].numpy().reshape((1, 128, 128, SLICES))))
        test_loss_value, grads = compute_autoencoder_gradients(model, curr_batch_x, curr_batch_y)

        print("Test loss: " + str(test_loss_value))

        print("Epoch: " + str(epoch))
        if epoch % 5 == 0:
            plot3d(np.array(model.predict(save_image_input.reshape((1, 128, 128, SLICES)))).reshape((128, 128, SLICES)), "3d/output" + str(epoch) + ".nii.gz")
            save_image(np.array(model.predict(save_image_input.reshape((1, 128, 128, SLICES)))).reshape((128, 128, SLICES))[:, :, 10], "images/" + str(epoch) + ".jpg")
            
            layer_names = []
            for layer in model.layers:
                layer_names.append(layer.name)
            
            layer_outputs = [layer.output for layer in model.layers] 
            activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
            activations = activation_model.predict(save_image_input.reshape((1, 128, 128, SLICES)))
    
            images_per_row = 8
            
            for layer_name, layer_activation in zip(layer_names, activations):
                n_features = layer_activation.shape[-1]
                size = layer_activation.shape[1]
                n_cols = n_features // images_per_row
                display_grid = np.zeros((size * n_cols, images_per_row * size))
                for col in range(n_cols):
                    for row in range(images_per_row):
                        channel_image = layer_activation[0, :, :, col * images_per_row + row]
                        display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
                plt.figure(figsize=(32, 32))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                plt.savefig("activations/" + layer_name + ".jpg", cmap='viridis')
                plt.close()
            
            model.save('saved_model/my_model')
            print("Saved")

        train_loss_results.append(epoch_loss_avg.result())
    
    fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
    axes.set_ylabel("Loss", fontsize=14)
    axes.plot(train_loss_results)
    axes.set_xlabel("Epoch", fontsize=14)
    plt.savefig("loss.png")


def train_simple_classifier():
    print("Creating model...")
    model = create_simple_classifier()

    model.summary()

    print("Creating training objects...")
    optimizer = get_optimizer()

    print("Pulling tfrecords...")
    dataset = open_tfrecords(["./data/tfrecords/Classifier_Data_Train/data1.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data2.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data3.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data4.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data5.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data6.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data7.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data8.tfrecord"])
    
    dataset_test = open_tfrecords(["./data/tfrecords/Classifier_Data_Test/data1.tfrecord",
        "./data/tfrecords/Classifier_Data_Test/data2.tfrecord"])

    print("Training...")

    curr_batch_x = None
    curr_batch_y = None

    test_data_x = None
    test_data_y = None

    train_loss_results = []

    accuracy = tf.keras.metrics.BinaryAccuracy()
    test_accuracy = tf.keras.metrics.BinaryAccuracy()

    for epoch in range(EPOCHS):
        epoch_loss_avg = tf.keras.metrics.Mean()
        batch_sample = 0
        save_image_input = None
        saved = False
        for sample in tqdm(dataset):
            if batch_sample == 0:
                curr_batch_x = sample[1].numpy().reshape((1, 128, 128, SLICES))
                curr_batch_y = sample[0].numpy().reshape((1, 1))
                batch_sample += 1
            elif batch_sample % BATCH_SIZE == 0:
                loss_value, grads = compute_simple_classifier_gradients(model, curr_batch_x, curr_batch_y)
                epoch_loss_avg.update_state(loss_value)
                accuracy.update_state(curr_batch_y, model.predict(curr_batch_x))
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                print("Accuracy: " + str(accuracy.result().numpy()))
                batch_sample = 0
            else:
                curr_batch_x = np.concatenate((curr_batch_x, sample[1].numpy().reshape((1, 128, 128, SLICES))))
                curr_batch_y = np.concatenate((curr_batch_y, sample[0].numpy().reshape((1, 1))))
                batch_sample += 1
        
        batch_sample = 0
        for sample in tqdm(dataset_test):
            if batch_sample == 0:
                curr_batch_x = sample[1].numpy().reshape((1, 128, 128, SLICES))
                curr_batch_y = sample[0].numpy().reshape((1, 1))
                batch_sample += 1
            else:
                curr_batch_x = np.concatenate((curr_batch_x, sample[1].numpy().reshape((1, 128, 128, SLICES))))
                curr_batch_y = np.concatenate((curr_batch_y, sample[0].numpy().reshape((1, 1))))
        print(model.predict(curr_batch_x[:5]))
        print(curr_batch_y[:5])
        test_loss_value, grads = compute_simple_classifier_gradients(model, curr_batch_x, curr_batch_y)
        test_accuracy.update_state(curr_batch_y, model.predict(curr_batch_x))
        
        print("Train accuracy: " + str(accuracy.result().numpy()))
        print("Test accuracy: " + str(test_accuracy.result().numpy()))

        accuracy.reset_states()
        test_accuracy.reset_states()

        train_loss_results.append(epoch_loss_avg.result())

        print("Epoch: " + str(epoch))
    
    fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
    axes.set_ylabel("Loss", fontsize=14)
    axes.plot(train_loss_results)
    axes.set_xlabel("Epoch", fontsize=14)
    plt.savefig("loss.png")


def train_simple_comparison_model():
    print("Creating model...")
    input_num = 1
    model = create_simple_comparison_model(input_num)
    
    model.summary()

    dataset_reference = open_tfrecords(["./data/tfrecords/Reference/data1.tfrecord"])

    ad_reference = []

    ref_i = 0
    for sample in dataset_reference:
        if ref_i >= input_num:
            ad_reference = sample[1].numpy().reshape((1, 128, 128, SLICES))
            break
        ref_i += 1

    print("Creating training objects...")
    optimizer = get_optimizer()
    
    dataset = open_tfrecords(["./data/tfrecords/Classifier_Data_Train/data1.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data2.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data3.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data4.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data5.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data6.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data7.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data8.tfrecord"])
    
    dataset_test = open_tfrecords(["./data/tfrecords/Classifier_Data_Test/data1.tfrecord",
        "./data/tfrecords/Classifier_Data_Test/data2.tfrecord"])

    print("Training...")

    curr_batch_x = None
    curr_batch_y = None

    test_data_x = None
    test_data_y = None

    train_loss_results = []

    accuracy = tf.keras.metrics.BinaryAccuracy()
    test_accuracy = tf.keras.metrics.BinaryAccuracy()

    for epoch in range(EPOCHS):
        epoch_loss_avg = tf.keras.metrics.Mean()
        batch_sample = 0
        save_image_input = None
        saved = False
        for sample in tqdm(dataset):
            if batch_sample == 0:
                curr_batch_x = [[ad_reference, sample[1].numpy().reshape((1, 128, 128, SLICES))]]
                curr_batch_y = sample[0].numpy().reshape((1, 1))
                batch_sample += 1
            elif batch_sample % BATCH_SIZE == 0:
                test_model = tf.keras.Model(inputs=model.input, outputs=model.layers[7].output)
                print(test_model.predict(curr_batch_x))
                print(curr_batch_y)
                loss_value, grads = compute_simple_comparison_gradients(model, curr_batch_x, curr_batch_y)
                epoch_loss_avg.update_state(loss_value)
                accuracy.update_state(curr_batch_y, model.predict(curr_batch_x))
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                print("Accuracy: " + str(accuracy.result().numpy()))
                batch_sample = 0
            else:
                curr_batch_x += [[ad_reference, sample[1].numpy().reshape((1, 128, 128, SLICES))]]
                curr_batch_y = np.concatenate((curr_batch_y, sample[0].numpy().reshape((1, 1))))
                batch_sample += 1
        
        batch_sample = 0
        for sample in tqdm(dataset_test):
            if batch_sample == 0:
                curr_batch_x = [[ad_reference, sample[1].numpy().reshape((1, 128, 128, SLICES))]]
                curr_batch_y = sample[0].numpy().reshape((1, 1))
                batch_sample += 1
            else:
                curr_batch_x += [[ad_reference, sample[1].numpy().reshape((1, 128, 128, SLICES))]]
                curr_batch_y = np.concatenate((curr_batch_y, sample[0].numpy().reshape((1, 1))))
        test_loss_value, grads = compute_simple_comparison_gradients(model, curr_batch_x, curr_batch_y)
        test_accuracy.update_state(curr_batch_y, model.predict(curr_batch_x))
        
        print("Train accuracy: " + str(accuracy.result().numpy()))
        print("Test accuracy: " + str(test_accuracy.result().numpy()))

        accuracy.reset_states()
        test_accuracy.reset_states()

        train_loss_results.append(epoch_loss_avg.result())

        print("Epoch: " + str(epoch))
    
    fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
    axes.set_ylabel("Loss", fontsize=14)
    axes.plot(train_loss_results)
    axes.set_xlabel("Epoch", fontsize=14)
    plt.savefig("loss.png")


def main():
    train_simple_classifier()
    #train_simple_comparison_model()

    
main()