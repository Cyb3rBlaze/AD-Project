from model import create_autoencoder, create_simple_classifier, create_simple_comparison_model, create_av45_fdg_comparison_model
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
BATCH_SIZE = 2
EPOCHS = 30


def save_image(data, filePath):
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(filePath)


def plot3d(data, filePath):
    final_arr = np.array([])
    count = 0
    for i in data:
        if count == 0:
            final_arr = i.reshape((128, 128, 1))
        else:
            final_arr = np.concatenate((final_arr, i.reshape(128, 128, 1)), axis=2)
        count += 1

    print(final_arr.shape)

    img = nib.Nifti1Image(final_arr, np.eye(4))
    nib.save(img, filePath)


def train_autoencoder():
    print("Creating model...")
    model = create_simple_classifier([SLICES, 128, 128, 1], [16, 32], [3, 3], [1, 1], ["same", "same"], [2, 2])
    model.summary()

    print("Creating training objects...")
    optimizer = get_optimizer()

    print("Pulling tfrecords...")
    dataset = open_tfrecords(["./data/tfrecords/Classifier_Data_Train/data1.tfrecord", 
    "./data/tfrecords/Classifier_Data_Train/data2.tfrecord"])
    
    dataset_test = open_tfrecords(["./data/tfrecords/Classifier_Data_Test/data1.tfrecord"])

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
                curr_batch_x = sample[1].numpy().reshape((1, SLICES, 128, 128, 1))
                curr_batch_y = sample[1].numpy().reshape((1, SLICES, 128, 128, 1))
                batch_sample += 1
            elif batch_sample % BATCH_SIZE == 0 and saved == True:
                loss_value, grads = compute_simple_classifier_gradients(model, curr_batch_x, curr_batch_y)
                epoch_loss_avg.update_state(loss_value)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                print("Train loss: " + str(loss_value))
                batch_sample = 0
            elif saved == True:
                curr_batch_x = np.concatenate((curr_batch_x, sample[1].numpy().reshape((1, SLICES, 128, 128, 1))))
                curr_batch_y = np.concatenate((curr_batch_y, sample[1].numpy().reshape((1, SLICES, 128, 128, 1))))
                batch_sample += 1
            if saved == False:
                save_image_input = sample[1].numpy().reshape((SLICES, 128, 128, 1))
                plot3d(sample[1].numpy().reshape((SLICES, 128, 128)), "3d/input.nii.gz")
                saved = True
        
        batch_sample = 0
        for sample in tqdm(dataset_test):
            if batch_sample == 0:
                curr_batch_x = sample[1].numpy().reshape((1, SLICES, 128, 128, 1))
                curr_batch_y = sample[1].numpy().reshape((1, SLICES, 128, 128, 1))
                batch_sample += 1
            else:
                curr_batch_x = np.concatenate((curr_batch_x, sample[1].numpy().reshape((1, SLICES, 128, 128, 1))))
                curr_batch_y = np.concatenate((curr_batch_y, sample[1].numpy().reshape((1, SLICES, 128, 128, 1))))
        test_loss_value, grads = compute_autoencoder_gradients(model, curr_batch_x, curr_batch_y)

        print("Test loss: " + str(test_loss_value))

        print("Epoch: " + str(epoch))
        if epoch % 5 == 0:
            plot3d(np.array(model.predict(save_image_input.reshape((1, SLICES, 128, 128, 1)))).reshape((SLICES, 128, 128)), "3d/output" + str(epoch) + ".nii.gz")
            
            '''layer_names = []
            for layer in model.layers:
                layer_names.append(layer.name)
            
            layer_outputs = [layer.output for layer in model.layers] 
            activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
            activations = activation_model.predict(save_image_input.reshape((1, SLICES, 128, 128, 1)))
    
            images_per_row = 8
            
            for layer_name, layer_activation in zip(layer_names, activations):
                n_features = layer_activation.shape[-1]
                size = layer_activation.shape[]
                n_cols = n_features // images_per_row
                display_grid = np.zeros((size * n_cols, images_per_row * size))
                for col in range(n_cols):
                    for row in range(images_per_row):
                        channel_image = layer_activation[0, col * images_per_row + row, :, :]
                        display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
                plt.figure(figsize=(32, 32))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                plt.savefig("activations/" + layer_name + ".jpg", cmap='viridis')
                plt.close()
            
            model.save('saved_model/my_model')'''
            print("Saved")

        train_loss_results.append(epoch_loss_avg.result())
    
    fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
    axes.set_ylabel("Loss", fontsize=14)
    axes.plot(train_loss_results)
    axes.set_xlabel("Epoch", fontsize=14)
    plt.savefig("loss.png")


def train_simple_classifier():
    print("Creating model...")
    model = create_simple_classifier([SLICES, 128, 128, 1], [8, 16], [3, 3], [1, 1], ["same", "same"], [2, 2])

    model.summary()

    print("Creating training objects...")
    optimizer = get_optimizer()

    print("Pulling tfrecords...")
    dataset = open_tfrecords(["./data/tfrecords/Classifier_Data_Train/data1.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data2.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data3.tfrecord",
        "./data/tfrecords/Classifier_Data_Train/data3.tfrecord"])
    
    dataset_test = open_tfrecords(["./data/tfrecords/Classifier_Data_Test/data1.tfrecord"])

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
                curr_batch_x = sample[1].numpy().reshape((1, SLICES, 128, 128, 1))
                curr_batch_y = sample[0].numpy().reshape((1, 1))
                batch_sample += 1
            elif batch_sample % BATCH_SIZE == 0:
                print(curr_batch_y)
                print(model.predict(curr_batch_x)[1])
                loss_value, grads = compute_simple_classifier_gradients(model, curr_batch_x, curr_batch_y, 0.2, 0.4)
                epoch_loss_avg.update_state(loss_value)
                accuracy.update_state(curr_batch_y, model.predict(curr_batch_x)[1])
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                print("Accuracy: " + str(accuracy.result().numpy()))
                batch_sample = 0
            else:
                curr_batch_x = np.concatenate((curr_batch_x, sample[1].numpy().reshape((1, SLICES, 128, 128, 1))))
                curr_batch_y = np.concatenate((curr_batch_y, sample[0].numpy().reshape((1, 1))))
                batch_sample += 1
            if saved == False:
                save_image_input = sample[1].numpy().reshape((SLICES, 128, 128, 1))
                plot3d(sample[1].numpy().reshape((SLICES, 128, 128)), "3d/input.nii.gz")
                saved = True
        
        batch_sample = 0
        for sample in dataset_test:
            if batch_sample == 0:
                curr_batch_x = sample[1].numpy().reshape((1, SLICES, 128, 128, 1))
                curr_batch_y = sample[0].numpy().reshape((1, 1))
                batch_sample += 1
            else:
                curr_batch_x = np.concatenate((curr_batch_x, sample[1].numpy().reshape((1, SLICES, 128, 128, 1))))
                curr_batch_y = np.concatenate((curr_batch_y, sample[0].numpy().reshape((1, 1))))
                batch_sample += 1
            if batch_sample == 10:
                print(curr_batch_y)
                print(model.predict(curr_batch_x)[1])
                test_loss_value, grads = compute_simple_classifier_gradients(model, curr_batch_x, curr_batch_y, 0.2, 0.4)
                test_accuracy.update_state(curr_batch_y, model.predict(curr_batch_x)[1])
                batch_sample = 0
        
        if epoch % 5 == 0:
            plot3d(np.array(model.predict(save_image_input.reshape((1, SLICES, 128, 128, 1)))[0]).reshape((SLICES, 128, 128)), "3d/output" + str(epoch) + ".nii.gz")

            model.save('saved_model/my_model')
            print("Saved")
        
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
    model = create_av45_fdg_comparison_model([SLICES, 128, 128, 1], [4, 8], [3, 3], ["same", "same"], [2, 2])
    
    model.summary()

    dataset_reference = open_tfrecords(["./data/tfrecords/Reference/data1.tfrecord"])

    ad_reference_av45 = []
    ad_reference_fdg = []
    cn_reference_av45 = []
    cn_reference_fdg = []

    ref_i = 0
    for sample in dataset_reference:
        if ref_i == 0:
            cn_reference_av45 = sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[0].reshape((1, SLICES, 128, 128, 1))
            cn_reference_fdg = sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[1].reshape((1, SLICES, 128, 128, 1))
        elif ref_i == 1:
            ad_reference_av45 = sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[0].reshape((1, SLICES, 128, 128, 1))
            ad_reference_fdg = sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[1].reshape((1, SLICES, 128, 128, 1))
        ref_i += 1

    print("Creating training objects...")
    optimizer = get_optimizer()
    
    #dataset = open_tfrecords(["./data/tfrecords/Comparison_Data_Train/data1.tfrecord"])

    dataset = open_tfrecords(["./data/tfrecords/Comparison_Data_Train/data1.tfrecord",
        "./data/tfrecords/Comparison_Data_Train/data2.tfrecord",
        "./data/tfrecords/Comparison_Data_Train/data3.tfrecord"])
    
    dataset_test = open_tfrecords(["./data/tfrecords/Comparison_Data_Test/data1.tfrecord"])

    print("Training...")

    curr_batch_x = None
    curr_batch_y = None

    test_data_x = None
    test_data_y = None

    train_loss_results = []
    test_loss_results = []

    train_accuracy_results = []
    test_accuracy_results = []

    accuracy = tf.keras.metrics.BinaryAccuracy()
    test_accuracy = tf.keras.metrics.BinaryAccuracy()

    epoch_loss_avg = tf.keras.metrics.Mean()
    test_loss_avg = tf.keras.metrics.Mean()

    for epoch in range(EPOCHS):
        batch_sample = 0
        save_image_input = None
        saved = False
        ad_data_av45 = np.array([])
        ad_data_fdg = np.array([])
        cn_data_av45 = np.array([])
        cn_data_fdg = np.array([])
        for sample in tqdm(dataset):
            if batch_sample == 0:
                curr_batch_x_av45 = sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[0].reshape((1, SLICES, 128, 128, 1))
                curr_batch_x_fdg = sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[1].reshape((1, SLICES, 128, 128, 1))
                curr_batch_y = sample[0].numpy().reshape((1, 1))
                batch_sample += 1
            elif batch_sample % BATCH_SIZE == 0:
                print(curr_batch_y)
                print(model.predict((curr_batch_x_av45, curr_batch_x_fdg))[2])
                loss_value, grads = compute_simple_comparison_gradients(model, [curr_batch_x_av45, curr_batch_x_fdg], curr_batch_y, 0.3, 0.7)
                epoch_loss_avg.update_state(loss_value)
                accuracy.update_state(curr_batch_y, model.predict((curr_batch_x_av45, curr_batch_x_fdg))[2])
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                print("Accuracy: " + str(accuracy.result().numpy()))
                batch_sample = 0
            else:
                curr_batch_x_av45 = np.concatenate((curr_batch_x_av45, sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[0].reshape((1, SLICES, 128, 128, 1))))
                curr_batch_x_fdg = np.concatenate((curr_batch_x_fdg, sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[1].reshape((1, SLICES, 128, 128, 1))))
                curr_batch_y = np.concatenate((curr_batch_y, sample[0].numpy().reshape((1, 1))))
                batch_sample += 1
            if saved == False:
                save_image_input = sample[1].numpy().reshape((2, SLICES, 128, 128, 1))
                plot3d(curr_batch_x_av45[0], "3d/input.nii.gz")
                saved = True
        
        batch_sample = 0
        ad_data_av45 = np.array([])
        ad_data_fdg = np.array([])
        cn_data_av45 = np.array([])
        cn_data_fdg = np.array([])
        correct_having = 0
        correct_not_having = 0
        for sample in tqdm(dataset_test):
            curr_output = model.predict((sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[0].reshape((1, SLICES, 128, 128, 1)), sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[1].reshape((1, SLICES, 128, 128, 1))))[2][0][0]
            #Calculating sensitivity
            if curr_output > 0.5 and sample[0].numpy().reshape((1, 1))[0] == 1:
                correct_having += 1
            #Calculating specificity
            elif curr_output < 0.5 and sample[0].numpy().reshape((1, 1))[0] == 0:
                correct_not_having += 1
            if batch_sample == 0:
                curr_batch_x_av45 = sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[0].reshape((1, SLICES, 128, 128, 1))
                curr_batch_x_fdg = sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[1].reshape((1, SLICES, 128, 128, 1))
                curr_batch_y = sample[0].numpy().reshape((1, 1))
                batch_sample += 1
            else:
                curr_batch_x_av45 = np.concatenate((curr_batch_x_av45, sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[0].reshape((1, SLICES, 128, 128, 1))))
                curr_batch_x_fdg = np.concatenate((curr_batch_x_fdg, sample[1].numpy().reshape((2, SLICES, 128, 128, 1))[1].reshape((1, SLICES, 128, 128, 1))))
                curr_batch_y = np.concatenate((curr_batch_y, sample[0].numpy().reshape((1, 1))))
                batch_sample += 1
            if batch_sample == BATCH_SIZE:
                print(curr_batch_y)
                print(model.predict((curr_batch_x_av45, curr_batch_x_fdg))[2])
                test_loss_value, grads = compute_simple_comparison_gradients(model, (curr_batch_x_av45, curr_batch_x_fdg), curr_batch_y, 0.3, 0.7)
                test_loss_avg.update_state(test_loss_value)
                test_accuracy.update_state(curr_batch_y, model.predict((curr_batch_x_av45, curr_batch_x_fdg))[2])
                batch_sample = 0
        
        if epoch % 2 == 0:
            plot3d(np.array(model.predict((save_image_input.reshape((2, SLICES, 128, 128, 1))[0].reshape((1, SLICES, 128, 128, 1)), \
                save_image_input.reshape((2, SLICES, 128, 128, 1))[1].reshape((1, SLICES, 128, 128, 1))))[0]).reshape((SLICES, 128, 128)), "3d/output" + str(epoch) + ".nii.gz")

            model.save('saved_model/my_model')
            print("Saved")

        print("Train accuracy: " + str(accuracy.result().numpy()))
        print("Test accuracy: " + str(test_accuracy.result().numpy()))
        print("Sensitivity: " + str(correct_having/11))
        print("Specificity: " + str(correct_not_having/11))

        train_loss_results.append(epoch_loss_avg.result())
        test_loss_results.append(test_loss_avg.result())

        train_accuracy_results.append(accuracy.result())
        test_accuracy_results.append(test_accuracy.result())

        accuracy.reset_states()
        test_accuracy.reset_states()

        epoch_loss_avg.reset_states()
        test_loss_avg.reset_states()

        print("Epoch: " + str(epoch))
    
    fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
    axes.set_ylabel("Loss", fontsize=14)
    axes.plot(train_loss_results)
    axes.plot(test_loss_results)
    axes.set_xlabel("Epoch", fontsize=14)
    plt.savefig("loss.png")

    fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
    axes.set_ylabel("Accuracy", fontsize=14)
    axes.plot(train_accuracy_results)
    axes.plot(test_accuracy_results)
    axes.set_xlabel("Epoch", fontsize=14)
    plt.savefig("accuracy.png")


def main():
    #train_simple_classifier()
    train_simple_comparison_model()
    #train_autoencoder()

    
main()