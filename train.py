from model import create_model
from train_utils import normalize, get_train_data, get_loss_object, get_optimizer, create_trainable_data

import tensorflow as tf
from tqdm import tqdm

EXAMPLES = 12
SLICES = 160
BATCH_SIZE = 4
EPOCHS = 200

def main():
    print("Creating model...")
    model = create_model([128, 128, SLICES], [64, 128, 256, 512], [3, 3, 3, 3], [1, 1, 1, 1], ["same", "same", "same", "same"], [2, 2, 2, 2])
    model.summary()

    print("Creating training objects...")
    binary_cross_entropy = get_loss_object()
    adam_optimizer = get_optimizer()

    model.compile(optimizer=adam_optimizer, loss=binary_cross_entropy)

    print("Gathering data...")
    train_x_control, train_x_ad = get_train_data("./data/Alzheimer's/AV45(Amyloid)", "./data/Control/AV45(Amyloid)", SLICES, EXAMPLES)
    
    train_x, train_y = create_trainable_data(train_x_control, train_x_ad, SLICES, EXAMPLES)
    print("Training...")

    model.fit([train_x[0].reshape((EXAMPLES, 128, 128, SLICES))[2:EXAMPLES], train_x[1].reshape((EXAMPLES, 128, 128, SLICES))[2:EXAMPLES]], train_y.reshape((EXAMPLES, 1))[2:EXAMPLES], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
    #model.fit([train_x[0].reshape((EXAMPLES, SLICES, 128, 128))[2:32], train_x[1].reshape((EXAMPLES, SLICES, 128, 128))[2:32]], train_y.reshape((EXAMPLES, 1))[2:32], batch_size=BATCH_SIZE, epochs=EPOCHS)

    for i in range(EXAMPLES):
        print(train_y.reshape((EXAMPLES, 1))[i])
        print(model.predict([train_x[0].reshape((EXAMPLES, SLICES, 128, 128))[i].reshape((1, SLICES, 128, 128)), train_x[1].reshape((EXAMPLES, SLICES, 128, 128))[i].reshape((1, SLICES, 128, 128))]))

main()