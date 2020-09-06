from model import create_model
from train_utils import normalize, get_train_data, get_loss_object, get_optimizer, create_trainable_data

import tensorflow as tf
from tqdm import tqdm

def main():
    print("Creating model...")
    model = create_model([360, 128, 128], [64, 128, 128], [3, 3, 3], [1, 2, 2], ["same", "same", "same"], [2, 2, 2])
    model.summary()

    print("Creating training objects...")
    binary_cross_entropy = get_loss_object()
    adam_optimizer = get_optimizer()

    model.compile(optimizer=adam_optimizer, loss=binary_cross_entropy)

    print("Gathering data...")
    train_x_control, train_x_ad = get_train_data("./data/Alzheimer's/AV45(Amyloid)", "./data/Control/AV45(Amyloid)")

    #model.fit([train_x[0][0][0].reshape((1, 360, 128, 128)), train_x[0][0][1].reshape((1, 360, 128, 128))], train_y[0].reshape((1, 1)), batch_size=4, epochs=2)
    
    train_x, train_y = create_trainable_data(train_x_control, train_x_ad)
    model.fit([train_x[0].reshape((8, 360, 128, 128))[2:7], train_x[1].reshape((8, 360, 128, 128))[2:7]], train_y.reshape((8, 1))[2:7], batch_size=4, epochs=6)
    
    print(model.predict([train_x[0].reshape((8, 360, 128, 128))[0].reshape((1, 360, 128, 128)), train_x[1].reshape((8, 360, 128, 128))[0].reshape((1, 360, 128, 128))]))

main()