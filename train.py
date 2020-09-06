from model import create_model
from train_utils import normalize, get_train_data, get_loss_object, get_optimizer

import tensorflow as tf
from tqdm import tqdm

def main():
    print("Creating model...")
    model = create_model([360, 16384, 1], [64, 128, 128], [3, 3, 3], [1, 2, 2], ["same", "same", "same"], [2, 2, 2])
    model.summary()

    print("Gathering data...")

    train_x, train_y = get_train_data("./data/Alzheimer's/AV45(Amyloid)", "./data/Control/AV45(Amyloid)", 4)

    binary_cross_entropy = get_loss_object()
    adam_optimizer = get_optimizer()

main()