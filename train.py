import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "data.json"
MODEL_SAVE_PATH = "model.h5"


def get_data_splits(data_path):
    # load dataset

    # create train/validate/test data splits

    
    pass

def main():

    # load train/test/validation splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # build CNN model

    # train model

    # evaluate model

    # save model
