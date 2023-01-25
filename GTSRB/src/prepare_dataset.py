import os
import csv
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


def preprocess_img(image, target_size):
    # Histgram Normalization in v channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:, :, 2])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # resize image
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

def read_img(path):
    # Reads the image at path, checking if it was really loaded
    img = cv2.imread(path)
    assert img is not None, "No image found at {0:s}".format(path)
    return img

def write_img(path, image):
    # Wrapper to allow easy replacement of image write function
    cv2.imwrite(path, image)


def _read_labels(path, labels_filename="labels.csv"):
    # Reads the labels.csv produced in the format of process_orig_data
    # :return: a list of (filename, label) tuples
    filelist = []
    with open(os.path.join(path, labels_filename)) as f:
        f.readline()
        for line in f:
            fname, label = line.split(",")
            label = int(label)
            filelist.append((fname, label))
    
    return filelist

def read_imgs_from_dir(path, labels_filename="labels.csv"):
    filelist = _read_labels(path, labels_filename)
    X,Y=[],[]
    for img_name, label in filelist:
        img = read_img(os.path.join(path, img_name))
        X.append(img)
        Y.append(label)
    return np.array(X), np.array(Y)


def _process_labels_file(path, target_size, labels_filename="labels.csv"):
    filelist = _read_labels(path, labels_filename)
    X,Y=[],[]
    for img_name, label in filelist:
        img = read_img(os.path.join(path, img_name))
        img = preprocess_img(img, target_size)
        X.append(img)
        Y.append(label)
    return np.array(X), np.array(Y)

def gtsrb(train_path, test_path, target_size, labels_filename="labels.csv"):
    '''
    Loads the *processed* data from the GTSRB dataset
    for training and evaluation.
    Assumes each folder specified in train_path, test_path
    contains a labels.csv in the formate of process_orig_data
    Also subtracts the average pixel value of the test dataset.
    If either of train_path or test_path is unspecified, returns
    an empty array in the corresponding positions
    :return: X_train, Y_train, X_test, Y_test
    '''

    if train_path is not None:
        X_train, Y_train = _process_labels_file(train_path, target_size, labels_filename)
        Y_train = to_categorical(Y_train, 43)
    else:
        print("training data is not found.")
        X_train = []
        Y_train = []

    if test_path is not None:
        X_test, Y_test = _process_labels_file(test_path, target_size, labels_filename)
        Y_test = to_categorical(Y_test, 43)
    else:
        print("test data is not found.")
        X_test = []
        Y_test = []

    return X_train, Y_train, X_test, Y_test 


