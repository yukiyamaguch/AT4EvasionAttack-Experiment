import os
import csv
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds


def preprocess_img(image):
    # resize image
    image = np.reshape(image, (28,28,1))
    image = image / 255.0
    return image

def read_img(path):
    # Reads the image at path, checking if it was really loaded
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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


def _process_labels_file(path, labels_filename="labels.csv"):
    filelist = _read_labels(path, labels_filename)
    X,Y=[],[]
    for img_name, label in filelist:
        img = read_img(os.path.join(path, img_name))
        img = preprocess_img(img)
        X.append(img)
        Y.append(label)
    return np.array(X), np.array(Y)



def fashionMNIST(train, test):
    if train is not None:
        train = tfds.load('fashion_mnist', split='train', shuffle_files=True)
        assert isinstance(train, tf.data.Dataset)
        X_train, Y_train = [sample['image'] for sample in train], [sample['label'] for sample in train]
        X_train = np.array(X_train) / 255.0
        Y_train = to_categorical(Y_train, 10)
    else:
        print('training data is not found.')
        X_train = []
        Y_train = []

    if test is not None:
        test = tfds.load('fashion_mnist', split='test', shuffle_files=True)
        assert isinstance(test, tf.data.Dataset)
        X_test, Y_test = [sample['image'] for sample in test], [sample['label'] for sample in test]
        X_test = np.array(X_test) / 255.0
        Y_test = to_categorical(Y_test, 10)
    else:
        print('training data is not found.')
        X_test = []
        Y_test = []
    return X_train, Y_train, X_test, Y_test

def prepare_victim_dataset():
    _,_,testX, testY = fashionMNIST(None, 1)

    # class id 7: Sneaker > class id 9, Ankle boot
    victimID = [i for i in range(len(testX)) if np.all(testY[i] == to_categorical([7], 10)[0])]
    victimX = testX[victimID]
    for i in range(50):
        path = f'dataset/victim/{i+1:02}.png'
        img = victimX[i] * 255.0
        write_img(path, img)

def victim_dataset(victim_path, labels_filename='labels.csv'):
    if victim_path is not None:
        X_victim, Y_victim = _process_labels_file(victim_path, labels_filename)
        Y_victim = to_categorical(Y_victim, 10)
    else:
        print('victim data is not found.')
        X_victim = []
        Y_victim = []
    return X_victim, Y_victim

