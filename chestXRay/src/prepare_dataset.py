import os
import csv
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


def preprocess_img(image):
    # resize image
    height, width, _ = image.shape
    left  = int(0.1 * width)
    right = int(0.9 * width)
    up    = int(0.15 * height)
    down  = int(1.0 * height)
    image = image[up:down, left:right, :]
    image = cv2.resize(image, (256,256))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = np.reshape(image, (256, 256, 1))
    image = image / 255.0
    return image

def read_img(path):
    # Reads the image at path, checking if it was really loaded
    img = cv2.imread(path)
    assert img is not None, "No image found at {0:s}".format(path)
    return img

def write_img(path, image):
    # Wrapper to allow easy replacement of image write function
    heigt, width, _ = image.shape
    #image = np.reshape(image, (heigt, width))
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


def _process_labels_file(path, target_size, labels_filename="labels.csv"):
    filelist = _read_labels(path, labels_filename)
    X,Y=[],[]
    for img_name, label in filelist:
        img = read_img(os.path.join(path, img_name))
        img = preprocess_img(img, target_size)
        X.append(img)
        Y.append(label)
    return np.array(X), np.array(Y)


def read_images_from_dir(path):
    # Normal
    npath = os.path.join(path, 'NORMAL')
    image_names = os.listdir(npath)
    image_paths = [os.path.join(npath, img_name) for img_name in image_names]
    Nimgs, Nlabels = [], []
    for i in range(len(image_paths)):
        image = read_img(image_paths[i])
        image = preprocess_img(image)
        Nimgs.append(image)
        Nlabels.append([1,0])
    # PNEUMONIA
    ppath = os.path.join(path, 'PNEUMONIA')
    image_names = os.listdir(ppath)
    image_paths = [os.path.join(ppath, img_name) for img_name in image_names]
    Pimgs, Plabels = [], []
    for i in range(len(image_paths)):
        image = read_img(image_paths[i])
        image = preprocess_img(image)
        Pimgs.append(image)
        Plabels.append([0,1])

    imgs = Nimgs + Pimgs
    labels = Nlabels + Plabels
    # shuffle dataset
    ds = [(i, l) for i, l in zip(imgs, labels)]
    np.random.shuffle(ds)
    imgs = np.array([img for img, _ in ds])
    labels = np.array([label for _, label in ds])
    return imgs, labels

def chestXray(train_path, test_path):
    if train_path is not None:
        X_train, Y_train = read_images_from_dir(train_path)
    else:
        print("training data is not found.")
        X_train,Y_train = [],[]

    if test_path is not None:
        X_test, Y_test = read_images_from_dir(test_path)
    else:
        print("test data is not found.")
        X_test,Y_test = [],[]
    return X_train, Y_train, X_test, Y_test

def chestXray_victim():
    ppath = 'dataset/victim/PNEUMONIA'
    image_names = os.listdir(ppath)
    image_paths = [os.path.join(ppath, img_name) for img_name in image_names]
    Pimgs, Plabels = [], []
    for i in range(len(image_paths)):
        image = read_img(image_paths[i])
        image = preprocess_img(image)
        Pimgs.append(image)
        Plabels.append([0,1])
    return np.array(Pimgs), np.array(Plabels)
