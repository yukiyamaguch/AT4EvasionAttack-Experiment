import os
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import numpy as np

from model import victim_model, proxy_model
from prepare_dataset import gtsrb

def lr_schedule(epoch, lr=0.01):
    return lr * (0.1**int(epoch/10))

def whitebox():
    # settings
    width = 48
    height = 48
    input_shape = (width, height, 3)
    classes = 43
    batch_size = 32
    epochs = 50

    # load model
    model = victim_model(input_shape, classes)
    model.summary()

    # callbacks
    weight_path = os.path.join("model/", "whitebox")
    model_check_point = ModelCheckpoint(filepath=weight_path, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss')
    learning_rate_scheduler = LearningRateScheduler(lr_schedule)
    #callbacks = [model_check_point, learning_rate_scheduler]
    callbacks = [model_check_point, early_stopping, learning_rate_scheduler]
    #callbacks = [model_check_point]

    # dataset
    trainX, trainY, testX, testY = gtsrb("dataset/training", "dataset/test", (48,48))

    # train
    model.fit(trainX, trainY,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        validation_split=0.1)

    # test
    Y_preds = model.predict(testX)
    Y_preds = np.argmax(Y_preds, axis=1)
    testY = np.argmax(testY, axis=1)
    acc = np.mean([1 if Y_preds[i]==testY[i] else 0 for i in range(len(testY))])
    print("Accuracy: "+str(acc))

def proxy():
    # settings
    width = 48
    height = 48
    input_shape = (width, height, 3)
    classes = 43
    batch_size = 32
    epochs = 50

    model = proxy_model(input_shape, classes)
    model.summary()

    # callbacks
    weight_path = os.path.join("model/", "proxy")
    model_check_point = ModelCheckpoint(filepath=weight_path, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss')
    learning_rate_scheduler = LearningRateScheduler(lr_schedule)
    callbacks = [model_check_point, early_stopping, learning_rate_scheduler]

    # dataset
    trainX, trainY, testX, testY = gtsrb("dataset/training", "dataset/test", (48,48))
    trainX = trainX[:int(len(trainX)*0.7)]
    trainY = trainY[:int(len(trainY)*0.7)]

    # train
    model.fit(trainX, trainY,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        validation_split=0.1)

    # test
    Y_preds = model.predict(testX)
    Y_preds = np.argmax(Y_preds, axis=1)
    testY = np.argmax(testY, axis=1)
    acc = np.mean([1 if Y_preds[i]==testY[i] else 0 for i in range(len(testY))])
    print("Accuracy: "+str(acc))

def distil():
    # settings
    width = 48
    height = 48
    input_shape = (width, height, 3)
    classes = 43
    batch_size = 32
    epochs = 50

    victim = tf.keras.models.load_model('model/checkpoint')
    model = proxy_model(input_shape, classes)
    model.summary()

    # callbacks
    weight_path = os.path.join("model/", "distil")
    model_check_point = ModelCheckpoint(filepath=weight_path, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss')
    learning_rate_scheduler = LearningRateScheduler(lr_schedule)
    callbacks = [model_check_point, early_stopping, learning_rate_scheduler]

    # dataset
    trainX, trainY, testX, testY = gtsrb("dataset/training", "dataset/test", (48,48))
    trainX = trainX[:int(len(trainX)*0.7)]
    trainY = trainY[:int(len(trainY)*0.7)]
    print(len(trainX))
    exit()
    predsY = victim.predict(trainX)

    # train
    model.fit(trainX, predsY,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        validation_split=0.1)

    # test
    Y_preds = model.predict(testX)
    Y_preds = np.argmax(Y_preds, axis=1)
    testY = np.argmax(testY, axis=1)
    acc = np.mean([1 if Y_preds[i]==testY[i] else 0 for i in range(len(testY))])
    print("Accuracy: "+str(acc))

def main():
    #whitebox()
    #proxy()
    distil()

if __name__ == '__main__':
    main()

