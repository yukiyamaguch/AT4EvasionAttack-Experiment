import os
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import numpy as np
from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainer
from art.attacks.evasion import BasicIterativeMethod

from model import victim_model, proxy_model
from prepare_dataset import fashionMNIST

trainX, trainY, testX, testY = fashionMNIST(True, True)
#tf.compat.v1.disable_eager_execution()

def lr_schedule(epoch, lr=0.01):
    return lr * (0.1**int(epoch/10))

def whitebox():
    # settings
    width = 28
    height = 28
    input_shape = (width, height, 1)
    classes = 10
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
    trainX, trainY, testX, testY = fashionMNIST(True, True)

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
    width = 28
    height = 28
    input_shape = (width, height, 1)
    classes = 10
    batch_size = 32
    epochs = 50

    model = proxy_model(input_shape, classes)
    model.summary()

    # callbacks
    weight_path = os.path.join("model/", "proxy0.5")
    model_check_point = ModelCheckpoint(filepath=weight_path, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss')
    learning_rate_scheduler = LearningRateScheduler(lr_schedule)
    callbacks = [model_check_point, early_stopping, learning_rate_scheduler]

    # dataset
    trainX, trainY, testX, testY = fashionMNIST(True, True)
    trainX = trainX[:int(len(trainX)*0.25)]
    trainY = trainY[:int(len(trainY)*0.25)]

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

def adv_train():
    # settings
    width = 28
    height = 28
    input_shape = (width, height, 1)
    classes = 10
    batch_size = 32
    epochs = 50

    model = victim_model(input_shape, classes)
    model.summary()


    classifier = KerasClassifier(model=model, clip_values=(0,1), use_logits=False)

    # callbacks
    weight_path = os.path.join("model/", "adv_train_victim")
    model_check_point = ModelCheckpoint(filepath=weight_path, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss')
    learning_rate_scheduler = LearningRateScheduler(lr_schedule)
    callbacks = [model_check_point, early_stopping, learning_rate_scheduler]

    # dataset
    #trainX, trainY, testX, testY = fashionMNIST(True, True)

    # train
    bim = BasicIterativeMethod(estimator=classifier, eps=0.05)
    adv_trainer = AdversarialTrainer(classifier, attacks=bim, ratio=1.0)
    adv_trainer.fit(trainX, trainY,
        batch_size=batch_size,
        nb_epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        validation_split=0.1)
    '''
    model.fit(trainX, trainY,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        validation_split=0.1)
    '''

    # test
    Y_preds = classifier.predict(testX)
    Y_preds = np.argmax(Y_preds, axis=1)
    testY = np.argmax(testY, axis=1)
    acc = np.mean([1 if Y_preds[i]==testY[i] else 0 for i in range(len(testY))])
    print("Accuracy: "+str(acc))


def main():
    #whitebox()
    proxy()
    #adv_train()


if __name__ == '__main__':
    main()

