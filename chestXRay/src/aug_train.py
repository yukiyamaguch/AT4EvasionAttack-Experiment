import os
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import numpy as np
import imgaug as aug
import imgaug.augmenters as iaa

from model import victim_model, proxy_model
from prepare_dataset import chestXray

aug.seed(11)

def lr_schedule(epoch, lr=0.0001):
    return lr * (0.1**int(epoch/10))

def data_gen(trainX, trainY, batch_size):
    seq = iaa.OneOf([iaa.Fliplr(), iaa.Affine(rotate=10)])
    n=len(trainX)
    steps = n//batch_size

    batch_data = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
    batch_label = np.zeros((batch_size, 2), dtype=np.float32) 
    indices = np.arange(n)

    i=0
    while True:
        np.random.shuffle(indices)
        count=0
        next_batch = indices[i*batch_size: (i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            x = trainX[idx]
            y = trainY[idx]
            batch_data[count]=x
            batch_label[count]=y
            if np.argmax(y)==0 and count+3 < batch_size-1:
                aug_img1 = seq.augment_image(x)
                aug_img2 = seq.augment_image(x)
                batch_data[count+1] = aug_img1
                batch_label[count+1] = y
                batch_data[count+2] = aug_img2
                batch_label[count+2] = y
                count+=3
            else:
                count+=1
            if count == batch_size:
                break
        i+=1
        yield batch_data, batch_label
        if i >= steps:
            i=0

def whitebox():
    # settings
    width = 256
    height = 256
    #input_shape = (width, height)
    input_shape = (width, height, 3)
    classes = 2
    batch_size = 16
    epochs = 50

    # load model
    model = victim_model(input_shape, classes)
    model.summary()

    # callbacks
    weight_path = os.path.join("model/", "whitebox")
    model_check_point = ModelCheckpoint(filepath=weight_path, save_best_only=True)
    early_stopping = EarlyStopping(patience=5)
    #learning_rate_scheduler = LearningRateScheduler(lr_schedule)
    #callbacks = [model_check_point, learning_rate_scheduler]
    #callbacks = [model_check_point, early_stopping, learning_rate_scheduler]
    callbacks = [model_check_point, early_stopping]
    #callbacks = [model_check_point]

    # dataset
    trainX, trainY, testX, testY = chestXray('dataset/train', 'dataset/test')
    _, _, valX, valY = chestXray(None, 'dataset/val')

    train_datagen = data_gen(trainX, trainY, batch_size=batch_size)
    nb_train_steps = len(trainX) // batch_size

    # train
    model.fit_generator(train_datagen,
        epochs=epochs,
        steps_per_epoch=nb_train_steps,
        callbacks=callbacks,
        validation_data=(valX, valY),
        class_weight={0:1.0, 1:0.4})

    # test
    Y_preds = model.predict(testX)
    Y_preds = np.argmax(Y_preds, axis=1)
    testY = np.argmax(testY, axis=1)
    acc = np.mean([1 if Y_preds[i]==testY[i] else 0 for i in range(len(testY))])
    print("Accuracy: "+str(acc))

def proxy():
    # settings
    width = 256
    height = 256
    #input_shape = (width, height)
    input_shape = (width, height, 3)
    classes = 2
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
    # train dataset size / 3
    trainX, trainY, testX, testY = chestXray('dataset/train', 'dataset/test')
    _, _, valX, valY = chestXray(None, 'dataset/val')

    # train
    model.fit(trainX, trainY,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        validation_data=(valX, valY))

    # test
    Y_preds = model.predict(testX)
    Y_preds = np.argmax(Y_preds, axis=1)
    testY = np.argmax(testY, axis=1)
    acc = np.mean([1 if Y_preds[i]==testY[i] else 0 for i in range(len(testY))])
    print("Accuracy: "+str(acc))


def main():
    whitebox()
    #proxy()

if __name__ == '__main__':
    main()

