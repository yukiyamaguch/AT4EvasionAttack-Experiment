import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, Activation, Lambda, LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, GroupNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import binary_accuracy

from prepare_dataset import gtsrb, write_img
from model import victim_model


def generator_loss(y_pred, y_true):
    return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred-y_true), axis=-1)) -0.3, 0), axis=-1)

def custom_acc(y_pred, y_true):
    return binary_accuracy(K.round(y_true), K.round(y_pred))

def build_discriminator(inputs):
    D = Conv2D(32, 4, strides=(2,2))(inputs)
    D = LeakyReLU()(D)
    D = Dropout(0.4)(D)
    D = Conv2D(64, 4, strides=(2,2))(D)
    D = BatchNormalization()(D)
    D = LeakyReLU()(D)
    D = Dropout(0.4)(D)
    D = Flatten()(D)
    D = Dense(128)(D)
    D = BatchNormalization()(D)
    D = LeakyReLU()(D)
    D = Dense(1, activation='sigmoid')(D)
    return D

def build_generator(inputs):
    G = Conv2D(8, 3, padding='same')(inputs)
    G = GroupNormalization(8)(G)
    G = Activation(tf.nn.leaky_relu)(G)
    G = Conv2D(16, 3, strides=(2,2), padding='same')(G)
    G = GroupNormalization(16)(G)
    G = Activation(tf.nn.leaky_relu)(G)
    G = Conv2D(32, 3, strides=(2,2), padding='same')(G)
    G = GroupNormalization(32)(G)
    G = Activation(tf.nn.leaky_relu)(G)
    
    residual = G
    for _ in range(4):
        G = Conv2D(32, 3, padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation(tf.nn.leaky_relu)(G)
        G = Conv2D(32, 3, padding='same')(G)
        G = BatchNormalization()(G)
        G = layers.add([G, residual])
        residual = G

    G = Conv2DTranspose(16, 3, strides=(2,2), padding='same')(G)
    G = GroupNormalization(16)(G)
    G = Activation(tf.nn.leaky_relu)(G)
    G = Conv2DTranspose(8, 3, strides=(2,2), padding='same')(G)
    G = GroupNormalization(8)(G)
    G = Activation(tf.nn.leaky_relu)(G)
    G = Conv2D(1, 3, padding='same')(G)
    G = GroupNormalization(1)(G)
    G = layers.add([G, inputs])
    return G

# victim model
# input -> G(input) -> D(G(input))
# input -> G(input) -> victim(G(input))
def victim_model():
    pass


def get_batches(generator, start, end, x_train, y_train):
    x_batch = x_train[start:end]
    Gx_batch = generator.predict_on_batch(x_batch)
    y_batch = y_train[start:end]
    return x_batch, Gx_batch, y_batch

def train_D_on_batch(discriminator, batches):
    x_batch, Gx_batch, _ = batches
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(x_batch, np.random.uniform(0.9, 1, size=(len(x_batch), 1)) )
    d_loss_fake = discriminator.train_on_batch(Gx_batch, np.zeros((len(Gx_batch), 1)) )
    d_loss = 0.5*np.add(d_loss_real, d_loss_fake)
    return d_loss

def test_D_on_batch(discriminator, x, Gx):
    d_loss_real = discriminator.test_on_batch(x, np.ones((len(x),1)))
    d_loss_fake = discriminator.test_on_batch(Gx, np.zeros((len(Gx),1)))
    d_loss = 0.5*np.add(d_loss_real, d_loss_fake)
    return d_loss

def train_stacked_on_batch(victim, discriminator, stacked, batches, target_class):
    x_batch, _, y_batch = batches

    discriminator.trainable = False
    victim.trainable = False
    stacked_loss = stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), to_categorical(target_class, 43)])
    return stacked_loss

def test_stacked_on_batch(victim, discriminator, stacked, target_class, x):
    return stacked.test_on_batch(x, [x, np.ones((len(x), 1)), to_categorical(target_class, 43)])

def train_AdvGAN():
    # settings
    width = 48
    height = 48
    input_shape = (width, height, 3)
    target_class = 3    # 60km/h Limit
    epochs = 1000
    save_path = 'model/generator'

    # load dataset
    trainX, trainY, testX, testY = gtsrb('dataset/training', 'dataset/victim', (width,height))

    batch_size = 32
    num_batches = len(trainX) // batch_size
    if len(trainX) % batch_size != 0:
        num_batches += 1

    # load model
    victim_model = tf.keras.models.load_model('model/checkpoint')
    victim_model.trainable = False

    # make generator
    inputs = Input(shape=input_shape)
    outputs = build_generator(inputs)
    generator = Model(inputs, outputs)

    # make discriminator
    outputs = build_discriminator(generator(inputs))
    discriminator = Model(inputs, outputs)
    discriminator.compile(loss=keras.losses.binary_crossentropy, optimizer=SGD(0.01), metrics=[custom_acc])

    # make stacked model
    stacked = Model(inputs = inputs, outputs = [generator(inputs), discriminator(generator(inputs)), victim_model(generator(inputs))])
    stacked.compile(loss=[generator_loss, keras.losses.binary_crossentropy, keras.losses.categorical_crossentropy], optimizer=Adam(0.0002))

    # train AdvGAN
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')
        batch_index = 0
        for batch in range(num_batches - 1):
            start = batch_size * batch_index
            end = batch_size * (batch_index+1)
            batches = get_batches(generator, start, end, trainX, trainY)
            train_D_on_batch(discriminator, batches)
            train_stacked_on_batch(victim_model, discriminator, stacked, batches, batch_size*[target_class])
            batch_index += 1
        start = batch_size * batch_index
        end = len(trainX)
        x_batch, Gx_batch, y_batch = get_batches(generator, start, end, trainX, trainY)

        train_D_on_batch(discriminator, (x_batch, Gx_batch, y_batch))
        train_stacked_on_batch(victim_model, discriminator, stacked, (x_batch, Gx_batch, y_batch), len(y_batch)*[target_class])


        # calc test loss
        Gx = generator.predict_on_batch(testX)
        (d_loss, d_acc) = test_D_on_batch(discriminator, testX, Gx)
        (g_loss, hinge_loss, gan_loss, adv_loss) = test_stacked_on_batch(victim_model, discriminator, stacked, len(testY)*[target_class], testX)
        preds = victim_model.predict_on_batch(Gx)
        victim_acc = np.sum(np.argmax(preds, axis=1) == np.argmax(testY, axis=1)) /len(testY)

        print(f'Discriminator -- Loss: {d_loss},\tAccuracy: {d_acc*100}\nGenerator -- Loss: {gan_loss}\nHinge Loss: {hinge_loss}\nVictim Loss: {adv_loss},\tAccuracy: {victim_acc*100}')
        if (epoch+1) % 10 == 0:
            for i in range(10):
                img = np.clip(Gx[i], 0, 1)
                write_img(os.path.join('output/AdvGAN/test', f'{epoch+1:04}-{i+1:02}.png'), 255*img)

        generator.save(save_path)

def generate_AE():
    generator = tf.keras.models.load_model('model/generator')
    victim_model = tf.keras.models.load_model('model/checkpoint')
    _,_,testX, testY = gtsrb(None, 'dataset/victim', (48,48))
    GX = generator.predict(testX)
    preds = victim_model.predict(GX)
    for i in range(10):
        print(f'{i+1:02}.png:\tclass id: {np.argmax(preds[i])}\n')
        write_img(f'output/AdvGAN/temp-{i+1:02}.png', 255*np.clip(GX[i],0,1))

def main():
    train_AdvGAN()
    #generate_AE()
    exit()

if __name__ == '__main__':
    main()
