# Refference: https://duongnt.com/adversarial-example/ Accessed: 2022/12/16
import os
import tensorflow as tf
import numpy as np
import cv2

from model import victim_model
from prepare_dataset import gtsrb, write_img
from tensorflow.keras.utils import to_categorical

INITIAL_LEARNING_RATE = 1e-3

def calculate_lr(grad_max, grad_min):
    grad_abs_max = max(np.abs(grad_max), np.abs(grad_min))
    return INITIAL_LEARNING_RATE / grad_abs_max

# settings
# loop settings
eps = 1e-2
iterate = 10000
# output path
path = 'output/RP2/white'
#path = 'output/RP2/proxy'

# target class
t = 3 # 60km Limit
t = to_categorical(t, num_classes=43)

# load imgs, labels
_,_, testX, testY = gtsrb(None, 'dataset/victim', (48,48))
# load mask
mask = cv2.imread('dataset/mask2.png')
mask = cv2.resize(mask, (48,48))
mask = mask / 255.0

# load model
model = tf.keras.models.load_model('model/checkpoint')

count = 1
for image in testX:
    print(f'{count:02}.png')
    image=np.array([image])
    # initial perturbation
    perturbation = np.random.rand(1,48,48,3)
    # initial Adversarial Example
    AE = tf.Variable(np.multiply(mask, perturbation))
    # numpy mask to tf mask
    mask = tf.constant(mask)

    # learning AE
    for i in range(iterate):
        with tf.GradientTape() as tape:
            tape.watch(AE)
            # Mask(AE) = AE times mask
            MaskAE = tf.math.multiply(AE, mask)
            # INPUT = AE + image
            INPUT = tf.add(image, MaskAE)
            ClippedINPUT = tf.clip_by_value(INPUT, clip_value_min=0, clip_value_max=1)
            diff = tf.subtract(t, model(ClippedINPUT))
            diff2 = tf.math.square(diff)
            diff3 = tf.math.reduce_sum(diff2)
            loss = tf.math.sqrt(diff3)
        gradient = tape.gradient(loss, AE)
        gradient_max = tf.math.reduce_max(gradient).numpy()
        gradient_min = tf.math.reduce_min(gradient).numpy()

        learning_rate = calculate_lr(gradient_max, gradient_min)
        AE.assign_sub(gradient * learning_rate)
        if loss < eps:
            # print('Stop training at step {}/{}, loss: {}'.format(i, iterate, loss))
            break
        #if i%100 == 0:
        #    print('Step {}/{}, loss: {}'.format(i, iterate, loss))

    perturbation = AE.numpy()
    ae=np.clip(perturbation, 0,1)
    aeimg = np.clip(perturbation + image, 0, 1)
    pred = model.predict(aeimg)
    print('60kmL Preds: {}'.format(pred[0][3]))
    print('Stop  Preds: {}'.format(pred[0][14]))
    print(f'argmax: {np.argmax(pred[0])}')
    write_img(os.path.join(path, f'AE-{count:02}.png'), 255*ae[0])
    count += 1
    print()



