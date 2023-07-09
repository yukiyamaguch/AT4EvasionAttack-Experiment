import tensorflow as tf
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
import argparse
import os
from tensorflow.keras.utils import to_categorical

from prepare_dataset import gtsrb, write_img
from model import victim_model
from utils import l2_norm_of_perturbation

tf.compat.v1.disable_eager_execution()
parser = argparse.ArgumentParser(description='Fast Gradient Sign Method test program')
parser.add_argument('--AK', choices=['white', 'proxy'], help='Attacker\'s Knowledge.')
parser.add_argument('--eps', type=float, default=0.1, help='hyperparameter')
parser.add_argument('--output_path', help='save image path')
args = parser.parse_args()

# load imgs, labels
_, _, testX, testY = gtsrb(None, 'dataset/victim', (48,48))

epsilon = args.eps
mode = args.AK

# loadmodel
if mode == 'white':
    model = tf.keras.models.load_model('model/checkpoint')
elif mode == 'proxy':
    model = tf.keras.models.load_model('model/proxy')
    victim = tf.keras.models.load_model('model/checkpoint')
    victim_cls = KerasClassifier(model=victim, clip_values=(0,1), use_logits=False)
else:
    print('Invalid Mode Error')
    exit()

# art classifier
classifier = KerasClassifier(model=model, clip_values=(0,1), use_logits=False)

# evaluate
preds = classifier.predict(testX)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(testY, axis=1)) /len(testY)
print('Test Accuracy :{:f}%'.format(accuracy*100))

# generate AE
attack = FastGradientMethod(estimator=classifier, eps=epsilon, targeted=True)
testAE = attack.generate(x=testX, y=to_categorical(len(testY)*[3], num_classes=43))

# evaluate
preds = classifier.predict(testAE)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(testY, axis=1)) /len(testY)
print('Adv. Accuracy :{:f}%'.format(accuracy*100))
count=1
norm=0
acc=0
for i in range(len(testAE)):
    if mode == 'white':
        pred = classifier.predict(np.array([testAE[i]]))
    elif mode == 'proxy':
        pred = victim_cls.predict(np.array([testAE[i]]))
    print(f'{i+1=:02}')
    print(f'Stop  Acc:{pred[0][14]}')
    print(f'60kmL Acc:{pred[0][3]}')
    print(f'argmax   :{np.argmax(pred[0])}')
    print(f'Norm     :{l2_norm_of_perturbation(testAE[i] - testX[i])}')
    print()
    norm += l2_norm_of_perturbation(testAE[i] - testX[i])
    if np.argmax(pred[0]) == 14:
        acc += 1
    count+=1
print(f'Attack Success Rate: {1 - acc/len(testAE)}')
print(f'Mean of Norm: {norm/len(testAE)}')
print(f'Hyperparameter eps: {epsilon}')

temp = np.clip(255*testAE[0], 0, 255)
write_img(os.path.join(args.output_path, f'eps{epsilon}.png'), temp)
temp = np.clip(255*testX[0], 0, 255)
write_img(os.path.join(args.output_path, 'normal.png'),temp)

