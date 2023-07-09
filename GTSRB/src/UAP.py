import tensorflow as tf
import numpy as np
from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import KerasClassifier
import argparse
import os
from tensorflow.keras.utils import to_categorical

from prepare_dataset import gtsrb, write_img, read_imgs_from_dir
from model import victim_model
from utils import l2_norm_of_perturbation

tf.compat.v1.disable_eager_execution()
parser = argparse.ArgumentParser(description='Universal Adversarial Perturbation test program')
parser.add_argument('--AK', choices=['white', 'proxy'], help='Attacker\'s Knowledge.')
parser.add_argument('--eps', type=float, default=1.0, help='hyperparameter')
parser.add_argument('--output_path', help='save image path')
args = parser.parse_args()

# load imgs, labels
_, _, testX, testY = gtsrb(None, 'dataset/victim', (48,48))
#imgs, _= read_imgs_from_dir('dataset/victim')

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
#attack = UniversalPerturbation(classifier=classifier, attacker='fgsm', eps=15, norm=2, delta=0.1, verbose=True)
attack = UniversalPerturbation(classifier=classifier, attacker='fgsm', eps=epsilon, norm=2, delta=0.1, verbose=True)
testAE = attack.generate(x=testX,y=testY, eps=2.0)
uae = attack.noise
print(f'uae min: {np.min(uae)},\tuae max: {np.max(uae)}')
testAE = np.array([testX[i] + uae[0] for i in range(len(testX))])
testAE = np.clip(testAE, 0,1)
print(testAE.shape)

if mode == 'white':
    preds = classifier.predict(np.array([testAE[i]]))
elif mode == 'proxy':
    preds = victim_cls.predict(np.array([testAE[i]]))
norm=0
acc=0
for i in range(len(testX)):
    norm += l2_norm_of_perturbation(testAE[i]-testX[i])
    if np.argmax(preds[i]) == 14:
        acc+=1
print(f'Mean of Norm: {norm/len(testAE)}')
print(f'diff of id 0: {l2_norm_of_perturbation(testAE[0]-testX[0])}')
# evaluate
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(testY, axis=1)) /len(testY)
print('Adv. Accuracy :{:f}%'.format(accuracy*100))
print(f'Acc: {acc/len(testX)}')

temp = np.clip(255*testAE[0], 0, 255)
write_img(os.path.join(args.output_path, f'delta{delta}.png'), temp)
temp = np.clip(255*testX[0], 0, 255)
write_img(os.path.join(args.output_path, 'normal.png'),temp)
