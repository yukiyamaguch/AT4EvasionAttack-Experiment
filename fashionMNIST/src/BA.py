import tensorflow as tf
import numpy as np
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import KerasClassifier
import argparse
import os
from tensorflow.keras.utils import to_categorical

from prepare_dataset import write_img, victim_dataset
from model import victim_model
from utils import l2_norm_of_perturbation

tf.compat.v1.disable_eager_execution()
parser = argparse.ArgumentParser(description='Boundary Attack test program')
parser.add_argument('--AK', choices=['white',
                                     'proxy',
                                     'proxy0.5',
                                     'robust_white',
                                     'robust_proxy',
                                     'robust_proxy0.5'], help='Attacker\'s Knowledge.')
parser.add_argument('--query', type=int, default=1000, help='hyperparameter')
parser.add_argument('--output_path', default='.', help='save image path')
args = parser.parse_args()

# load imgs, labels
testX, testY = victim_dataset('dataset/victim')

query = args.query
mode = args.AK

# loadmodel
if mode == 'white':
    model = tf.keras.models.load_model('model/whitebox')
elif mode == 'proxy':
    model = tf.keras.models.load_model('model/proxy')
    victim = tf.keras.models.load_model('model/whitebox')
    victim_cls = KerasClassifier(model=victim, clip_values=(0,1), use_logits=False)
elif mode == 'proxy0.5':
    model = tf.keras.models.load_model('model/proxy0.5')
    victim = tf.keras.models.load_model('model/whitebox')
    victim_cls = KerasClassifier(model=victim, clip_values=(0,1), use_logits=False)
elif mode == 'robust_white':
    model = tf.keras.models.load_model('model/adv_train_victim')
elif mode == 'robust_proxy':
    model = tf.keras.models.load_model('model/proxy')
    victim = tf.keras.models.load_model('model/adv_train_victim')
    victim_cls = KerasClassifier(model=victim, clip_values=(0,1), use_logits=False)
elif mode == 'robust_proxy0.5':
    model = tf.keras.models.load_model('model/proxy0.5')
    victim = tf.keras.models.load_model('model/adv_train_victim')
    victim_cls = KerasClassifier(model=victim, clip_values=(0,1), use_logits=False)
else:
    print('Invalid Mode Error')
    exit()

# art classifier
# white
classifier = KerasClassifier(model=model, clip_values=(0,1), use_logits=False)

# evaluate
preds = classifier.predict(testX)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(testY, axis=1)) /len(testY)
print('Test Accuracy :{:f}%'.format(accuracy*100))

# generate AE
adv_init = None
iter_step = 200
#attack = BoundaryAttack(estimator=classifier, targeted=True, max_iter=10000, delta=0.01, epsilon=0.1, min_epsilon=0)
attack = BoundaryAttack(estimator=classifier, targeted=False, max_iter=query, delta=0.01, epsilon=0.01, min_epsilon=0)
#testAE = attack.generate(x=testX, y=to_categorical(len(testY)*[9], num_classes=10))
testAE = attack.generate(x=testX)

# evaluate
preds = classifier.predict(testAE)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(testY, axis=1)) /len(testY)
print('Adv. Accuracy :{:f}%'.format(accuracy*100))
count=1
norm=0
acc=0
for i in range(len(testAE)):
    # white
    if mode == 'white' or mode == 'robust_white':
        pred = classifier.predict(np.array([testAE[i]]))
    elif mode == 'proxy' or mode == 'proxy0.5' or mode == 'robust_proxy' or mode == 'robust_proxy0.5':
        pred = victim_cls.predict(np.array([testAE[i]]))
    print(f'{i+1=:02}')
    print(f'Sneaker    Acc:{pred[0][7]}')
    print(f'Ankle boot Acc:{pred[0][9]}')
    print(f'argmax   :{np.argmax(pred[0])}')
    print(f'Norm     :{l2_norm_of_perturbation(testAE[i] - testX[i])}')
    print()
    norm += l2_norm_of_perturbation(testAE[i] - testX[i])
    if np.argmax(pred[0]) == 7:
        acc += 1
    count+=1
print(f'Attack Success Rate: {1 - acc/len(testAE)}')
print(f'Mean of Norm: {norm/len(testAE)}')
print(f'Hyperparameter query: {query}')

temp = np.clip(255*testAE[0], 0, 255)
write_img(os.path.join(args.output_path, f'query{query}.png'), temp)
#temp = np.clip(255*testX[0], 0, 255)
#write_img(os.path.join(args.output_path, 'normal.png'),temp)

