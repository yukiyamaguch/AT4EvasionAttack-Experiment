import tensorflow as tf
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

from prepare_dataset import gtsrb, write_img
from model import victim_model
from utils import l2_norm_of_perturbation

tf.compat.v1.disable_eager_execution()

# load imgs, labels
_, _, testX, testY = gtsrb(None, 'dataset/victim', (48,48))

# loadmodel
# white
#model = tf.keras.models.load_model('model/checkpoint')
# Proxy
model = tf.keras.models.load_model('model/proxy')
victim = tf.keras.models.load_model('model/checkpoint')
victim_cls = KerasClassifier(model=victim, clip_values=(0,1), use_logits=False)

# art classifier
classifier = KerasClassifier(model=model, clip_values=(0,1), use_logits=False)

# evaluate
preds = classifier.predict(testX)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(testY, axis=1)) /len(testY)
print('Test Accuracy :{:f}%'.format(accuracy*100))

# generate AE
attack = FastGradientMethod(estimator=classifier, eps=0.20, targeted=True)
testAE = attack.generate(x=testX, y=np.array(len(testY)*[3]))

# evaluate
preds = classifier.predict(testAE)
accuracy = np.sum(np.argmax(preds, axis=1) == np.argmax(testY, axis=1)) /len(testY)
print('Adv. Accuracy :{:f}%'.format(accuracy*100))
count=1
norm=0
acc=0
for i in range(len(testAE)):
    # white
    pred = classifier.predict(np.array([testAE[i]]))
    # Proxy
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
