import cv2
import tensorflow as tf
import numpy as np
from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import KerasClassifier

from prepare_dataset import gtsrb, write_img, read_imgs_from_dir
from model import victim_model
from utils import l2_norm_of_perturbation

tf.compat.v1.disable_eager_execution()

# load imgs, labels
_, _, testX, testY = gtsrb(None, 'dataset/victim', (48,48))
imgs, _= read_imgs_from_dir('dataset/victim')

# loadmodel
#model = tf.keras.models.load_model('model/checkpoint')
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
#attack = UniversalPerturbation(classifier=classifier, attacker='fgsm', eps=15, norm=2, delta=0.1, verbose=True)
attack = UniversalPerturbation(classifier=classifier, attacker='fgsm', norm=2, delta=0.1, verbose=True)
testAE = attack.generate(x=testX,y=testY, eps=2.0)
uae = attack.noise
print(f'uae min: {np.min(uae)},\tuae max: {np.max(uae)}')
testAE = np.array([testX[i] + uae[0] for i in range(len(testX))])
testAE = np.clip(testAE, 0,1)
print(testAE.shape)

#preds = classifier.predict(testAE)
preds = victim_cls.predict(testAE)
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

uae = 255*cv2.resize(uae[0], (256,256))
output = np.clip(imgs[0] + uae, 0, 255)
write_img('temp.png', output)
