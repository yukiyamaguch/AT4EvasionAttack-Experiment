import numpy as np
import tensorflow as tf
from model import victim_model
from prepare_dataset import gtsrb

model = tf.keras.models.load_model('model/distil')
model.summary()

_,_,testX,testY=gtsrb(None, "dataset/victim", (48,48))
# test
Y_preds = model.predict(testX)
Y_preds = np.argmax(Y_preds, axis=1)
testY = np.argmax(testY, axis=1)
acc = np.mean([1 if Y_preds[i]==testY[i] else 0 for i in range(len(testY))])
print("Accuracy: "+str(acc))

