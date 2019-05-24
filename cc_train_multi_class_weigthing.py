import numpy as np
import scipy.sparse 
import tensorflow as tf
import tensorflow.keras as keras
from classifier_chains import ClassifierChain

np.random.seed(0)
tf.set_random_seed(0)

X_train = scipy.sparse.load_npz('X_train.npz')
X_test = scipy.sparse.load_npz('X_test.npz')

y_train = np.load('y_train.npy')

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=50, input_dim=X_train.shape[1], activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(units=2, input_dim=50, activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(lr=0.1, decay=1e-15, momentum=.9)

model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy')

optimizers= [sgd_optimizer]
losses = ['binary_crossentropy']

cc = ClassifierChain(classifier=model, num_labels=2, name='AG_NT_multi_class_weights', optimizers=optimizers, losses=losses, create_missing=True)

sample_weights = cc.compute_sample_weights(y_train, debalancing=1.4)

predefined_sample_weights = [sample_weights, sample_weights]

cc.fit(X=X_train, y=y_train, epochs=15, batch_size=64, predefined_weights=predefined_sample_weights)

preds_train  = cc.predict(X_train)

preds_test = cc.predict(X_test)

np.save('preds_train_AG_NT_multi_class_weights.npy', preds_train)

np.save('preds_test_AG_NT_multi_class_weights.npy', preds_test)


