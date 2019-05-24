import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import mmh3
from nltk.stem.porter import PorterStemmer
# stopwords can be downloaded by importing nltk to python and then executing the command nltk.download('stopwords')
from nltk.corpus import stopwords
# We will use itertools to efficiently flatten lists of lists
import itertools
# Due to the huge amount of data we will need sparse matrices
from scipy.sparse import csr_matrix

from math_vectorizer import MathHashingVectorizer


np.random.seed(123)
tf.set_random_seed(123)

data = pd.read_csv('AGNT.csv')
print(data.head())

random_training_data = data.sample(frac=0.8, random_state=123)
random_test_data = data.drop(index=random_training_data.index)

vectorizer = MathHashingVectorizer()


X_train = vectorizer.vectorize(random_training_data['Titles'], max_words=15)
print(X_train)
y_train = random_training_data[random_training_data.columns[1:]].values

X_test = vectorizer.vectorize(random_test_data['Titles'], max_words=15)
y_test = random_test_data[random_test_data.columns[1:]].values

# model = keras.models.Sequential()

# model.add(keras.layers.Dense(units=50, input_dim=X_train.shape[1], activation='relu'))
# model.add(keras.layers.Dropout(0.1))
# model.add(keras.layers.Dense(units=2, input_dim=50, activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(lr=0.1, decay=1e-15, momentum=.9)

# model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy')

optimizers = [sgd_optimizer]
losses = ['binary_crossentropy']

from classifier_chains import ClassifierChain

# Added after running 15 epochs 
model = keras.models.load_model('classifier_0_epochs_15.h5')
chain = ClassifierChain(classifier=model, num_labels=2, optimizers=optimizers, losses=losses, name='chain')
epochs = [0, 15]
debalancing = [0, 1.7]
# chain.fit_single(0, X=X_train, y=y_train, epochs=15, batch_size=64, weights_mode='chain', save_after='completion')
chain.fit(X=X_train, y=y_train, epochs=epochs, batch_size=64, weights_mode='chain', debalancing=debalancing)






#
#mean_vals = np.mean(X_train, axis=0)
#if np.amin(np.std(X_train, axis=0)) > 0.01:
#    std_val = np.std(X_train)
#else:
#    std_val = np.std(X_train)
#
#X_train_centered = (X_train - mean_vals)/std_val
#y_train = random_training_data[random_training_data.columns[1:]].values
#print(X_train_centered)
#
#model = keras.models.Sequential()
#
#model.add(keras.layers.Dense(units=20, input_dim=X_train_centered.shape[1], activation='relu'))
#model.add(keras.layers.Dense(units=20, input_dim=20, activation='relu'))
#model.add(keras.layers.Dense(units=2, input_dim=20, activation='softmax'))
#
#sgd_optimizer = keras.optimizers.SGD(lr=0.1, decay=1e-7, momentum=.9)
#
#model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy')
#
#from classifier_chains import ClassifierChain, DataSetDistributor
#
#y_train0 = ClassifierChain.project_to_binary(y_train, 0)
#
#weights = ClassifierChain.compute_sample_weights_chain(
#    X_train_centered, y_train0, preds_start_index=X_train_centered.shape[1], debalancing_term=0.01)
#
#
#
#
# print(model.layers)

# training_generator, steps_per_epoch = balanced_batch_generator(X_train_centered, y_train, batch_size=10, random_state=123)










