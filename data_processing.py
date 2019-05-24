""" 
Script for converting the titles in AGNT.csv to sparse vectors with 2**21 features that can be used for training our
chain classifier. In this process we will construct a training set and a test set that we save for later use.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import scipy.sparse 
from math_vectorizer import MathHashingVectorizer
# We set the random seed for reproducibility
np.random.seed(1)
tf.set_random_seed(1)

data = pd.read_csv('AGNT.csv')

# Extract 80% of the data frame at random for the training sample and the 
# remaining 20% for testing.

random_training_data = data.sample(frac=0.8, random_state=1)
random_test_data = data.drop(index=random_training_data.index)

# Now we start vectorizing our training data 
vectorizer = MathHashingVectorizer(random_seed=1, n=2**21)
X_train = vectorizer.vectorize(random_training_data['Titles'], max_words=15)
# Create an ndarray with the corresponding training labels 
y_train = random_training_data[random_training_data.columns[1:]].values

# Similarly we construct the data set for testing
X_test = vectorizer.vectorize(random_test_data['Titles'], max_words=15)
y_test = random_test_data[random_test_data.columns[1:]].values 

# Finally we save our work

# We first save the data frames 

store = pd.HDFStore('training_and_test_df.h5')
store['random_training_data'] = random_training_data 
store['random_test_data'] = random_test_data

# Now we store the sparse matrices of training and test features respectively. 

scipy.sparse.save_npz('X_train.npz', X_train)
scipy.sparse.save_npz('X_test.npz', X_test)

# The training and test labels are numpy ndarrays and are also easily saved. 

np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)










