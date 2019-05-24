import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack


class ClassifierChain(object):
    """
    Class for implementation of classifier chains using keras.Sequential models.

    Upon construction a list of binary classifiers (self.classifiers) is created. The fit method trains each of the
    classifiers to predict if the sample has the corresponding label (given the predictions of the previous
    classifiers in the chain).

    Methods:
        __init__; Constructor method.

        append_classifier: Add a classifier

        replace_classifier: Replace a classifier

        project_to_binary: Static method used to transform a multi-label to a binary label.

        compute_sample_weights: Static method returning class weights in terms of the samples.

        compute_sample_weights_chain: Static method enabling class-weights to be updated during training of the chain.

        create_batch_generator: Static method returning a python generator of training batches.

        fit_single: Trains a single classifier in the chain.

        predict_single: Static method for binary classification of samples in a sparse matrix.

        generate_next_data_set: Static method generating the data set for the next classifier in the chain.

        fit: Trains the classifier chain.

        predict: Predicts the multi-labels of the samples.

    """

    def __init__(self, classifier, num_labels, name, optimizers=None, losses=None, create_missing=False):
        """
        Constructor method. Creates a chain of classifiers to be used on multi-label classification.

        :param classifier: The first classifier to appear in the CC (classifier chain).

        :param num_labels: The number of labels in the multi-label classification task.

        :param name: The name of the classifier chain. This is used during saving.

        :param optimizers: optional, default = None. List of length num_labels -1 where each entry is a keras optimizer.

        :param losses: optional, default = None. List of length num_labels - 1 where each entry is a loss function.

        :param create_missing: optional, default = False. If set to True classifiers are automatically created
        with configurations inferred from the input classifier. These classifiers are then
        compiled with the optimizer and loss function at the corresponding index in optimizers and losses respectively.

        :type classifier: keras.Sequential model

        :type num_labels: int

        :type name: str

        :type optimizers: list

        :type losses: list

        :type create_missing: bool
        """
        self.classifier = classifier
        self.num_labels = num_labels
        self.classifiers = []
        self.classifiers.append(self.classifier)
        self.classifier_config = self.classifier.get_config()
        self.classifier_input_shape = self.classifier_config['layers'][0]['config']['batch_input_shape']

        self.name = name
        if create_missing:
            self.optimizers = optimizers
            self.losses = losses
            for i in range(1, self.num_labels):
                cfg = self.classifier_config.copy()
                cfg['layers'][0]['config']['batch_input_shape'] = \
                    (self.classifier_input_shape[0], self.classifier_input_shape[1] + i)
                self.classifiers.append(keras.models.Sequential.from_config(cfg))
                self.classifiers[i].compile(optimizer=self.optimizers[i - 1], loss=self.losses[i - 1])

    def append_classifier(self, filename):
        """
        Loads and appends a (possibly previously trained) classifier to self.classifiers.
        If len(self.classifiers) = self.num_labels the classifier is not added.

        :param filename: Name of the file that holds the classifier.

        :return: None

        :type filename: str
        """
        if len(self.classifiers) == self.num_labels:
            print('There are already as many classifiers as labels in the chain')
        else:
            self.classifiers.append(keras.models.load_model(filename))

    def replace_classifier(self, i, classifier):
        """
        Replaces the i'th classifier in our chain (we count from 0) with the given input classifier.

        :param i: The index of the classifier to be replaced.

        :param classifier: The replacement classifier.

        :return: None

        :type i: int

        :type classifier: keras Sequential model
        """
        self.classifiers[i] = classifier

    @staticmethod
    def project_to_binary(y, i):
        """
        Converts a multi-label to a binary label.

        If y is a one dimensional array consisting of zeros and ones. Then if the i'th entry equals 1
        this method returns the array ([1,0]) and otherwise it returns ([0,1]). For two-dimensional input this
        carried out row wise.

        :param y: Multi-labels.

        :param i: Index to consider.

        :return: Binary label.

        :type y: ndarray, shape = (num_samples, self.num_labels),

        :type i: int, 0 <= i <= self.num_labels - 1

        :rtype: ndarray, shape = (num_samples, 2)
        """
        ones = np.ones(y.shape[0])
        zeros = np.zeros(y.shape[0])
        M_1 = np.column_stack((ones.T, zeros))
        M_2 = np.column_stack((zeros.T, ones))
        y_i = y[:, i]
        d_i = ones - y_i
        y_ret = (M_1.T * y_i).T + (M_2.T * d_i).T
        return y_ret

    @staticmethod
    def compute_sample_weights(y, debalancing=0):
        """
        Returns class weights in terms of sample weights.

        :param y: Multi-labels.

        :param debalancing: optional, default = 0. Term to add more importance to frequently occurring labels.

        :return: Sample weights.

        :type y: ndarray, shape (num_samples, num_labels)

        :type debalancing: float

        :rtype ndarray, shape = (num_samples,)
        """
        unique_rows, inverse, counts = np.unique(y, axis=0, return_inverse=True, return_counts=True)
        weights = np.array([float(y.shape[0]) / float(counts[m]) + debalancing for m in inverse])
        weights = weights / np.amax(weights)
        return weights


    @staticmethod
    def compute_sample_weights_chain(X, y, preds_start_index, debalancing=0):
        """
        This method allows our classifier chain to automatically infer class-weights during training (in chain mode).

        The method is roughly implemented
        as follows: The i'th classifier in the chain receives a sample together with the predictions of the i-1 former
        classifiers. For each sample we extract the tuple of predictions and append the corresponding label, i.e.
        ([1,0]) or ([0,1]). We then count the number of times each unique tuple occurs and weight the corresponding
        samples by the rule total_num_samples/occurrences * (scaling factor). The debalancing parameter allows
        for giving some additional weight to (all samples) so that more frequently occurring labels get some more
        importance during training.

        :param X: Sparse matrix where the columns up to preds_start_index are features and remaining columns
        indicate which label was predicted by each (previous) binary classifier in the chain.

        :param y: The sample labels.

        :param preds_start_index: Indicates the index of the first column in X where predictions have been made.

        :param debalancing: Adds importance to more frequent labels (during training).

        :return: Sample weights

        :type X:csr-matrix, shape= (num_samples, num_features + num_predicted_labels)

        :type y: ndarray, shape = (nun_samples, 2)

        :type preds_start_index: int

        :type debalancing: float

        :rtype: ndarray, shape = (num_samples,)
        """
        y = csr_matrix(y)
        data = hstack([X, y]).tocsr()
        preds_and_label = data[:, preds_start_index:].toarray()
        weights = ClassifierChain.compute_sample_weights(preds_and_label, debalancing=debalancing)
        return weights

    @staticmethod
    def create_batch_generator(
            X, y_input, batch_size=32, shuffle=False, weights_mode=None, predefined_weights=None,
            preds_start_index=None, debalancing=0):
        """
        Creates a python generator of training batches.

        :param X: Sparse matrix of samples (and possibly labels predicted previously in the chain).

        :param batch_size: optional,default = 32. Number of samples in each batch to be yielded from the created
        generator.

        :param shuffle: optional, default = False. Whether to shuffle the samples before creating the generator.

        :param weights_mode: optional, default = None. If this input is 'chain' then samples are automatically
         weighted by the method compute_sample_weights_chain.

        :param predefined_weights: optional, default = None. Weights for the samples.
        This argument is only considered if weights_mode is None.

        :param preds_start_index: optional, default = None. Only relevant if weights_mode is set to 'chain'
        in which case this parameter refers to the index of the first column corresponding to previous predictions.

        :param debalancing: optional, default = 0. Only relevant if weights_mode is chain in which case this parameter
        adds importance to frequently occurring labels.

        :return: A batch generator

        :type X: csr-matrix, shape = (num_samples, num_features + (possibly previously predicted labels)

        :type y_input: ndarray, shape = (num_samples, 2)

        :type shuffle: bool

        :type weights_mode: str

        :type predefined_weights: ndarray, shape = (num_samples,)

        :type preds_start_index: int

        :type debalancing: float

        :rtype: generator

        """
        X_copy = X.copy()
        y_copy = csr_matrix(np.copy(y_input))
        if weights_mode == 'chain':
            if preds_start_index is None:
                preds_start_index = X_copy.shape[1]

            weights = ClassifierChain.compute_sample_weights_chain(
                X_copy, y_copy, preds_start_index=preds_start_index, debalancing=debalancing)
            weights = csr_matrix(weights)

        elif predefined_weights is not None:
            weights = csr_matrix(predefined_weights)
        else:
            weights = csr_matrix(np.ones(X.shape[0]))
        if shuffle:
            data = hstack([X_copy, y_copy, weights.transpose()])
            data = data.tocsr()
            row_indices = np.arange(X_copy.shape[0])
            np.random.shuffle(row_indices)
            data = data[row_indices]
            X_copy = data[:, :-3]
            y_copy = data[:, -3:-1]
            weights = data[:, -1].toarray().flatten()

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_copy[i: i + batch_size, :].toarray()
            y_batch = y_copy[i: i + batch_size, :].toarray()
            weights_batch = weights[i: i + batch_size]

            if weights_mode == 'chain':

                yield (X_batch, y_batch, weights_batch)

            elif predefined_weights is not None:
                yield (X_batch, y_batch, weights_batch)

            else:
                yield (X_batch, y_batch)

    def fit_single(
            self, i, X, y, epochs=1, batch_size=32, verbose=1, shuffle=True, weights_mode=None,
            predefined_weights=None, preds_start_index=None, debalancing=0, save_after=None):
        """
        Train a single classifier in the chain.

        :param i: Number of the classifier in the chain (we count from 0)

        :param X: Samples (with possibly previous predictions appended as columns)

        :param y: class labels

        :param epochs: optional, default=1. Number of training epochs.

        :param batch_size: optional, default = 32. Number of samples per gradient update.

        :param verbose: optional, default = 1. This argument is passed to classifier.fit_generator

        :param shuffle: optional, default True. Whether to shuffle the training set at the start of each epoch.

        :param weights_mode: optional, default None. If set to 'chain' samples will automatically be weighted by the
        method self.compute_sample_weights_chain.

        :param predefined_weights: optional, default = None. Weights for the samples. These weights will
        only be considered if weights_mode is None.

        :param preds_start_index: optional, default = None. Only relevant if weights_mode is chain in which case it
        indicates the index of the first column of X corresponding to previous predictions.

        :param debalancing: optional, default 0. Only relevant if weights_mode is chain in which case
        the parameter adds importance to frequently occurring labels.

        :param save_after: optional, default None. If set to 'epoch classifier' i will be saved in its entirety
        after every epoch. If set to 'completion' we save the classifier after the method is complete.


        :type i: int

        :type X: csr_matrix, shape= (num_samples, num_features + (possibly previous predictions))

        :type y: ndarray, shape= num_samples, 2)

        :type epochs: int

        :type batch_size: int

        :type verbose: int

        :type shuffle: bool

        :type weights_mode: str

        :type predefined_weights: ndarray, shape = (num_samples,)

        :type preds_start_index: int

        :type debalancing: float

        :type save_after: str


        :return: None
        """

        steps_per_epoch = int(np.ceil(X.shape[0] / batch_size))
        y_input = self.project_to_binary(y, i)
        # todo: rewrite the batch generator so it restarts when all batches have been yielded, in that way we don't
        #  need to loop over the epochs, and can use the epochs parameter in keras.Sequential.fit_generator instead.
        for epoch in range(epochs):
            print('Training classifier %d: Epoch %d/%d' % (i, epoch + 1, epochs))
            batch_generator = self.create_batch_generator(
                X, y_input, batch_size=batch_size, shuffle=shuffle, weights_mode=weights_mode,
                predefined_weights=predefined_weights, preds_start_index=preds_start_index,
                debalancing=debalancing)

            self.classifiers[i].fit_generator(
                generator=batch_generator, steps_per_epoch=steps_per_epoch, verbose=verbose)
            if save_after == 'epoch':
                self.classifiers[i].save('%s_classifier_%s_epoch_%s.h5' % (self.name, str(i), str(epoch)))
        
        if save_after == 'completion':
            if epochs > 0:
                self.classifiers[i].save('%s_classifier_%s_epochs_%s.h5' % (self.name, str(i), str(epochs)))

    @staticmethod
    def predict_single(X, classifier, batch_size=32):
        """
        The input classifier makes predictions based on the input csr-matrix X.

        :param X: sparse matrix of features (and possibly previously predicted labels)

        :param classifier: The classifier that is to perform the predictions.

        :param batch_size: optional, default = 32. How many samples to predict at a time. All predictions will be
        accumulated in an ndarray.

        :return: one dimensional ndarray of predictions (consisting of zeros and ones)

        :type X: csr-matrix, shape = (num_samples, num_features + (possibly previously predicted labels))

        :type classifier: keras Sequential model.

        :type batch_size: int

        :rtype ndarray, shape = (num_samples,)
        """
        y = np.zeros(X.shape[0])
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i: i + batch_size, :].toarray()
            y[i: i + batch_size] = (np.argmax(classifier.predict(X_batch, batch_size=batch_size), axis=1) == 0).astype(int)

        return y

    @staticmethod
    def generate_next_data_set(X, classifier, batch_size=32):
        """
         Creates the next sample in the chain classifier.

        :param X: sparse matrix of samples (and possibly previously predicted labels).

        :param classifier: The classifier used to predict from X thus generating the next data set.

        :param batch_size: optional, default = 32. The number of samples to predict at a time.

        :type X: csr-matrix, shape=(num_samples, features + (possibly previously predicted labels))

        :type classifier: keras Sequential model

        :type batch_size: int
            optional, default = 32
        :rtype : csr-matrix, shape = (X.shape[0], X.shape[1] + 1)
        """
        preds = csr_matrix(ClassifierChain.predict_single(X=X, classifier=classifier, batch_size=batch_size))
        X_next = hstack([X, preds.transpose()]).tocsr()
        return X_next

    def fit(
            self, X, y, epochs=1, batch_size=32, verbose=1,
            weights_mode=None, predefined_weights=None, debalancing=0, shuffle=True, save_after='classifier'):
        """
        Trains all classifiers in the chain

        :param X: Sparse matrix of samples .

        :param y: The sample labels.

        :param epochs: optional, default = 1. Either a list of integers where entry i corresponds to the number of
        epochs the i'th classifier is to be trained on, or an integer n in which case all classifiers
        will be trained on n epochs.

        :param batch_size: optional, default = 32. Number of samples per gradient update (for each of the classifiers).

        :param verbose: optional, default = 1. This parameter is eventually passed to each classifiers fit_generator
        method.

        :param weights_mode: optional, default = None. If set to 'chain' samples will automatically be weighted
        by the method self.compute_sample_weights_chain

        :param predefined_weights: optional, default = None. List of weights for the samples trained per classifier
        (The entries of the list must be an ndarray of shape = (num_samples,)). If the length of this list is less than
        self.num_labels the remaining classifiers are trained on automatically weighted samples if weights_mode is
        'chain'. Otherwise these samples will not be weighted.

        :param debalancing: optional, default = 0. Either a List of length self.num_labels containing float values or a
        single float value. In the latter case a list of floats equal to this given value will be considered.
        Only relevant if weights_mode is 'chain' in which case the i'th entry of the list corresponds to adding
        importance to frequently occuring samples while training the i'th classifier in the chain.

        :param shuffle: optional, default = True. Whether to shuffle the samples at the beginning of every epoch.

        :param save_after: optional, default = 'classifier'. If set to 'classifier' we make each classifier save after
        it is finished training. If set to 'epoch' it saves each classifier after every epoch the classifier has run.

        :type X: csr_matrix, shape = (num_samples, num_features)

        :type y: ndarray, shape= (num_samples, num_labels)

        :type epochs: int or list

        :type batch_size: int

        :type verbose: int

        :type weights_mode: str

        :type predefined_weights: list

        :type debalancing: float or list

        :type save_after: str
        :
        """
        if len(self.classifiers) < self.num_labels:
            raise IndexError('Method fit can only be run when there are as many classifiers as labels')
        if type(epochs) == int:
            epochs_list = self.num_labels * [epochs]
        else:
            epochs_list = epochs

        if type(debalancing) == int:
            debalancing_list = self.num_labels * [debalancing]
        else:
            debalancing_list = debalancing

        F = X.copy()
        preds_start_index = X.shape[1]
        for i in range(self.num_labels):
            if save_after == 'classifier':
                save_after = 'completion'
            elif save_after == 'epoch':
                save_after = 'epoch'

            current_mode = weights_mode
            current_weights = None

            if predefined_weights is not None:
                if i < len(predefined_weights):
                    current_weights = predefined_weights[i]
                    current_mode = None
                else:
                    predefined_weights = None

            self.fit_single(
                i=i, X=F, y=y, epochs=epochs_list[i], batch_size=batch_size, shuffle=shuffle,
                verbose=verbose, weights_mode=current_mode, predefined_weights=current_weights,
                preds_start_index=preds_start_index, debalancing=debalancing_list[i], save_after=save_after)

            if i == self.num_labels - 1:
                break

            F = self.generate_next_data_set(F, self.classifiers[i], batch_size=batch_size)

    def predict(self, X, batch_size=32):
        """
        Predicts multi-labels of the input samples.

        :param X: Sparse matrix of samples

        :param batch_size: The number of samples to predict at a time (all predictions are eventually accumulated into
        a single ndarray).

        :return ndarray of Predictions. If entry i,j is 1 then label j was predicted for sample i, otherwise it was not.

        :type X: csr-matrix, shape = (num_samples, num_features)

        :type batch_size: int

        :rtype: ndarray, shape = (num_samples, num_labels)
        """
        if len(self.classifiers) < self.num_labels:
            raise IndexError('Method predict can only be run when there are as many classifiers as labels.')
        F = X.copy()
        for i in range(self.num_labels):
            F = self.generate_next_data_set(X=F, classifier=self.classifiers[i], batch_size=batch_size)
        return F[:, X.shape[1]:].toarray()
































































