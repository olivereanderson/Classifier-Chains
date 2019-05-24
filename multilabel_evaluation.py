import numpy as np
import pandas as pd


class MultilabelPredictionEvaluater(object):
    """
    Class for evaluation and comparison of multi-label classifiers.

    Most methods of this class take an ndarray of predictions as input and compare them (in some way) to the
    correct multi-labels which the constructor method stores as self.y.

    Methods:
        __init__: Constructor method.

        correct_predictions: Gives an ndarray indicating correctness of the predictions.

        correct_predictions_per_sample: Returns an ndarray indicating the number of correct predictions per sample.

        correct_predictions_per_label: Returns an ndarray indicating how many times each label was correctly predicted.

        num_correct_predictions: Returns the total number of correct predictions.

        strict_accuracy: Strict evaluation measure for multi-label classifiers.

        false_positives: Returns an ndarray indicating the presence of false positives among the predictions.

        false_positives_per_sample: Indicates the number of false positives per sample.

        false_positives_per_label: Returns an ndarray indicating the number of false positives per label.

        num_false_positives: Gives the total number of false positive predictions.

        false_negatives: Returns an ndarray indicating presence of false negatives among the predictions.

        false_negatives_per_sample: Indicates how many false negative predictions there are per sample.

        false_negatives_per_label: Indicates how many false negative predictions were made for each label.

        num_false_negatives: The total number of false negative predictions.

        accuracy: Standard evaluation measure for multi-label classifiers.

        preds_per_label: Returns the number of times each label was predicted.

        comparison_table: Returns a table comparing predictions performed by different classifiers.
    """
    def __init__(self, y):
        """
        Constructor method.
        :param y: Correct multi-labels.

        :type y: ndarray, shape = (num_samples, num_labels).
        """
        self.y = y

    def correct_predictions(self, x):
        """
        Returns an ndarray where entry i,j is True if the corresponding prediction was correct.

        :param x: Predictions

        :return: ndarray indicating correct predictions.

        :type x: ndarray, shape = (num_samples, num_labels) = self.y.shape

        :rtype : ndarray, shape = (num_samples, nun_labels) = self.y.shape = x.shape
        """
        comparison = np.equal(x, self.y)
        return comparison

    def correct_predictions_per_sample(self, x):
        """
        Returns an ndarray where the i'th entry is the number of correct predictions in sample i

        :param x: Predictions.

        :return: Array indicating the number of correct predictions per sample.

        :type x: ndarray, shape = (num_samples, num_labels) = self.y.shape

        :rtype: ndarray, shape = (num_samples,)
        """
        return np.sum(self.correct_predictions(x), axis=1)

    def correct_predictions_per_label(self, x):
        """
        Returns an ndarray where the i'th entry is the number of times the corresponding label was correctly predicted.

        :param x: Predictions.

        :return ndarray indicating the number of correct predictions per label.

        :type x: ndarray, shape = (num_samples, num_labels) = self.y.shape

        :rtype: ndarray, shape = (num_labels)
        """
        return np.sum(self.correct_predictions(x), axis=0)

    def num_correct_predictions(self, x):
        """
        Returns the total number of all correct predictions.

        :param x: Predictions.

        :return: The total number of correct predictions.

        :type x: ndarray, shape = (num_samples, num_labels) = self.y.shape

        :rtype: int
        """
        return int(np.sum(self.correct_predictions_per_sample(x), axis=0))

    def strict_accuracy(self, x):
        """
        A strict evaluation measure. Only the samples where every label was correctly predicted contribute to the score.

        :param x: Predictions.

        :return: The sum of all samples that were labeled correctly divided by the total number of samples.

        :type x: ndarray, shape = (num_samples, num_labels)

        :rtype: float
        """
        comparison = self.correct_predictions(x).astype(int)
        correct = np.ones(self.y.shape[0])
        for i in range(self.y.shape[1]):
            correct *= comparison[:, i]

        return np.sum(correct, axis=0) / self.y.shape[0]

    def false_positives(self, x):
        """
        Returns an ndarray where the i,j'th entry is True if the corresponding prediction was a false positive.

        :param x: Predictions.

        :return ndarray indicating false predictions.

        :type x: ndarray, shape = (num_samples, num_labels) = self.y.shape

        :rtype: ndarray, shape = (num_samples, num_labels) = x.shape = self.y.shape
        """
        ones = np.ones(shape=self.y.shape)
        fp = np.equal((x - self.y), ones)
        return fp

    def false_positives_per_sample(self, x):
        """
        Returns an ndarray where the i'th entry is the number of false positives predicted for sample i.

        :param x: Predictions.

        :return ndarray indicating the number of false positives per sample.

        :type x: ndarray, shape = (num_samples, num_labels) = self.y.shape

        :rtype: ndarray, shape = (num_samples,)
        """
        return np.sum(self.false_positives(x), axis=1)

    def false_positives_per_label(self, x):
        """
        Returns an ndarray indicating the number of false positives per label.

        The i'th entry of the returned ndarray is the number of times the corresponding label was a false positive
        among the predictions.

        :param x: Predictions.

        :return: ndarray indicating false positives per label.

        :type x: ndarray, shape = (num_samples, num_labels) = self.y.shape

        :rtype: ndarray, shape = (num_labels,)
        """
        return np.sum(self.false_positives(x), axis=0)

    def num_false_positives(self, x):
        """
        Returns the total number of all false positive predictions.

        :param x: Predictions.

        :return: Total number of false positive predictions.

        :type x: ndarray, shape = (num_samples, num_labels) = self.y.shape

        :rtype: int
        """
        return int(np.sum(self.false_positives_per_sample(x), axis=0))

    def false_negatives(self, x):
        """
        Returns an ndarray where the i,j'th entry is True if the corresponding prediction was a false negative.

        :param x: Predictions.

        :return: ndarray indicating false negatives.

        :type x: ndarray, shape = (num_samples, num_labels)

        :rtype: ndarray, shape = (num_samples, num_labels) = x.shape = self.y.shape
        """
        minus_ones = -1 * np.ones(shape=self.y.shape)

        fn = np.equal((x - self.y), minus_ones)

        return fn

    def false_negatives_per_sample(self, x):
        """
        Returns an ndarray where the i'th entry is the number of false negatives predicted for sample i.

        :param x: Predictions.

        :return: ndarray indicating false predictions per sample.

        :type x: ndarray, shape = (num_samples, num_labels)

        :rtype: ndarray, shape = (num_samples,)
        """
        return np.sum(self.false_negatives(x), axis=1)

    def false_negatives_per_label(self, x):
        """
        Returns an ndarray indicating the number of false positives among the labels.

        The i'th entry of the returned ndarray is the number of times the corresponding label was identified as
        a false negative among the predictions.

        :param x: Predictions.

        :return: ndarray indicating false negatives per label.

        :type x: ndarray, shape = (num_samples, num_labels) = self.y.shape

        :rtype: ndarray, shape = (num_labels,)
        """
        return np.sum(self.false_negatives(x), axis=0)

    def num_false_negatives(self, x):
        """
        Returns the total number of false negative predictions.

        :param x: Predictions.

        :return The total number of false negatives among the predictions.

        :type x: ndarray, shape = (num_samples, num_labels) = self.y.shape

        :rtype: int
        """
        return int(np.sum(self.false_negatives_per_sample(x), axis=0))

    def accuracy(self, x, lenience=None):
        """
        Standard evaluation measure for multi-label classification problems.

        This evaluation measure can for instance be found in the paper 'Classifier chains for
        multi-label classification' by Jessee Read et.al.

        :param x: Predictions.

        :param lenience: optional, default = None. If set to 'false positives' (resp. 'false negatives' )
        we are more lenient towards false positive (resp. false negative) predictions.

        :return sum of terms between 0.0 and 1/num_samples where each sample with at least one correctly predicted
        label contributes to the sum.

        :type x: ndarray, shape = (num_samples, num_labels)

        :type lenience: str

        :rtype: float
        """

        correct = self.correct_predictions_per_sample(x)

        num_labels_vector = self.y.shape[1]*np.ones(self.y.shape[0])
        fp = self.false_positives_per_sample(x)
        fn = self.false_negatives_per_sample(x)

        denominator = num_labels_vector + fp + fn

        if lenience == 'false positives':
            denominator -= fp

        elif lenience == 'false negatives':
            denominator -= fn

        return np.sum(correct/denominator, axis=0) / self.y.shape[0]

    @staticmethod
    def preds_per_label(x):
        """
        Returns an ndarray where the i'th entry is the number of times the corresponding label was predicted.

        :param x: Predictions.

        :return: ndarray indicating predictions per label.

        :type x: ndarray, shape = (num_samples, num_labels)

        :rtype: ndarray, shape = (num_labels,)
        """
        return np.sum(x, axis=0)

    def comparison_table(self, predictions, labels):
        """
        Returns a table comparing predictions performed by different classifiers.

        More precisely the method creates a pandas DataFrame with columns: strict accuracy, accuracy, false positives,
        false negatives, most false positives and most false negatives. The i'th row corresponds to the i'th element
        in the input list of predictions.

        :param predictions: List of ndarray's of shape = (num_samples, num_labels)

        :param labels: List of length num_labels where every entry is a string.

        :return: comparison table

        :type predictions: list

        :type labels: list

        :rtype: pandas DataFrame
        """
        columns = ['strict accuracy', 'accuracy', 'false positives ', 'false negatives',
                   'most false positives', 'most false negatives']
        # We will fill in the values in the pandas DataFrame by applying functions to each column. Let us first
        # set create a DataFrame of shape (len(predictions), len(columns)) where all entries of the i'th row
        # is the integer i.
        temp_list = list(np.arange(len(predictions)))
        temp_list = len(columns) * [temp_list]
        temp_array = np.array(temp_list).T
        temp_array = temp_array.astype(int)

        df = pd.DataFrame(temp_array, index=list(range(len(predictions))), columns=columns)

        # We now create the functions that are to be applied to each column of our DataFrame respectively.
        def f_1(x):
            return self.strict_accuracy(predictions[x])

        def f_2(x):
            return self.accuracy(predictions[x])

        def f_3(x):
            return self.num_false_positives(predictions[x])

        def f_4(x):
            return self.num_false_negatives(predictions[x])

        def f_5(x):
            return labels[int(np.argmax(self.false_positives_per_label(predictions[x])))]

        def f_6(x):
            return labels[int(np.argmax(self.false_negatives_per_label(predictions[x])))]

        functions_list = [f_1, f_2, f_3, f_4, f_5, f_6]
        column_function_dict = {key: value for (key, value) in zip(columns, functions_list)}

        # We iterate over the columns in df where column_series is the Series corresponding to column
        # in the list columns. For each of these columns we apply the function corresponding to column and update
        # the values in df.
        for column, column_series in df.iteritems():
            df.update(column_series.apply(column_function_dict[column]))

        return df


class MaskCreater(object):
    """
    Class for extracting several interesting sub-sets from a given data set.

    Methods:
        __init__: Constructor method.
        __call__: Creates a mask that can be used to extract the sub-set where certain labels (don't) appear.
    """
    def __init__(self, y):
        """
        Constructor method.

        :param y: Multi-labels
        :type y: ndarray, shape = (num_samples, num_labels).
        """
        self.y = y

    def __call__(self, col_ones=None, col_zeros=None):
        """
        Call method.

        This method constructs a mask that can be used to extract the subset of our data set of samples labeled with
        a given set of labels and not labeled with another given set of labels.

        :param col_ones: List of indices corresponding to the columns corresponding to the labels that should appear.

        :param col_zeros: List of indices corresponding to the columns corresponding to the labels that we do not
         want to appear.

        :return: Mask to extract the desired sub-data set.

        :type col_ones: list

        :type col_zeros: list

        :rtype: ndarray, shape = (num_labels,)

        """
        ones = np.ones(shape=self.y.shape)
        zeros = np.zeros(shape=self. y.shape)
        comp_ones = np.equal(self.y, ones)
        comp_zeros = np.equal(self.y, zeros)

        one_mask = np.ones(self.y.shape[0]).astype(bool)

        if col_ones is None:
            col_ones = []

        for index in col_ones:
            one_mask *= comp_ones[:, index]

        zero_mask = np.ones(self.y.shape[0]).astype(bool)

        if col_zeros is None:
            col_zeros = []

        for index in col_zeros:
            zero_mask *= comp_zeros[:, index]

        return one_mask * zero_mask




































