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


class MathHashingVectorizer(object):
    """Class for transformation of (mathematical) text to vectors with numerical entries.

        The typical use of this class is to transform texts by first removing certain stopwords stored in (self.stop)
        and then turning the result into sparse vectors that can be used as samples when training a classifier.

        Methods:
            __init__: Constructor method.

            seeded_mmh3: Seeded hashing function based on Murmurhash3.

            update_stopwords: Append stopwords.

            remove_from_text: Removes all stopwords from the input text.

            porter_stem: Applies Porter stemming to the input text.

            stem: Applies Porter stem to all text not in math mode.

            encode: Returns an ndarray where entries correspond to hashing values of the input text.

            vectorize_encoded: Transforms a pandas Series of encoded values to a sparse matrix.

            vectorize: Transforms a pandas Series of texts to a sparse matrix suitable for training a classifier.
    """

    def __init__(self, random_seed=123, n=2**21):
        """
        Constructor method.

        :param random_seed: set a random seed for reproducibility.
        :param n: The dimension of the hashing space, or in other words the number of features.

        :type random_seed: int
        :type n: int
        """
        self.random_seed = random_seed
        self.n = n
        self.math_stopwords = ['proof', 'theorem', 'proposition', 'definition', 'lemma', 'counterexample']
        self.math_stopwords += ['counterexamples', 'conjecture', 'conjectures', 'proofs', 'definitions']
        self.math_stopwords += ['theorems', 'propositions', 'lemmas']
        self.math_stopwords += ['Generalization', 'result', 'generalization', 'appendix', 'corollary']
        self.math_stopwords += ['generalizations', 'generalisation', 'generalisations']
        self.stop = stopwords.words('english') + stopwords.words('french') + self.math_stopwords
        self.porter = PorterStemmer()

    def seeded_mmh3(self, x):
        """
        Seeded non-cryptographic hash function found in the mmh3 library (a python wrapper for Murmurhash 3).
        :param x: text
        :return: hash value

        :type x: str
        :rtype: int
        """
        return mmh3.hash(x, seed=self.random_seed)

    def update_stopwords(self, x):
        """
        Adds stopwords.
        :param x: List of strings to be appended to self.stopwords

        :type x: list
        """
        self.stop += x

    def remove_from_text(self, text):
        """
        Removes all stopwords from the given string.
        :param text: Text
        :return: Text with all words contained in self.stopwords removed.

        :type text: str
        :rtype: str
        """
        text = ' '.join(i for i in text.split() if i.lower() not in self.stop)
        return text

    def porter_stem(self, text):
        """
        stems the words appearing in the input text using PorterStem()

        :param text: Text to be stemmed.
        :return: Stemmed text.

        :type text: str
        :rtype: str
        """
        stem_split = [self.porter.stem(word) for word in text.split()]
        return ' '.join(word for word in stem_split)

    def stem(self, text):
        """
        This method applies PorterStemmer to all text not in math_mode (that is not wrapped in $ symbols).

        ::warning: This method has not been tested yet.

        :param text: Text to be transformed
        :return: Stemmed text

        :type text: str
        :rtype: str
        """

        print('This is an early version of stem and we advice caution')
        if text == '':
            return ''

        if text.count('$') % 2 != 0:
            print('text contains an odd number of $ symbols and we will treat it wit porter_stem')
            return self.porter_stem(text)

        txt = text
        transformed_text = ''
        while len(txt) > 0:
            first_occurrence = txt.find('$')
            second_occurrence = txt.find('$', first_occurrence + 1)
            if first_occurrence != 0:
                transformed_text += self.porter_stem(txt[: first_occurrence]) + ' ' + \
                                    txt[first_occurrence: second_occurrence + 1] + ' '
                txt = txt[second_occurrence + 1:]
            else:
                transformed_text += txt[:second_occurrence + 1] + ' '
                txt = txt[second_occurrence + 1:]

        return transformed_text

    def encode(self, text, max_words=40, hash_function=None, stemming=False):
        """
        Removes stopwords from a text and encodes the result into an array of integers using a hashing function.

        :param text: Text to be encoded.

        :param max_words: Maximum number of words in the string to be considered.

        :param hash_function: optional, default = None. A hashing function. If no hashing function is provided
        self.seeded_mmh3 will be applied.

        :param stemming: optional, default = False. If set to True we stem the words in the text (using self.stem)
        before encoding.

        :return: ndarray of hashing values.

        :type text: str

        :type max_words: int

        :type hash_function: function

        :type stemming: bool

        :rtype: ndarray, shape = (max_words,)
        """

        text = self.remove_from_text(text)
        if stemming:
            text = self.stem(text)

        filters = '!"$\n \t'

        if hash_function is None:
            hash_function = self.seeded_mmh3

        pre_vec = keras.preprocessing.text.hashing_trick(
            text=text, n=self.n, hash_function=hash_function, lower=False, filters=filters)

        vec = pre_vec[:max_words]

        if len(vec) < max_words:
            m = max_words - len(vec)
            extension = ((self.n) * np.ones(m)).astype(int)
            vec = np.concatenate((vec, extension))

        return vec

    def vectorize_encoded(self, df, max_words=40, mode='binary'):
        """
        Transforms a pandas Series of ndarray's to a csr-matrix.

         The non-zero entries of each row in the returned sparse matrix corresponds to the values in the corresponding
         ndarray of hashing values.

        :param df: Series where each entry is an ndarray of shape (max_words,) .

        :param max_words: optional, default = 40. Maximum number of words to consider.

        :param mode: optional, default = 'binary'. If set to 'binary' a non-zero entry is set to 1 if the index of the
        column (in the given row) appeared in the corresponding ndarray . If mode is set to 'count' then an entry d in
        a column means that the index of this column appeared d times in the corresponding ndarray.

        :return: Sparse matrix

        :type df: pandas Series

        :type max_words: int

        :type mode: str

        :rtype: csr_matrix, shape = (df.shape[0], self.n)
        """
        # We will construct a csr_matrix from (data, row_ind, col_ind)
        s = df.shape[0]
        columns_per_row = df.apply(lambda x: np.unique(x))
        columns_per_row = columns_per_row.apply(lambda x: x.astype(int))
        columns_per_row = columns_per_row.apply(lambda x: list(x))
        columns_lists = columns_per_row.tolist()
        cols = list(itertools.chain.from_iterable(columns_lists))
        cols = np.array(cols)
        # our col_ind will be cols. Let us now create our row_ind
        lengths = list(columns_per_row.apply(lambda x: len(x)).values)
        rows = []
        for i in range(s):
            v = lengths[i] * [i]
            rows += v
        rows = np.array(rows)
        # rows is our row_ind. We now find the data to insert into our sparse matrix
        if mode == 'count':
            def counts(x):
                uniq, ret_counts = np.unique(x, return_counts=True)
                return ret_counts

            counted = df.apply(counts)
            counted = df.apply(lambda x: list(x)).tolist()
            data = list(itertools.chain.from_iterable(counted))
            data = np.array(data)

        else:
            data = np.ones(cols.shape[0])

        X = csr_matrix((data, (rows, cols)), shape=(df.shape[0], self.n + 1))
        return X[:, : -1]

    def vectorize(self, df, max_words=40, mode='binary'):
        """
        Transforms a pandas Series of strings to a csr-matrix with integer entries suitable for training a classifier.

        The transformation is carried out by first applying self.encode to each entry of the series and then applying
        self.vectorize_encoded to the result.

        :param df: Series of (mathematical) text to be transformed.

        :param max_words: optional, default = 40. Maximum number of words to consider in each entry of the Series

        :param mode: 'binary' or 'count'. see the documentation for self.vectorize_encoded.

        :return: Sparse matrix of vectorized samples.

        :type df: pandas Series

        :type max_words: int

        :type mode: str

        :rtype csr-matrix, shape = (df.shape[0], self.n)
        """
        encoded_df = df.apply(lambda x: self.encode(x, max_words=max_words))
        return self.vectorize_encoded(df=encoded_df, max_words=max_words, mode=mode)


