"""
Script for evaluating our trained CC's
"""


import numpy as np
import pandas as pd 
from multilabel_evaluation import MultilabelPredictionEvaluater, MaskCreater


# Load the training labels and the corresponding predictions of our CC's:

# The correct labels. 
y_train = np.load('y_train.npy')

# Predictions by the CC  trained in chain mode with debalancing [0, 1.4]
preds_train_AG_NT_chain = np.load('preds_train_AG_NT_chain.npy')

# Predictions by the CC trained with multi-class weighting (debalancing = 1.4).
preds_train_AG_NT_multi_class_weights = np.load('preds_train_AG_NT_multi_class_weights.npy')

# Predictions by the CC trained without sample weighting 
preds_train_AG_NT_no_sample_weights = np.load('preds_train_AG_NT_no_sample_weights.npy')

# Predictions by the CC where the first classifier was trained with no sample weights while the second used automatic sample 
# weighting with debalancing set to 1.4 
preds_train_AG_NT_no_weights_automatic = np.load('preds_train_AG_NT_no_weights_automatic.npy')

preds_train_list = [preds_train_AG_NT_chain, preds_train_AG_NT_multi_class_weights, preds_train_AG_NT_no_sample_weights]
preds_train_list += [preds_train_AG_NT_no_weights_automatic]

# We now write some auxiliary functions that we use to extract subsets of our training sets 
def produce_masks_AGNT(y):
    """
    Function that creates masks allowing for extraction of interesting subsets of our AGNT training and test sets.

    More precisely masks are created that extract each of the following subsets respectively:
    samples labeled 'Algebraic geometry', samples labeled 'Number theory', samples labeled 'Algebraic geometry and not
    Number theory', samples labeled 'Number theory but not Algebraic geometry', samples labeled 'Algebraic geometry and
    Number theory' and finally all samples.

    The function returns a dictionary where the keys are descriptions of the subsets the corresponding masks extract.


    :param y: Multi-labels

    :return: Dictionary between subset descriptions and masks

    :type y: ndarray, shape = (num_samples, num_labels)

    :rtype: dict
    """
    mask_creater = MaskCreater(y)
    masks = []
    mAG = mask_creater(col_ones=[0])
    masks.append(mAG)
    mNT = mask_creater(col_ones=[1])
    masks.append(mNT)
    mAGnotNT = mask_creater(col_ones=[0], col_zeros=[1])
    masks.append(mAGnotNT)
    mNTnotAG = mask_creater(col_ones=[1], col_zeros=[0])
    masks.append(mNTnotAG)
    mAGandNT = mask_creater(col_ones=[0, 1])
    masks.append(mAGandNT)
    mAgorNT = mask_creater()
    masks.append(mAgorNT)

    names = ['Algebraic geometry', 'Number Theory', 'Algebraic geometry but not Number theory']
    names += ['Number theory but not Algebraic geometry', 'Algebraic geometry and Number theory', 'No mask']

    names_mask_dict = {key: value for (key, value) in zip(names, masks)}

    return names_mask_dict



def apply_mask_preds(mask, preds):
    """
    Function for extracting subsets of predictions.

    :param mask: Mask that extracts the subsets
    :param preds: List of predictions
    :return: List of extracted predictions

    :type mask: ndarray, entries must be of type bool
    :type preds: list, entries must be ndarrays with shape[0] = mask.shape[0]
    :rtype: list
    """
    return [pred[mask] for pred in preds]

names_mask_train = produce_masks_AGNT(y_train)

# The labels of our classes are AG for algebraic geometry and NT for number theory 
labels = ['AG', 'NT']

print('We compare the predictions on various subsets of the training set')
for key, value in names_mask_train.items():
    print(key)
    evaluater = MultilabelPredictionEvaluater(y_train[value])
    tbl = evaluater.comparison_table(predictions=apply_mask_preds(value, preds_train_list), labels=labels)
    tbl = tbl.rename({0: 'chain', 1:'multi-class weights' , 2: 'no weights', 3: 'no weight + automatic'},axis='index')
    tbl = tbl.sort_values(by=['accuracy'], ascending=False)
    print(tbl)
    print()
    print()

# Load the test labels and the corresponding predictions of our CC's

# The correct labels 
y_test = np.load('y_test.npy')

# Predictions by the CC  trained in chain mode with debalancing [0, 1.4]
preds_test_AG_NT_chain = np.load('preds_test_AG_NT_chain.npy')

# Predictions by the CC trained with multi-class weighting (debalancing = 1.4).
preds_test_AG_NT_multi_class_weights = np.load('preds_test_AG_NT_multi_class_weights.npy')

# Predictions by the CC trained without sample weighting 
preds_test_AG_NT_no_sample_weights = np.load('preds_test_AG_NT_no_sample_weights.npy')

# Predictions by the CC where the first classifier was trained with no sample weights while the second used automatic sample 
# weighting with debalancing set to 1.4 
preds_test_AG_NT_no_weights_automatic = np.load('preds_test_AG_NT_no_weights_automatic.npy')

preds_test_list = [preds_test_AG_NT_chain, preds_test_AG_NT_multi_class_weights, preds_test_AG_NT_no_sample_weights]
preds_test_list += [preds_test_AG_NT_no_weights_automatic]

names_mask_test = produce_masks_AGNT(y_test)

print('We compare the predictions on subsets of the test set' + '\n')
for key, value in names_mask_test.items():
    print(key)
    evaluater = MultilabelPredictionEvaluater(y_test[value])
    tbl = evaluater.comparison_table(predictions=apply_mask_preds(value, preds_test_list), labels=labels)
    tbl = tbl.rename({0: 'chain', 1:'multi-class weights' , 2: 'no weights', 3: 'no weights + automatic'},axis='index')
    tbl = tbl.sort_values(by=['accuracy'], ascending=False)

    print(tbl)
    print()
    print()











