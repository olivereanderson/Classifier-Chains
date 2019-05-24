from AGNTanalysis import AGNT, AG_and_NT, AG_not_NT, NT_not_AG
from math_vectorizer import MathHashingVectorizer 
import numpy as np

encoder = MathHashingVectorizer(random_seed=0, n=2**21)
encode = encoder.encode
X_AG_not_NT = np.array(AG_not_NT['Titles'].apply(lambda x: encode(x, max_words=15)).tolist())
X_AG_not_NT  = list(X_AG_not_NT.flatten())

X_NT_not_AG = np.array(NT_not_AG['Titles'].apply(lambda x: encode(x, max_words=15)).tolist())
X_NT_not_AG = list(X_NT_not_AG.flatten())


def compare_encoded(x, comparison):
    counter = 0 
    for num in x:
        if num in comparison and num != 2**21:
            counter += 1
    return counter 

comp_AG_not_NT = AG_and_NT['Titles'].apply(lambda x: compare_encoded(list(encode(x, max_words=15)), X_AG_not_NT))

comp_NT_not_AG = AG_and_NT['Titles'].apply(lambda x: compare_encoded(list(encode(x, max_words=15)), X_NT_not_AG))

s_comp_AG_not_NT = comp_AG_not_NT.sum(axis=0)
s_comp_NT_not_AG = comp_NT_not_AG.sum(axis=0)
print(comp_AG_not_NT)
print(comp_NT_not_AG)
print(s_comp_AG_not_NT)
print(s_comp_NT_not_AG)


