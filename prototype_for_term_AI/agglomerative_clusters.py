"""Intuitively, the clusters seem to make sense. Still need a better approach."""

from sklearn.cluster import AgglomerativeClustering
import numpy as np


def filter_flatten(matrices, max_size):
    """ find maximum max_size sentences of the length between 5 and 10 words, then flatten the representations of the words into one
    vector. Return the list of vectors of all sentences (length between 5 and 10 words) and also indices of the sentences,
    so their text can be looked up (text and matrix representation of a text have the same index, but are in different
     files"""
    long_sentences = [(np.array(sentence)[:5], i) for i, sentence in enumerate(matrices) if 5 <= len(sentence) <= 10
                      and i < max_size]
    flatten_matrices = np.zeros((len(long_sentences), 1500))
    for i, sentence in enumerate(long_sentences):
        sentence = np.concatenate([np.array(word) for word in sentence[0]])
        flatten_matrices[i] = sentence
    return flatten_matrices, [el[1] for el in long_sentences]

# sentences containing 'AI'
data, labels_indices = filter_flatten(np.load("../data/sentences_vectors.npy"), 2000)
labelsAI = np.load("../data/containAI.npy")[labels_indices].reshape((-1, 1))    # according texts

# random sentences
data_comparison, comp_indices = filter_flatten(np.load("../data/random_sentences_vectors.npy"), 2000)
labels_random = np.load("../data/random_sentences.npy")[comp_indices].reshape((-1, 1))  # according texts


all_labels = np.concatenate((labelsAI, labels_random))

# clustering
number_of_clusters = 2
linkage = ["complete"]
clustering = AgglomerativeClustering(linkage=linkage, n_clusters=number_of_clusters)
results = clustering.fit_predict(np.concatenate((data, data_comparison)))

# print all sentences from the first cluster
print(all_labels[results == 0])
