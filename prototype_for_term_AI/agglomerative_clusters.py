"""Intuitively, the clusters seem to make sense. Still need a better approach."""


from sklearn.cluster import AgglomerativeClustering
import numpy as np


def filter_flatten(matrices,max_size):
    long_sentences = [(np.array(sentence)[:5],i) for i,sentence in enumerate(matrices) if 5 <= len(sentence) <= 10 and i<max_size]
    flatten_matrices = np.zeros((len(long_sentences), 1500))
    for i, sentence in enumerate(long_sentences):
        sentence = np.concatenate([np.array(word) for word in sentence[0]])
        flatten_matrices[i] = sentence
    return flatten_matrices,[el[1] for el in long_sentences]


data,labels_indices = filter_flatten(np.load("../data/sentences_vectors.npy"),2000)
data_comparison,comp_indices = filter_flatten(np.load("../data/random_sentences_vectors.npy"),2000)
labelsAI = np.load("../data/containAI.npy")[labels_indices].reshape((-1,1))
labels_random = np.load("../data/random_sentences.npy")[comp_indices].reshape((-1,1))
all_labels = np.concatenate((labelsAI,labels_random))

number_of_clusters = 2
for linkage in (["complete"]):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=number_of_clusters)
    results = clustering.fit_predict(np.concatenate((data,data_comparison)))

print(all_labels[results==0])
