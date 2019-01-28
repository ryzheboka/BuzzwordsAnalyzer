"""I couldn't assign a mining to clusters after experimenting with parameters
Note: A part of code is taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
"""

from sklearn.cluster import DBSCAN
import numpy as np


def filter_flatten(matrices,max_size):
    long_sentences = [(np.array(sentence)[:5],i) for i,sentence in enumerate(matrices) if 5 <= len(sentence) <= 10 and i<max_size]
    flatten_matrices = np.zeros((len(long_sentences), 1500))
    for i, sentence in enumerate(long_sentences):
        sentence = np.concatenate([np.array(word) for word in sentence[0]])
        flatten_matrices[i] = sentence
    return flatten_matrices

data = np.load("../data/sentences_vectors.npy")
data_comparison = np.load("../data/random_sentences_vectors.npy")
print([filter_flatten(data,2000).shape,filter_flatten(data_comparison,244).shape])
sentences = np.concatenate([filter_flatten(data,2000),filter_flatten(data_comparison,244)])
print(sentences.shape)

db = DBSCAN(eps=4.5, min_samples=15).fit(sentences)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = sentences[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=7)

    xy = sentences[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

""" results on my data:
Estimated number of clusters: 2
Estimated number of noise points: 181

"""