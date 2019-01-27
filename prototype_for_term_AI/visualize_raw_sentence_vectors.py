"""using the guide from https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
to write the reduction og the dimensionality of the feature matrices, then plot the results """
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def filter_flatten_pca_transform(matrices):
    long_sentences = [np.array(sentence) for sentence in matrices if len(sentence) >= 10]
    flatten_matrices = np.zeros((len(long_sentences), 3000))

    for i, sentence in enumerate(long_sentences):
        sentence = sentence[:10]
        sentence = np.concatenate([np.array(word) for word in sentence])
        flatten_matrices[i] = sentence

    pca = PCA(n_components=2)

    principal_components = pca.fit_transform(flatten_matrices)
    principal_df = pd.DataFrame(data=principal_components
                               , columns=['principal component 1', 'principal component 2'])
    return principal_df

matrices_AI = np.load("../data/sentences_vectors.npy")
comparison_matrices = np.load("../data/sentences_vectors.npy")
df_AI = filter_flatten_pca_transform(matrices_AI)
df_comparison = filter_flatten_pca_transform(comparison_matrices)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_title("Sentences with occurrences of the term 'AI': 2 component PCA", fontsize = 16)

ax.scatter(df_AI["principal component 1"],df_AI["principal component 2"]
, c = 'red'
, s = 1)
ax.scatter(df_comparison["principal component 1"],df_comparison["principal component 2"]
, c = 'blue'
, s = 1)
ax.grid()
ax.legend(["occurence of the term 'AI'","random"])

plt.savefig("../results/sentences_with_the_term_AI_2d")
plt.show()
