"""using the guide from https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
to write the reduction og the dimensionality of the feature matrices, then plot the results """
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def filter_flatten_pca_transform(matrices,max_size):
    long_sentences = [(np.array(sentence)[:5],i) for i,sentence in enumerate(matrices) if 5 <= len(sentence) <= 10 and i<max_size]
    flatten_matrices = np.zeros((len(long_sentences), 1500))
    for i, sentence in enumerate(long_sentences):
        sentence = np.concatenate([np.array(word) for word in sentence[0]])
        flatten_matrices[i] = sentence

    pca = PCA(n_components=2)

    principal_components = pca.fit_transform(flatten_matrices)
    principal_df = pd.DataFrame(data=principal_components
                               , columns=['principal component 1', 'principal component 2'])
    return principal_df,len(long_sentences),[el[1] for el in long_sentences]

matrices_AI = np.load("../data/sentences_vectors.npy")
comparison_matrices = np.load("../data/sentences_vectors.npy")
raw_textes_AI = np.load("../data/containAI.npy")
df_AI,max_len,indices = filter_flatten_pca_transform(matrices_AI,2000)
df_comparison,_,_ = filter_flatten_pca_transform(comparison_matrices,max_len)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_title("Comparison of random sentences vs sentences with 'AI' occurrence (PCA)", fontsize = 14)
ax.grid()

ax.scatter(df_AI["principal component 1"],df_AI["principal component 2"]
, c = 'red'
, s = 1)
ax.scatter(df_comparison["principal component 1"],df_comparison["principal component 2"]
, c = 'blue'
, s = 1)
markers = ["d","+","x","|",">","v","<"]
for i in range(7):
    ax.scatter(df_AI["principal component 1"][i],df_AI["principal component 2"][i]
    , c = 'green'
    , s = 20, marker=markers[i])
    print("%d:%s"%(i,raw_textes_AI[indices[i]]))


ax.legend(["sentences with occurence of the term 'AI'","random sentences"])

plt.savefig("../data/probe")
plt.show()
