import pandas as pd
import numpy as np
import io
from sklearn.feature_extraction.text import CountVectorizer


"""function copied from https://fasttext.cc/docs/en/english-vectors.html"""
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def get_features(words,vectors):
    result = [vectors.loc[word].values for word in words if word in df_keys.values.reshape(-1)]
    if result:
        return np.stack(result)
    return None

"""The structure of unwrapped_vectors is nested, that's why they are turned into a Dataframe
before the words and their vectors are mapped together again"""
unwrapped_vectors = load_vectors("../data/wiki-news-300d-1M.vec")
df_keys = pd.DataFrame(list(unwrapped_vectors.keys()))
df_values = pd.DataFrame(list(unwrapped_vectors.values()))
vectors_df = pd.DataFrame(list(df_values.values), index=df_keys.values.reshape(-1))

sentences = np.load("../data/containAI.npy")
features = dict()
stopwords = np.load("../data/stopwords.txt")
vectorizer = CountVectorizer(stop_words=stopwords)
analyze = vectorizer.build_analyzer()

parsed_sentences = list()
for sentence in sentences:
    sentence_matrix = get_features(analyze(sentence),vectors_df)
    if not sentence_matrix is None:
        parsed_sentences.append(sentence_matrix)

np.save("../data/sentences_vectors",parsed_sentences)
