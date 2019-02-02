"""Load sentences, parse them and represent as matrices (vectors for each word that match the meaning). Save all
resulted sentences"""

import pandas as pd
import numpy as np
import io
from sklearn.feature_extraction.text import CountVectorizer


def load_vectors(fname):
    """function copied from https://fasttext.cc/docs/en/english-vectors.html, instruction to load data"""
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def get_features(words, vectors):
    """look up the vector for each word in words, then stack them"""
    result = [vectors.loc[word].values for word in words if word in df_keys.values.reshape(-1)]
    if result:
        return np.stack(result)
    return None


"""Note: The structure of unwrapped_vectors is nested, that's why the keys and values are turned into DataFrame
before the words and their vectors are mapped together again"""
unwrapped_vectors = load_vectors("../data/wiki-news-300d-1M.vec")  # load words representation
df_keys = pd.DataFrame(list(unwrapped_vectors.keys()))  # create a DataFrame from keys
df_values = pd.DataFrame(list(unwrapped_vectors.values()))  # create a DataFrame from values
vectors_df = pd.DataFrame(list(df_values.values), index=df_keys.values.reshape(-1))  # put them together

sentences = np.load("../data/containAI.npy")  # load sentences containing the term 'AI'
stopwords = np.load("../data/stopwords.txt")  # load stopwords

#   create an analyzer for parsing the sentences
vectorizer = CountVectorizer(stop_words=stopwords)
analyze = vectorizer.build_analyzer()

#   parse sentences and represent them as matrices (vectors of each word)
parsed_sentences = list()
for sentence in sentences:
    sentence_matrix = get_features(analyze(sentence), vectors_df)
    if not sentence_matrix is None:
        parsed_sentences.append(sentence_matrix)

np.save("../data/sentences_vectors", parsed_sentences)
