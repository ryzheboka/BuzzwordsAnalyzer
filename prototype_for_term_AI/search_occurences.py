import pandas as pd
import numpy as np
import re

"""
Note: The data isn't in the repository,
but downloaded from https://www.kaggle.com/quora/question-pairs-dataset
and https://www.kaggle.com/hsankesara/medium-articles

"""

def extract_sentences_from_article(art):
    ends_of_sentences = [m.start() for m in re.finditer('\.|!|\?', art)]
    occurencies_AI = [m.start() for m in re.finditer('(AI)|(Artificial Intelligence)', art)]
    sentences = [art[ends_of_sentences[e]:ends_of_sentences[e+1]] for e in range(len(ends_of_sentences)-1) for oc_ai in occurencies_AI
                 if ends_of_sentences[e]<oc_ai and ends_of_sentences[e+1]>oc_ai]
    return sentences

medium_articles_textes = pd.read_csv("../data/articles.csv")["text"]
questions = pd.read_csv("../data/questions.csv")["question1"].dropna()

articles_mask = medium_articles_textes.str.contains(" AI ")
filtered_articles = pd.concat(
    (medium_articles_textes[articles_mask],
     medium_articles_textes[-articles_mask][medium_articles_textes.str.contains(" Artificial Intelligence ")]))
sentences = [s for article in filtered_articles for s in extract_sentences_from_article(article)]
print(sentences)
question_mask = questions.str.contains(" AI ")
filtered_questions = pd.concat((
    questions[question_mask],
    questions[-question_mask][questions.str.contains(" Artificial Intelligence ")]))

np.save("../data/containAI.npy",np.concatenate((sentences,filtered_questions.values)))
