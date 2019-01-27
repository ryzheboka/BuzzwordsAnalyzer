import pandas as pd
import numpy as np
import re
import random

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


def extract_rand_sentence(art):
    ends_of_sentences = [m.start() for m in re.finditer('\.|!|\?', art)]
    rand_i = random.randint(0,len(ends_of_sentences)-2)
    return art[ends_of_sentences[rand_i]:ends_of_sentences[rand_i+1]]

medium_articles_textes = pd.read_csv("../data/articles.csv")["text"]
questions = pd.read_csv("../data/questions.csv")["question1"].dropna()

articles_mask = medium_articles_textes.str.contains(" AI ")
articles_mask2 = medium_articles_textes.str.contains(" Artificial Intelligence ")
filtered_articles = pd.concat(
    (medium_articles_textes[articles_mask],
     medium_articles_textes[np.bitwise_and(-articles_mask,articles_mask2)]))

other_articles = [extract_rand_sentence(art) for art in medium_articles_textes[np.bitwise_and(
    -articles_mask, -articles_mask2)][:700]]

sentences = list(set([s for article in filtered_articles for s in extract_sentences_from_article(article)]))
question_mask = questions.str.contains(" AI ")
question_mask2 = questions.str.contains(" Artificial Intelligence ")
filtered_questions = pd.concat((
    questions[question_mask],
    questions[np.bitwise_and(-question_mask,question_mask2)]))
filtered_questions = filtered_questions.drop_duplicates()
other_questions = list(set([q for q in questions[np.bitwise_and(-question_mask,-question_mask2)][:700]]))

np.save("../data/containAI.npy",np.concatenate((sentences,filtered_questions.values)))
np.save("../data/random_sentences.npy",np.concatenate((other_articles,other_questions)))

