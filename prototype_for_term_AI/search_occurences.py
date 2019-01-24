import pandas as pd
import numpy as np

"""
Note: The data isn't in the repository,
but downloaded from https://www.kaggle.com/quora/question-pairs-dataset
and https://www.kaggle.com/hsankesara/medium-articles

"""
medium_articles_textes = pd.read_csv("../data/articles.csv")["text"]
questions = pd.read_csv("../data/questions.csv")["question1"].dropna()

articles_mask = medium_articles_textes.str.contains(" AI ")
filtered_articles = pd.concat(
    (medium_articles_textes[articles_mask],
     medium_articles_textes[-articles_mask][medium_articles_textes.str.contains(" Artificial Intelligence ")]))

question_mask = questions.str.contains(" AI ")
filtered_questions = pd.concat((
    questions[question_mask],
    questions[-question_mask][questions.str.contains(" Artificial Intelligence ")]))

np.save("../data/containAI.npy",pd.concat((filtered_articles,filtered_questions)).values)
