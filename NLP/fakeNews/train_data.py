import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

df = pd.read_csv('/home/mitia/Documents/DataScience/NLP/fakeNews/train.csv')
df.describe()

X_vector = None

NotNecesaryWord = [" and ", " are ", " or ", " the ", " is ", " this ", " of ", "  "]
punctuations = [",",".",";","!","?", "\"", "(", ")", "@", "'"]

"""Vectroize Text """
def processText(text_table):
    """
        Take a table of messages and transform every message to a vector representoing the frequency of words.
        @param: iterable of str
        @return: matrix of vectors, array of every word used in message
    """
    matrix = []

    for text in text_table:
        text = text.lower()
        text = " " + text + " "
        for w in NotNecesaryWord: 
            text = text.replace(w, ' ')
        for p in punctuations:
            text = text.replace(p, '')
        matrix.append(text)
    
    return matrix


    vectorText = CountVectorizer()
    X_vector = vectorText.fit_transform(matrix)
    feature_names = vectorText.get_feature_names_out()


####### TRAIN THE MODEL  #######
Y = df["target"]
X = processText(df.text)
rs = 42

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(random_state=rs))
])

pipeline.fit(X, Y)


## save the model
joblib.dump(pipeline, 'pipeline.pkl')
print("Model is trained!")
