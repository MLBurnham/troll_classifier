# @Date:   2018-11-10T09:54:16-05:00
# @Last modified time: 2018-11-10T22:40:00-05:00
# @Python Version: 2.6.6
# @Environment: troll_classifier

from comet_ml import Experiment
import pandas as pd
from scipy.stats import randint as sp_randint
import spacy
import string
from tokenizer import *
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initiate Comet experiment
exp = Experiment(api_key="7lwi8TtwA4XBdxzkA9EyyovhI",
                 project_name="twitter-troll-classifier", workspace="mlburnham")

# Import data
tweets = pd.read_csv('labeled_data.csv')
grouped_tweets = tweets.groupby(['author', 'class'])['text'].apply(' '.join).reset_index()

# Train_test_split
X = grouped_tweets['text']
Y = grouped_tweets['class']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .25, random_state = 0)

# Define the pipeline
tfidf = TfidfVectorizer(tokenizer = spacy_tokenizer, max_features = 5000)
mnb_grid_pipe = Pipeline([('vect', tfidf),('fit', MultinomialNB())])

# Grid search
param_grid = {'vect__ngram_range': [(1, 1), (1, 2),(1,3)],
              'vect__max_df': ( 0.7,0.8,0.9,1.0),
              'vect__min_df': (1,2),
              'fit__alpha': ( 0.022,0.025, 0.028),
              }
grid = RandomizedSearchCV(mnb_grid_pipe, param_grid,
                          cv=3, n_iter=20, n_jobs=14, random_state=0)
grid.fit(X_train, y_train)
preds = grid.predict(X_test)

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

# logs for comet_ml
params = {"random_state": 0,
          "model_type": "Multinomial NB",
          "param_dist": str(param_grid)
          }

metrics = {"accuracy": accuracy,
           "precision": precision,
           "recall": recall,
           "f1": f1
           }

exp.log_dataset_hash(X_train)
exp.log_multiple_params(params)
exp.log_multiple_metrics(metrics)
