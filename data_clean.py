# @Author: Mike Burnham <MLBurnham>
# @Date:   2018-11-08T22:21:37-05:00
# @Project: Twitter Troll Classifier
# @Last modified by:   MLBurnham
# @Last modified time: 2018-11-09T22:00:40-05:00
# @Python Version: 2.6.6
# @Environment: troll_classifier

import pandas as pd
import re

congress_tweets = pd.read_csv("politician_tweets.csv")
troll_tweets = pd.read_csv("troll_tweets.csv")
trump_tweets = pd.read_csv("trump_tweets.csv", encoding ='Windows-1252')

# Congress
congress_tweets = congress_tweets[['Handle', 'Tweet']]
congress_tweets.columns = ['author', 'text']
congress_tweets['class'] = 0

# Trump
trump_tweets['author'] = 'realdonaldtrump'
trump_tweets['class'] = 0
trump_tweets = trump_tweets.sample(n = 200, random_state = 1)

# Trolls
troll_tweets = troll_tweets[['author', 'content']]
troll_tweets['class'] = 1
troll_tweets.rename(columns = {'content': 'text'}, inplace = True)
troll_tweets = troll_tweets.sample(n = 67759, random_state = 2) # sampling to match politician tweets

# Merging
labeled_tweets = pd.merge(congress_tweets, trump_tweets, how = 'outer')
labeled_tweets = pd.merge(labeled_tweets, troll_tweets, how = 'outer')
print(labeled_tweets.shape)

# Dropping ULRs
def drop_characters(tweet):
    """Drops URLs as well as characters not caught by spaCy"""
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'-', '', tweet)
    tweet = re.sub(r'\.', '', tweet)
    tweet = re.sub(r'"', '', tweet)
    return tweet
labeled_tweets['text'] = labeled_tweets['text'].apply(drop_characters)

grouped_tweets = labeled_tweets.groupby(['author', 'class'])['text'].apply(' '.join).reset_index()

pd.to_csv("")
