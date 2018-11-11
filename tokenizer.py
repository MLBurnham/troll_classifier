# @Date:   2018-11-10T18:42:50-05:00
# @Last modified time: 2018-11-10T18:46:16-05:00
import spacy
import string

nlp = spacy.load('en', disable = ['parser', 'ner', 'textcat'])
stopwords = spacy.lang.en.stop_words.STOP_WORDS
punctuations = string.punctuation
not_allowed = string.punctuation + string.digits

def spacy_tokenizer(tweet):
    """Tokenizes, removes stop words, and lemmatizes corpus"""
    tweet = nlp(tweet)
    tweet = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tweet]
    tweet = [tok for tok in tweet if (tok not in stopwords and tok not in not_allowed)]
    return tweet
