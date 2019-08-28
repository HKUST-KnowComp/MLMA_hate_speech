import re
from collections import Counter
import os
import matplotlib
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report
from annotated_data_processing import clean_text
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from skmultilearn.problem_transform import ClassifierChain
from sklearn.dummy import DummyClassifier
from constants import LABELS

#majority voting for multilabel tasks: annotator's sentiment and hostility type (tweet sentiment)
def lr_multilabel_classification(train_filename, dev_filename, test_filename, attribute):
  df_train = pd.read_csv(train_filename)
  df_dev = pd.read_csv(dev_filename)
  df_test = pd.read_csv(test_filename)
  mlb = MultiLabelBinarizer()
  X_train = df_train.tweet.apply(clean_text)
  y_train_text = df_train[attribute].apply(lambda x: x.split('_'))
  y_train = mlb.fit_transform(y_train_text)
  X_dev = df_dev.tweet.apply(clean_text)
  y_dev_text = df_dev[attribute].apply(lambda x: x.split('_'))
  y_dev = mlb.fit_transform(y_dev_text)
  X_test = df_test.tweet.apply(clean_text)
  y_test_text = df_test[attribute].apply(lambda x: x.split('_'))
  y_test = mlb.fit_transform(y_test_text)
  count_vect = CountVectorizer()
  X_train_counts = count_vect.fit_transform(X_train)
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  Y = mlb.fit_transform(y_train_text)
  classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', ClassifierChain(LogisticRegression()))])
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print('Test macro F1 score is %s' % f1_score(y_test, y_pred, average='macro'))
  print('Test micro F1 score is %s' % f1_score(y_test, y_pred, average='micro'))
  
#majority voting for multilabel tasks: annotator's sentiment and hostility type (tweet sentiment)
def majority_voting_multilabel_classification(train_filename, dev_filename, test_filename, attribute):
  df_train = pd.read_csv(train_filename)
  df_dev = pd.read_csv(dev_filename)
  df_test = pd.read_csv(test_filename)
  mlb = MultiLabelBinarizer()
  X_train = df_train.tweet.apply(clean_text)
  y_train_text = df_train[attribute].apply(lambda x: x.split('_'))
  y_train = mlb.fit_transform(y_train_text)
  X_dev = df_dev.tweet.apply(clean_text)
  y_dev_text = df_dev[attribute].apply(lambda x: x.split('_'))
  y_dev = mlb.fit_transform(y_dev_text)
  X_test = df_test.tweet.apply(clean_text)
  y_test_text = df_test[attribute].apply(lambda x: x.split('_'))
  y_test = mlb.fit_transform(y_test_text)
  count_vect = CountVectorizer()
  X_train_counts = count_vect.fit_transform(X_train)
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  Y = mlb.fit_transform(y_train_text)
  classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', ClassifierChain(DummyClassifier()))])

  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  print('Accuracy %s' % accuracy_score(y_pred, y_test))
  print('Test macro F1 score is %s' % f1_score(y_test, y_pred, average='macro'))
  print('Test micro F1 score is %s' % f1_score(y_test, y_pred, average='micro'))
  

#majority voting for non mumtilabel tasks namely: target, group and directness
def majority_voting_non_multilabel_classification(train_filename, dev_filename, test_filename, attribute):
  my_labels=LABELS[attribute]
  df_train = pd.read_csv(train_filename)
  df_dev = pd.read_csv(dev_filename)
  df_test = pd.read_csv(test_filename)
  X_train = df_train.tweet.apply(clean_text)
  y_train = df_train[attribute]
  X_dev = df_dev.tweet.apply(clean_text)
  y_dev = df_dev[attribute]
  X_test = df_test.tweet.apply(clean_text)
  y_test = df_test[attribute]
  count_vect = CountVectorizer()
  X_train_counts = count_vect.fit_transform(X_train)
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  dummy = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', DummyClassifier()),
               ])
  dummy.fit(X_train, y_train)
  y_pred = dummy.predict(X_test)
  print('Accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=my_labels,labels=my_labels))
  print('Test macro F1 score is %s' % f1_score(y_test, y_pred, average='macro'))
  print('Test micro F1 score is %s' % f1_score(y_test, y_pred, average='micro'))
  

#logistic regression for non mumtilabel tasks namely: target, group and directness
def lr_non_multilabel_classification(train_filename, dev_filename, test_filename, attribute):
  my_labels=LABELS[attribute]
  df_train = pd.read_csv(train_filename)
  df_dev = pd.read_csv(dev_filename)
  df_test = pd.read_csv(test_filename)
  X_train = df_train.tweet.apply(clean_text)
  y_train = df_train[attribute]
  X_dev = df_dev.tweet.apply(clean_text)
  y_dev = df_dev[attribute]
  X_test = df_test.tweet.apply(clean_text)
  y_test = df_test[attribute]
  count_vect = CountVectorizer()
  X_train_counts = count_vect.fit_transform(X_train)
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
  logreg.fit(X_train, y_train)
  y_pred = logreg.predict(X_test)
  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print('Test macro F1 score is %s' % f1_score(y_test, y_pred, average='macro'))
  print('Test micro F1 score is %s' % f1_score(y_test, y_pred, average='micro'))


def run_majority_voting(train_filename, dev_filename, test_filename, attribute):
  #multilabel tasks
  if(attribute=='sentiment' or attribute=='annotator_sentiment'):
    return majority_voting_multilabel_classification(train_filename, dev_filename, test_filename, attribute)
  #non mutilabel tasks
  elif(attribute=='target' or attribute =='group' or attribute=='directness'):
    return majority_voting_non_multilabel_classification(train_filename, dev_filename, test_filename, attribute)

def run_logistic_regression(train_filename, dev_filename, test_filename, attribute):
  #multilabel tasks
  if(attribute=='sentiment' or attribute=='annotator_sentiment'):
    return lr_multilabel_classification(train_filename, dev_filename, test_filename, attribute)
  #non mutilabel tasks
  elif(attribute=='target' or attribute =='group' or attribute=='directness'):
    return lr_non_multilabel_classification(train_filename, dev_filename, test_filename, attribute)
