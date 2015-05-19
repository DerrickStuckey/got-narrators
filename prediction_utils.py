__author__ = 'dstuckey'

import parse_chapters

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report

import numpy as np
from random import shuffle, seed

def partition(chapters, test_prop = 0.5, randomize=True):
    trainSize = int(len(chapters)*(1-test_prop))
    if (not randomize):
        seed(11235)
    shuffle(chapters)
    train_chapters = chapters[:trainSize]
    test_chapters = chapters[trainSize:]
    return train_chapters, test_chapters

def extract_y(chapters, narrator):
    return [int(x.name == narrator) for x in chapters]

def svm_classify(chapter_contents_train, y_train, chapter_contents_test,k=20):
    clf = SVC(class_weight='auto',probability=False,kernel="linear",tol=1e-3)
    return classify(clf, chapter_contents_train, y_train, chapter_contents_test,k)

def nb_classify(chapter_contents_train, y_train, chapter_contents_test,k=20):
    prior_1 = float(sum(y_train)) / len(y_train)
    prior_0 = float(len(y_train) - sum(y_train)) / len(y_train)
    # print "prior 0: ", prior_0
    # print "prior 1: ", prior_1
    # clf = MultinomialNB(class_prior=[prior_0,prior_1])
    clf = MultinomialNB(class_prior=[0.5,0.5]) #with actual priors, predicts 0 for all points
    return classify(clf, chapter_contents_train, y_train, chapter_contents_test,k)

def classify(clf, chapter_contents_train, y_train, chapter_contents_test,k=20):
    # build a classifier
    # k = 20

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
    X_train = vectorizer.fit_transform(chapter_contents_train)

    ch2 = SelectKBest(chi2, k=k)
    X_train = ch2.fit_transform(X_train, y_train)

    # get features used
    feature_names = np.asarray(vectorizer.get_feature_names())
    chisq_mask = ch2.get_support()
    features_masks = zip(feature_names,chisq_mask)
    selected_features = [z[0] for z in features_masks if z[1]]

    clf.fit(X_train, y_train)

    X_test = vectorizer.transform(chapter_contents_test)
    X_test = ch2.transform(X_test)

    preds = clf.predict(X_test)
    return preds, selected_features, clf
