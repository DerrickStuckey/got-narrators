__author__ = 'dstuckey'

import parse_chapters

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC

from sklearn.metrics import classification_report

import numpy as np
from random import shuffle

def partition(chapters, test_prop = 0.5):
    trainSize = int(len(chapters)*(1-test_prop))
    shuffle(chapters)
    train_chapters = chapters[:trainSize]
    test_chapters = chapters[trainSize:]
    return train_chapters, test_chapters

def extract_y(chapters, narrator):
    return [int(x.name == narrator) for x in chapters]

def classify(chapter_contents_train, y_train, chapter_contents_test):
    # build a classifier
    k = 20

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
    X_train = vectorizer.fit_transform(chapter_contents_train)

    ch2 = SelectKBest(chi2, k=k)
    X_train = ch2.fit_transform(X_train, y_train)

    clf = SVC(class_weight='auto',probability=True,kernel="linear",tol=1e-3)

    # get features used
    feature_names = np.asarray(vectorizer.get_feature_names())
    chisq_mask = ch2.get_support()
    features_masks = zip(feature_names,chisq_mask)
    selected_features = [z[0] for z in features_masks if z[1]]
    # print "selected features: ", selected_features

    clf.fit(X_train, y_train)

    X_test = vectorizer.transform(chapter_contents_test)
    X_test = ch2.transform(X_test)

    preds = clf.predict(X_test)
    return preds, selected_features
