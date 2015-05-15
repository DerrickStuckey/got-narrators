__author__ = 'dstuckey'

import parse_chapters

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC

from sklearn.metrics import classification_report

import numpy as np
from random import shuffle

book_filenames = ['raw_text/#1A Game of Thrones.txt',
                  'raw_text/#2A Clash of Kings.txt',
                  'raw_text/#3A Storm of Swords.txt',
                  'raw_text/#4A Feast for Crows.txt',
                  'raw_text/#5A Dance With Dragons cleaned.txt']

chapters = []

for book_filename in book_filenames:
    cur_chapters = parse_chapters.parse_chapters(book_filename)
    chapters = chapters + cur_chapters

#print len(chapters)
shuffle(chapters)

arya_chapters = [x for x in chapters if x.name == "ARYA"]
#print "arya chapters: ", len(arya_chapters)

jon_chapters = [x for x in chapters if x.name == "JON"]
# print "jon chapters: ", len(jon_chapters)

arya_y = [int(x.name == "ARYA") for x in chapters]
# print arya_y

jon_y = [int(x.name == "JON") for x in chapters]
# print jon_y

chapter_contents = [x.content for x in chapters]

#remove non-ascii characters from contents
#chapter_contents = [x.encode('ascii', 'ignore') for x in chapter_contents]

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

trainsize = 200
testsize = 311 - trainsize

# test on Arya
arya_y_train = arya_y[0:trainsize-1]
arya_y_test = arya_y[trainsize:310]

contents_train = chapter_contents[0:trainsize-1]
contents_test = chapter_contents[trainsize:310]

preds, selected_features = classify(contents_train, arya_y_train, contents_test)
print preds
print arya_y_test

print(classification_report(arya_y_test, preds))