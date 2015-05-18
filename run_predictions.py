__author__ = 'dstuckey'

import parse_chapters
import prediction_utils as pu

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

narrators = ['ARYA','JON','SANSA','TYRION']

for book_filename in book_filenames:
    cur_chapters = parse_chapters.parse_chapters(book_filename)
    chapters = chapters + cur_chapters

train_chapters, test_chapters = pu.partition(chapters, test_prop=0.3)
contents_train = [x.content for x in train_chapters]
contents_test = [x.content for x in test_chapters]

# run for several narrators:
for narrator in narrators:
    print "\n\n"
    print "Narrator: ", narrator
    y_train = pu.extract_y(train_chapters, narrator)
    y_test = pu.extract_y(test_chapters, narrator)

    preds, selected_features = pu.classify(contents_train, y_train, contents_test)

    print(classification_report(y_test, preds))

    print "selected features: "
    for feature in selected_features:
        print str(feature)

