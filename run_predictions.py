__author__ = 'dstuckey'

import parse_chapters
import prediction_utils as pu

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

book_filenames = ['raw_text/#1A Game of Thrones.txt',
                  'raw_text/#2A Clash of Kings.txt',
                  'raw_text/#3A Storm of Swords.txt',
                  'raw_text/#4A Feast for Crows.txt',
                  'raw_text/#5A Dance With Dragons cleaned.txt']

chapters = []

narrators = ['ARYA','JON','SANSA','TYRION','DAENERYS']

num_trials = 10

for book_filename in book_filenames:
    cur_chapters = parse_chapters.parse_chapters(book_filename)
    chapters = chapters + cur_chapters

train_chapters, test_chapters = pu.partition(chapters, test_prop=0.3, randomize=False)
contents_train = [x.content for x in train_chapters]
contents_test = [x.content for x in test_chapters]

# run for several narrators:
for narrator in narrators:
    print "\n\n"
    print "Narrator: ", narrator
    y_train = pu.extract_y(train_chapters, narrator)
    y_test = pu.extract_y(test_chapters, narrator)

    preds, selected_features, clf = pu.svm_classify(contents_train, y_train, contents_test,k=2000)
    # preds, selected_features, clf = pu.nb_classify(contents_train, y_train, contents_test,k=1000)

    coefs = clf.coef_.toarray()

    print "coefs len: ", len(coefs)
    print "coefs[0] len: ", len(coefs[0])

    class_1_coefs = coefs[0]

    print(classification_report(y_test, preds))

    # print "selected features: "
    # # print selected_features
    # for (feature,coef) in zip(selected_features,class_1_coefs):
    #     print str(feature), ": ", coef

    weighted_features = zip(selected_features,class_1_coefs)
    sorted_features = sorted(weighted_features, key=lambda x: x[1], reverse=True)
    print "top positive features: "
    print sorted_features[:10]

    print "top negative features: "
    print [x for x in sorted_features if x[1] < 0][-10:]
