__author__ = 'dstuckey'

import parse_chapters
import prediction_utils as pu

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from random import shuffle, seed


book_filenames = ['raw_text/#1A Game of Thrones.txt',
                  'raw_text/#2A Clash of Kings.txt',
                  'raw_text/#3A Storm of Swords.txt',
                  'raw_text/#4A Feast for Crows.txt',
                  'raw_text/#5A Dance With Dragons cleaned.txt']

chapters = []

# narrators = ['ARYA','JON','SANSA','TYRION','DAENERYS','CERSEI','JAIME']
narrators = ['ARYA','JON','SANSA','TYRION','DAENERYS','CERSEI','JAIME','BRAN','DAVOS','THEON','SAMWELL','EDDARD','BRIENNE']

crossval_k = 10

# trial_num should range from 0 to crossval_k-1
def crossval_split(data, crossval_k, trial_num):
    chunk_size = int(len(data)/crossval_k)

    start_idx = trial_num*chunk_size
    stop_idx = (trial_num+1)*chunk_size
    # print start_idx
    # print stop_idx
    test_chapters = data[start_idx:stop_idx]
    train_chapters_1 = data[:start_idx]
    train_chapters_2 = data[stop_idx:]
    #dbg
    # print "train_1 len: ", len(train_chapters_1)
    # print "train_2 len: ", len(train_chapters_2)
    train_chapters = train_chapters_1 + train_chapters_2
    return train_chapters, test_chapters

#test
# chapters = [1,2,3,4,5,6,7,8,9,10]
# train, test = crossval_split(chapters, 5, 1)
# exit()

for book_filename in book_filenames:
    cur_chapters = parse_chapters.parse_chapters(book_filename)
    chapters = chapters + cur_chapters

# randomize chapters order
shuffle(chapters)

#narrator names are keys
accuracy_vals = {}
precision_vals = {}
recall_vals = {}

for narrator in narrators:
    accuracy_vals[narrator] = []
    precision_vals[narrator] = []
    recall_vals[narrator] = []

for trial_num in range(0,crossval_k,1):
    train_chapters, test_chapters = crossval_split(chapters, crossval_k, trial_num)
    contents_train = [x.content for x in train_chapters]
    contents_test = [x.content for x in test_chapters]

    # run for several narrators:
    for narrator in narrators:
        print "\n\n"
        print "Narrator: ", narrator
        y_train = pu.extract_y(train_chapters, narrator)
        y_test = pu.extract_y(test_chapters, narrator)

        # preds, selected_features, clf = pu.svm_classify(contents_train, y_train, contents_test,k=2000)
        # coefs = clf.coef_.toarray()

        preds, selected_features, clf = pu.nb_classify(contents_train, y_train, contents_test,k=2000)
        # coefs = clf.feature_log_prob_

        # print "coefs len: ", len(coefs)
        # print "coefs[0] len: ", len(coefs[0])

        # class_1_coefs = coefs[-1]

        print(classification_report(y_test, preds))

        # print "selected features: "
        # # print selected_features
        # for (feature,coef) in zip(selected_features,class_1_coefs):
        #     print str(feature), ": ", coef

        # weighted_features = zip(selected_features,class_1_coefs)
        # sorted_features = sorted(weighted_features, key=lambda x: x[1], reverse=True)
        # print "top positive features: "
        # print sorted_features[:10]
        #
        # print "top negative features: "
        # print [x for x in sorted_features if x[1] < 0][-10:]

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, pos_label=1)
        recall = recall_score(y_test, preds, pos_label=1)

        accuracy_vals[narrator].append(accuracy)
        precision_vals[narrator].append(precision)
        recall_vals[narrator].append(recall)

def mean(x):
    return float(sum(x))/len(x)

for narrator in narrators:
    print narrator
    print "avg accuracy: ", mean(accuracy_vals[narrator])
    print "avg precision: ", mean(precision_vals[narrator])
    print "avg recall: ", mean(recall_vals[narrator])