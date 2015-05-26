__author__ = 'dstuckey'

import parse_chapters
import prediction_utils as pu

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from random import shuffle, seed
import csv

output_dir = "./results"

# saves score summary to CSV
def write_kfold_scores(classifier_type, crossval_k, narrator_scores):
    output_filename = output_dir + "/" + classifier_type + "_" + str(crossval_k) + "_fold_scores.csv"
    with open(output_filename, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['pov_character','accuracy','precision','recall'])
        for (narrator,accuracy,precision,recall) in narrator_scores:
            writer.writerow([narrator,accuracy,precision,recall])

book_filenames = ['raw_text/#1A Game of Thrones.txt',
                  'raw_text/#2A Clash of Kings.txt',
                  'raw_text/#3A Storm of Swords.txt',
                  'raw_text/#4A Feast for Crows.txt',
                  'raw_text/#5A Dance With Dragons cleaned.txt']

chapters = []

# narrators = ['ARYA','JON','SANSA','TYRION','DAENERYS','CERSEI','JAIME']

#larger set of narrators:
narrators = ['ARYA','JON','SANSA','TYRION','DAENERYS','CERSEI','JAIME','BRAN','DAVOS','SAMWELL','EDDARD','BRIENNE']

crossval_k = 3

# parse all books into chapters
for book_filename in book_filenames:
    cur_chapters = parse_chapters.parse_chapters(book_filename)
    chapters = chapters + cur_chapters

# randomize chapters order (using set seed to compare over multiple runs)
seed(11235)
shuffle(chapters)

#dictionaries holding a list for each narrator of performance metrics over trials
#narrator names are keys
accuracy_vals = {}
precision_vals = {}
recall_vals = {}

#initialize dictionary lists for each narrator
for narrator in narrators:
    accuracy_vals[narrator] = []
    precision_vals[narrator] = []
    recall_vals[narrator] = []

#calculates the mean of a list of numerical values
def mean(x):
    return float(sum(x))/len(x)

# run for both SVM and Naive Bayes classifiers
for classifier_type in ['svm', 'nb']:
    # train and test classifier for each train/test split over k folds
    for trial_num in range(0,crossval_k,1):
        train_chapters, test_chapters = pu.crossval_split(chapters, crossval_k, trial_num)
        contents_train = [x.content for x in train_chapters]
        contents_test = [x.content for x in test_chapters]

        # run for several narrators:
        for narrator in narrators:
            print "\n\n"
            print "Narrator: ", narrator
            y_train = pu.extract_y(train_chapters, narrator)
            y_test = pu.extract_y(test_chapters, narrator)

            if classifier_type == "svm":
                # SVM classifier
                preds, selected_features, clf = pu.svm_classify(contents_train, y_train, contents_test,k=2000)
            elif classifier_type == "nb":
                # Naive Bayes classifier
                preds, selected_features, clf = pu.nb_classify(contents_train, y_train, contents_test,k=2000)

            print(classification_report(y_test, preds))

            accuracy = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, pos_label=1)
            recall = recall_score(y_test, preds, pos_label=1)

            accuracy_vals[narrator].append(accuracy)
            precision_vals[narrator].append(precision)
            recall_vals[narrator].append(recall)

    for narrator in narrators:
        print narrator
        print "avg accuracy: ", mean(accuracy_vals[narrator])
        print "avg precision: ", mean(precision_vals[narrator])
        print "avg recall: ", mean(recall_vals[narrator])

    #calculate summary scores for each narrator
    narrator_scores = []
    for narrator in narrators:
        narrator_scores.append([narrator,mean(accuracy_vals[narrator]),mean(precision_vals[narrator]),mean(recall_vals[narrator])])

    #save summary to csv
    write_kfold_scores(classifier_type,crossval_k,narrator_scores)