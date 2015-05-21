__author__ = 'dstuckey'

import parse_chapters
import prediction_utils as pu
import csv

book_filenames = ['raw_text/#1A Game of Thrones.txt',
                  'raw_text/#2A Clash of Kings.txt',
                  'raw_text/#3A Storm of Swords.txt',
                  'raw_text/#4A Feast for Crows.txt',
                  'raw_text/#5A Dance With Dragons cleaned.txt']

chapters = []

features_dir = "./results/features"

def write_narrator_features(narrator, weighted_features):
    output_filename = features_dir + "/" + narrator + "_features.csv"
    with open(output_filename, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['feature','weight'])
        for weighted_feature in weighted_features:
            writer.writerow(weighted_feature)

# narrators = ['ARYA','JON','SANSA','TYRION','DAENERYS','CERSEI','JAIME']
narrators = ['ARYA','JON','SANSA','TYRION','DAENERYS','CERSEI','JAIME','BRAN','DAVOS','THEON','SAMWELL','EDDARD','BRIENNE']

for book_filename in book_filenames:
    cur_chapters = parse_chapters.parse_chapters(book_filename)
    chapters = chapters + cur_chapters

# since we are only interested in the features used and their weights after training,
# we use the full dataset as training data, and a throwaway test set
train_chapters, test_chapters = pu.partition(chapters, test_prop=0, randomize=True)
contents_train = [x.content for x in train_chapters]
contents_test = ["these","are","not","real","chapters"]

# run for all narrators:
for narrator in narrators:
    y_train = pu.extract_y(train_chapters, narrator)
    y_test = pu.extract_y(test_chapters, narrator)

    # SVM classifier
    preds, selected_features, clf = pu.svm_classify(contents_train, y_train, contents_test, k="all")
    coefs = clf.coef_.toarray()

    # Naive Bayes classifier
    # preds, selected_features, clf = pu.nb_classify(contents_train, y_train, contents_test,k="all")
    # coefs = clf.feature_log_prob_

    # print "coefs len: ", len(coefs)
    # print "coefs[0] len: ", len(coefs[0])

    class_1_coefs = coefs[-1]

    print "\n\n"
    print "Narrator: ", narrator

    weighted_features = zip(selected_features,class_1_coefs)
    sorted_features = sorted(weighted_features, key=lambda x: x[1], reverse=True)
    print "highest weighted features: "
    print sorted_features[:10]

    print "lowest weighted features: "
    print [x for x in sorted_features if x[1] < 0][-10:]

    write_narrator_features(narrator, sorted_features)
