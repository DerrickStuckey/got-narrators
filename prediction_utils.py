__author__ = 'dstuckey'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

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

# splits data into training and test set for k-fold cross-validation
# trial_num should range from 0 to crossval_k-1
def crossval_split(data, crossval_k, trial_num):
    chunk_size = int(len(data)/crossval_k)

    start_idx = trial_num*chunk_size
    stop_idx = (trial_num+1)*chunk_size

    test_chapters = data[start_idx:stop_idx]
    train_chapters_1 = data[:start_idx]
    train_chapters_2 = data[stop_idx:]

    train_chapters = train_chapters_1 + train_chapters_2
    return train_chapters, test_chapters

def extract_y(chapters, narrator):
    return [int(x.name == narrator) for x in chapters]

def svm_classify(chapter_contents_train, y_train, chapter_contents_test,k=20):
    clf = SVC(class_weight='auto',probability=False,kernel="linear",tol=1e-3)
    return classify(clf, chapter_contents_train, y_train, chapter_contents_test,k)

def nb_classify(chapter_contents_train, y_train, chapter_contents_test,equal_priors=True,k=20):
    # if specified, assume equal priors for classes '1' and '0'
    if equal_priors:
        priors = [0.5,0.5]
    else:
        # otherwise, calculate priors using training set frequencies
        prior_1 = float(sum(y_train)) / len(y_train)
        prior_0 = float(len(y_train) - sum(y_train)) / len(y_train)
        # print "prior 0: ", prior_0
        # print "prior 1: ", prior_1
        priors = [prior_0, prior_1]

    clf = MultinomialNB(class_prior=priors)
    return classify(clf, chapter_contents_train, y_train, chapter_contents_test,k)

def classify(clf, chapter_contents_train, y_train, chapter_contents_test,k=20):
    # convert the training data text to features using TF-IDF vectorization
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
    X_train = vectorizer.fit_transform(chapter_contents_train)
    # X_train_array = X_train.toarray()
    # print "tfidf vector length: ", len(X_train_array) #dbg
    # print "X_train_array[0] length: ", len(X_train_array[0]) #dbg

    # use only the best k features according to chi-sq selection
    ch2 = SelectKBest(chi2, k=k)
    X_train = ch2.fit_transform(X_train, y_train)

    # determine the actual features used after best-k selection
    feature_names = np.asarray(vectorizer.get_feature_names())
    chisq_mask = ch2.get_support()
    features_masks = zip(feature_names,chisq_mask)
    selected_features = [z[0] for z in features_masks if z[1]]

    # train the classifier
    clf.fit(X_train, y_train)

    # convert the test data text into features using the same vectorizer as for training
    X_test = vectorizer.transform(chapter_contents_test)
    X_test = ch2.transform(X_test)

    # obtain binary class predictions for the test set
    preds = clf.predict(X_test)
    return preds, selected_features, clf
