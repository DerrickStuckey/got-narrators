# got-narrators

Code for parsing the Game of Thrones book series into a set of chapters with the associated POV (point of view) character for each chapter, and classification of chapter POV character by chapter contents.

## File structure

**parse_chapters.py**
Module defining code to parse a given book into chapters with associated POV character per chapter.

**prediction_utils.py**
Module containing scikit-learn implementation of SVM and Naive Bayes text classifiers, using TF-IDF text feature extraction, and some general utility functions such as k-fold cross-validation splitting.

**run_predictions_crossval.py**
Executable script which runs parser for first 5 books of GoT series, trains SVM and Naive Bayes classifiers to predict POV character by chapter, tests with k-fold cross-validation, and exports a performance summary for each classifier to CSV files in the 'results' directory.

**top_features.py**
Executable script which runs parser for first 5 books of GoT series, trains an SVM classifier on the full corpus, and exports trained feature weights to a CSV file in 'results/features'.

**raw_text**
This directory holds the actual text of each of the first 5 GoT books.  The texts are not included in this repository for copyright reasons.

**results**
This directory contains CSV output for classifier performance.

**results/features**
This directory contains CSV output for feature weights used to determine character-word associations.
