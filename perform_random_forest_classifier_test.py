# TODO: Add ability to classify more than 2 classes.
from __future__ import division
import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from argparse import ArgumentParser
import os.path
import random
from ntpath import basename


def calculate_accuracy(preds, labels):
    """
    This function accepts the predictions from the classifier along with the actual labels for each data point.
    The two lists will be compared to calculate the accuracy of the accuracy.
    The function will also return the index of the data points that have been wrongly classified.
    :param preds: List of predictions from the classifier
    :param labels: List of true labels for each data point
    :result acc / len(labels): % accuracy of the classifier
    :result mis_classified: List containing the indexes of wrongly classifier data points
    :result precision: The precision value of the classifier on the given data
    :result recall: The recall value of the classifier on the given data
    :result f1: The f1 value of the classifier on the given data
    """
    acc = 0
    mis_classified = list()
    for i in range(0, len(preds)):
        if preds[i] == labels[i]:
            acc += 1
        else:
            mis_classified.append(i)
            continue
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    return acc / len(labels), mis_classified, precision, recall, f1


def calculate_roc(labels, preds_scores):
    """
    The function will generate a dataframe of false-positive and true-positive values based on test data predictions
    :param labels:
    :param preds_scores:
    :return fpr_tpr: False-Posttive, true-positive dataframe
    """
    fpr, tpr, _ = metrics.roc_curve(labels, preds_scores)
    fpr_tpr = pandas.DataFrame(dict(fpr=fpr, tpr=tpr))
    return fpr_tpr


def set_columns_for_tpr_df(cols, i):
    """
    This function generates appropriate column names for the fpr_tpr dataframe
    :param cols: Columns of trp dataframe from 1 iteration
    :param i: Current iteration number
    :return cols_list: New column list for master fpr_tpr dataframe
    """
    cols_list = list()
    for c in cols:
        cols_list.append(c + "_" + str(i))
    return cols_list


def rfclassifier(traindata, trainlabels, all_test_data, all_test_labels, cores, trees):
    """
    This function will create a new instance of Random Forest Classifier (scikit-learn) with 100 estimator trees.
    The function will fit the classifier on the provided training data and then test the classifier on the provided test
    data.
    Predictions on the training and test data are passed to the calculate_accuracy function to determine the % accuracy
    of the classifier and also find the mis-classified data points.
    The function returns the training and testing predictions from the classifier, the train and test data accuracies,
    importance of each data feature provided, out-of-bag error metric for classifier,
    confusion matrix ofr test data and a list of wringly classifier data points from the test data.
    :param traindata: A matrix of training data
    :param trainlabels: Array of true data labels for training data
    :param all_test_data: A dictionary of all test ndarrays as values and the file names as keys
    :param all_test_labels: A dictionary of all test class labels as values and the file names as keys
    :param cores: Number of cores to use for random forest classifier.
    :param trees: Number of decision tress to use
    :result oob_error: OOB error value on the training data
    :result imps: % importance of each variable in the classification
    :result test_acc: Dictionary of testing data accuracies, with accuracies as values and file names as keys
    :result test_precision: Dictionary of testing data precision, with precision as values and file names as keys
    :result test_recall: Dictionary of testing data recall value, with recall values as values and file names as keys
    :result test_f1: Dictionary of testing data f1 values, with f1 values as values and file names as keys
    :result test_mis_classified: Dictionary of mis-classified testing data, with a list of mis-classified data points
    as values and file names as keys
    :result test_fpr_tpr: Dictionary of fpr_tpr dataframe for testing data, with each dataframe as value and file names
    as the key
    """
    # Set-up classifier
    clf = RandomForestClassifier(n_estimators=trees, oob_score=True, n_jobs=cores)
    # Fit classifier on training data
    clf.fit(traindata, trainlabels)
    # Calculate out-of-bag error on training data
    oob_error = 1 - clf.oob_score_

    # Run prediction on test data
    test_preds = dict()
    test_acc = dict()
    test_mis_classified = dict()
    test_f1 = dict()
    test_recall = dict()
    test_precision = dict()
    test_fpr_tpr = dict()

    for k in all_test_data.keys():
        test_preds[k] = clf.predict(all_test_data[k])
        testing_preds_scores = clf.predict_proba(all_test_data[k][0])[:, 1]
        # Calculate accuracy of classifier on test data
        test_acc[k], test_mis_classified[k], test_precision[k], test_recall[k], test_f1[k] = \
            calculate_accuracy(test_preds[k], all_test_labels[k])
        test_fpr_tpr[k] = calculate_roc(all_test_data[k], testing_preds_scores)

    # Calculate importance of each feature
    imps = clf.feature_importances_ * 100

    return oob_error, imps, test_acc, test_precision, test_recall, test_f1, test_mis_classified, test_fpr_tpr


def organize_data(dataf, num, label1, label2):
    """
    This function will organize a dataframe to be compatible with the format required by the RFClassifier function
    The function accepts a pandas dataframe, an interger value to subset rows from each class and labels of the two
    classes to subset the dataframe on.
    The purpose of this function is to create a data matrix that contains a user-defined number of rows from each class.
    The function will return a data matrix that is compatible with the RFClassifier function. The data labels are
    separated into a separate array and returned along with the data matrix.
    :param dataf: Pandas DataFrame to subset the two classes from
    :param num: Integer value to subset a user-defined number of rows for each class
    :param label1: Label for first class. This label should be present in the 'Type' column of the input dataframe
    'dataf'
    :param label2: Label for the second class. This label should be present in the 'Type' column of the input dataframe
    'dataf'
    """
    # Check 'num' flag
    if num == 0:
        df = dataf
    else:
        # Split dataframe on specified number of rows.
        df = pandas.DataFrame()
        rows = random.sample(dataf[dataf['Type'] == label1].index, int(num))
        df = pandas.concat((df, dataf[dataf['Type'] == label1].loc[rows]))
        rows = random.sample(dataf[dataf['Type'] == label2].index, int(num))
        df = pandas.concat((df, dataf[dataf['Type'] == label2].loc[rows]))
    # Binarize labels.
    df = df.replace(label1, 0)
    df = df.replace(label2, 1)
    # Convert dataframe and labels into a numpy matrix
    lbls_ndarray = df['Type'].as_matrix()
    df_ndarray = df.drop('Type', 1).as_matrix()
    return list(df.index), df_ndarray, lbls_ndarray


def determine_number_of_training_data_points(df, label1, label2, training_perc):
    """
    This function will determine the number of data points in training and testing data sets. The split is decided based
    on the fraction provided by the user. eg: training_perc == 60, 60 % of input data will be used as training data.
    Remaining 40 % will be used as testing data.
    :param df: Input dataframe
    :param label1: Class 1
    :param label2: Class 2
    :param training_perc: Fraction of total data to be used as training data
    :return: Number of data points that will used as training data.
    """
    # If there are unequal number of data points for the 2 classes, the class with lesser data points is used as
    # reference to determine the number of points in training set.
    if len(df[df['Type'] == label1]) < len(df[df['Type'] == label2]):
        if training_perc < 1:
            train_num_div = training_perc * len(df[df['Type'] == label1])
        else:
            train_num_div = (int(training_perc) / 100 * len(df[df['Type'] == label1]))
    else:
        if training_perc < 1:
            train_num_div = training_perc * len(df[df['Type'] == label2])
        else:
            train_num_div = (int(training_perc) / 100 * len(df[df['Type'] == label2]))
    return train_num_div


def shuffle_class_labels(df):
    """
    This function will shuffle the class label for each data point. Assumes that the labels are in the column 'Type'
    :param df: Dataframe to perform shuffle on
    :return df_shuf: Dataframe with shuffled labels
    """
    df_shuf = df.drop('Type', 1)
    df_shuf['Type'] = numpy.random.permutation(df['Type'])
    return df_shuf


def check_and_read_input_df(locs):
    """
    This function will check if a given file exists and also checks if the file shows the correct format as required to
    run the classifier
    :param locs: List of locations of all given files. These include the input dataframe and any other file to test the
    classifier on
    :return d: Returns a dictionary, with the file names as keys and a flag as value. If the file passes all
    requirements, no flag is given, instead the dataframe is assigned as the value.
    """
    d = dict()
    for loc in locs:
        if os.path.isfile(loc):
            df = pandas.read_csv(loc, sep='\t', index_col=0)
            if 'Type' in df.columns:
                labels = set(df['Type'])
                if len(labels) != 2:
                    d[loc] = 1
                else:
                    d[loc] = df
            else:
                d['loc'] = 2
        else:
            d['loc'] = 0
    return d


def main():
    # Set up arguments for the script
    parser = ArgumentParser(prog='perform_random_forest_classifier_test',
                            description="Run Random Forest Classifier To Perform Classification Of Two-Class Data")
    parser.add_argument("input", help="Complete file name including the path to nput table that contains data points as"
                                      " rows and data features as columns. The table should contain a 'Type' column "
                                      "that lists the labels for each data point")
    parser.add_argument("output", help="Specify complete path to output directory")
    #    parser.add_argument("classes", nargs='+', help="Comma separated list of Classes within the data. eg: A,B")
    parser.add_argument("--training_perc", help="Percent value to divide the data into training and testing data.",
                        default=90)
    parser.add_argument("--iterations", help="Number of individual iterations to train and test the classifier.",
                        default=3)
    parser.add_argument("--shuffle_test", action="store_true",
                        help="Perform testing on data with shuffled class labels")
    parser.add_argument("--cores", help="Number of cores to use.", default=1)
    parser.add_argument("--trees", help="Number of decision trees to use in random forest", default=100)
    parser.add_argument("--add_test_data", default=numpy.nan, help="Additional test data sets if any. Comma separated "
                                                                   "list containing complete path and name of test "
                                                                   "dataframes. Dataframe should follow the same format"
                                                                   " as input dataframe")
    parser.add_argument("--predict_classes", default=numpy.nan, help="List of dataframes to predict classes for. "
                                                                     "Classes for this data are unknown. Classifier "
                                                                     "will be trained on input data to predict classes "
                                                                     "for this data")

    args = parser.parse_args()

    all_file_paths = [args.input]
    if args.add_test_data != numpy.nan:
        for i in args.add_test_data:
            all_file_paths.append(i)

    all_dfs = check_and_read_input_df(all_file_paths)
    for k in all_dfs.keys():
        if all_dfs[k] == 0:
            parser.error("%s not found ! Enter complete path and name of the file.") % k
        elif all_dfs[k] == 1:
            parser.error("Found more than 2 classes in %s ! This script currently works on 2-class data only.") % k
        elif all_df[k] == 2:
            parser.error("'Type' column not found in %s . Make sure dataframe contains a column called 'Type', with the"
                         " classes in it") % k
        else:

            continue

    all_labels = dict()
    for k in all_dfs.keys():
        all_labels[k] = list(set(all_dfs[k]['Type']))

    train_oob = pandas.DataFrame(index=range(0, int(args.iterations)), columns=['OOB_Error'])
    feature_importances = pandas.DataFrame(index=range(0, int(args.iterations)),
                                           columns=input_df.drop('Type', 1).columns)
    all_test_acc = dict()
    all_test_precision = dict()
    all_test_recall = dict()
    all_test_f1 = dict()
    all_test_mis_classified = dict()
    all_test_fpr_trp = dict()
    for i in range(0, int(args.iterations)):
        temp_dfs = dict()
        temp_labels = dict()
        train_num_div = determine_number_of_training_data_points(all_dfs[args.input], all_labels[args.input][0],
                                                                 all_labels[args.input][1], args.training_perc)
        # Divide input dataframe into training and testing dataframes
        inds, training_ndarray, training_labels_ndarray = organize_data(all_dfs[args.input], train_num_div,
                                                                        all_labels[args.input][0],
                                                                        all_labels[args.input][1])
        temp_testing_df = all_df[args.input].drop(inds, 0)
        inds, temp_dfs[args.input], temp_labels[args.input] = organize_data(temp_testing_df, 0,
                                                                            all_labels[args.input][0],
                                                                            all_labels[args.input][1])

        for k in all_dfs.keys():
            if k == args.input:
                continue
            else:
                inds, temp_dfs[k], temp_labels[k] = organize_data(all_dfs[k], 0,
                                                                  all_labels[args.input][0],
                                                                  all_labels[args.input][1])

        if args.shuffle_test:
            temp_testing_df_shuf = shuffle_class_labels(temp_testing_df)
            inds, temp_dfs['shuffle'], temp_labels['shuffle'] = organize_data(temp_testing_df_shuf, 0,
                                                                              all_labels[args.input][0],
                                                                              all_labels[args.input][1])

        # Send training and testing data to 'rfclassifier' function. Classifier will be built and tested in that
        # function
        train_oob.ix[i, 'OOB_Error'], feature_importances.loc[i], test_acc, test_precision, test_recall, test_f1,\
            test_mis_classified, test_fpr_tpr = rfclassifier(training_ndarray, training_labels_ndarray, temp_dfs,
                                                             temp_labels, args.cores, args.trees)

        for k in test_pred.keys():
            all_test_acc[k + '_' + str(i)] = test_acc
            all_test_precision[k + '_' + str(i)] = test_precision
            all_test_recall[k + '_' + str(i)] = test_recall
            all_test_f1[k + '_' + str(i)] = test_f1
            all_test_fpr_trp[k + '_' + str(i)] = test_fpr_tpr
            all_test_mis_classified[k + '_' + str(i)] = test_mis_classified

    # Compile accuracies across all iterations
    train_oob.to_csv(args.output + "/" + basename(args.input) + ".Training_Out_Of_Bag_Errors.txt", sep='\t')

    for k in test_acc.keys():
        metrics = pandas.DataFrame()
        # fpr_tpr = pandas.DataFrame()
        mis_classified = pandas.DataFrame()
        for j in range(0, args.iterations):
            metrics.loc['Accuracy', i] = all_test_acc[k + '_' + str(i)]
            metrics.loc['Precision', i] = all_test_precision[k + '_' + str(i)]
            metrics.loc['Recall', i] = all_test_recall[k + '_' + str(i)]
            metrics.loc['F1', i] = all_test_f1[k + '_' + str(i)]
            mis_classified[i] = all_test_mis_classified[k + '_' + str(i)]
            # TODO: Add fpr tpr output
        if 'shuffle' in k:
            metrics.to_csv(args.output + "/" + basename(args.input) +
                           ".Classifier_Metrics_On_Shuffled_Labels.txt", sep='\t')
        elif args.input in k:
            metrics.to_csv(args.output + "/" + basename(args.input) + ".Classifier_Metrics_On_Testing_Data.txt",
                           sep='\t')
            mis_classified.to_csv(args.output + "/" + basename(args.input) + ".Mis_Classified_Data.txt", sep='\t')
        else:
            metrics.to_csv(args.output + "/" + basename(k) + ".Classifier_Metrics.txt", sep='\t')
            mis_classified.to_csv(args.output + "/" + basename(k) + ".Mis_Classified_Data.txt", sep='\t')


if __name__ == '__main__':
    main()
