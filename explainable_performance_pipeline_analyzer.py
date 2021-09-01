# library imports
import json
import os
import pandas as pd
import numpy as np

# project imports
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from meta_data_table_generator import MetaDataTableGenerator
from explainable_performance_metrics import ExplainablePerformanceMetrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


class ExplainablePerformancePipelineAnalyzer:
    """
    A class that provides plots, CSV files with tables, and statistical outcomes of a meta-dataset of feature selection
    computed by the MetaDataTableGenerator.run class
    """

    def __init__(self):
        pass

    @staticmethod
    def run(data_folder_path: str,
            sub_tables_folder_path: str,
            results_folder_path: str):
        """
        # TODO: add here later
        """
        # read data from meta table file
        mdf = pd.read_csv(os.path.join(data_folder_path, MetaDataTableGenerator.FILE_NAME))

        ExplainablePerformancePipelineAnalyzer.divide_mdf_by_metric(mdf=mdf,
                                                                    sub_tables_folder_path=sub_tables_folder_path)
        ExplainablePerformancePipelineAnalyzer.index_all_classes_of_pairs_of_fs(
            sub_tables_folder_path=sub_tables_folder_path)
        ExplainablePerformancePipelineAnalyzer.train_and_test_local_classifier(
            sub_tables_folder_path=sub_tables_folder_path,
            results_folder_path=results_folder_path)

    @staticmethod
    def divide_mdf_by_metric(mdf,
                             sub_tables_folder_path: str):
        """
        This method drops all irrelevant columns in the MDF and divide the
        'x' columns and the 'y' columns that starts with 'expandability-' in the dataframe.

        Afterward, it divide the dataframe to the 'x' columns and the 'y' columns with
        the results of a singular metric at a time. Each such dataframe is saved with name that
        contains the name of the metric in the 'results_folder_path' folder.

        Note: if sub-MDF file is already in place, skip it re-generation

        :return: a dictionary: keys are metric names, and the values are the obtained dataframes.
        """
        # make sure we have target folder
        try:
            os.makedirs(sub_tables_folder_path)
        except:
            pass

        # create sub-table for every metric from MetaDataTableGenerator.METRICS
        for metric in MetaDataTableGenerator.METRICS:
            # create sub_mdf using the columns with x_ and with the relevant metric
            sub_mdf = mdf[[c for c in mdf.columns if
                           c.startswith("x-") or (c.startswith("expandability-") and metric in c)]]
            sub_mdf.to_csv(os.path.join(sub_tables_folder_path, "sub_table_for_{}.csv".format(metric)))

    @staticmethod
    def index_all_classes_of_pairs_of_fs(sub_tables_folder_path):
        # TODO: fix documentation
        """
        Assumption: the MDF is obtained from the 'ExplainablePerformancePipelineAnalyzer.divide_mdf_by_metric' method.
        This method add a column to the mdf with the index of the best 'y' column (starting with 0).
        """
        # read sub_mdf
        for filename in os.listdir(sub_tables_folder_path):
            sub_mdf = pd.read_csv(os.path.join(sub_tables_folder_path, filename)).drop('Unnamed: 0', axis=1)

            # convert column to integers
            mapper = {value: index for index, value in
                      enumerate(list(sub_mdf[[c for c in sub_mdf.columns if not c.startswith('x-')]].columns))}
            # save mapper for later usage
            with open('classification_results_path/mapper.json', 'w') as f:
                json.dump(mapper, f)

            # find column with maximal value
            sub_mdf['best'] = sub_mdf[[c for c in sub_mdf.columns if not c.startswith('x-')]].idxmax(axis="columns")

            sub_mdf['best'] = sub_mdf['best'].apply(lambda x: mapper[x])
            sub_mdf.drop([c for c in sub_mdf.columns if c.startswith('expandability-')], axis=1).to_csv(
                os.path.join(sub_tables_folder_path, filename))

    @staticmethod
    def train_and_test_local_classifier(sub_tables_folder_path: str,
                                        results_folder_path: str):

        """
        Assumption: the MDF is obtained from the 'ExplainablePerformancePipelineAnalyzer.divide_mdf_by_metric' and
        'ExplainablePerformancePipelineAnalyzer.index_all_classes_of_pairs_of_fs' methods.
        This method train a list of classifiers, test them based on the "accuracy" metric and
        save the best model and its test results in files in the 'results_folder_path' folder.
        """
        try:
            os.makedirs(results_folder_path)
        except:
            pass

        full_scores_list = []
        full_confusion_matrix_list = []
        for filename in os.listdir(sub_tables_folder_path):
            sub_mdf = pd.read_csv(os.path.join(sub_tables_folder_path, filename)).drop('Unnamed: 0', axis=1)
            X = sub_mdf[[c for c in sub_mdf.columns if c.startswith("x-")]]
            y = sub_mdf['best']

            # clean the data and fix anomalies
            X.replace(np.inf, np.nan, inplace=True)
            X.replace(-np.inf, np.nan, inplace=True)
            y.replace(np.inf, np.nan, inplace=True)
            y.replace(-np.inf, np.nan, inplace=True)

            cv_scores = []
            cv_predictions = []
            X = X.sample(frac=1, random_state=123)
            y = y[X.index]

            num_of_folds = 5
            for i in range(num_of_folds):
                test_size = int(X.shape[0] / num_of_folds)
                # save index of test set
                test_index = X.iloc[(i * test_size):((i + 1) * test_size), :].index

                # define X_test, y_test
                X_test = X.loc[test_index]
                y_test = y.loc[test_index]
                # define X_train, y_train
                X_train = X.loc[~X.index.isin(test_index)]
                y_train = y.loc[~y.index.isin(test_index)]

                # run classifiers
                clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True)
                scores, predictions = clf.fit(X_train, X_test, y_train, y_test)

                # set index of predictions to be test_index
                predictions.index = test_index

                cv_scores.append(scores)
                cv_predictions.append(predictions)

            # create final prediction by concatenating predictions
            predictions = pd.concat(cv_predictions, axis=0)
            predictions['truth'] = y[predictions.index]

            # convert back to class names
            with open('classification_results_path/mapper.json', 'r') as f:
                mapper = json.load(f)
            reverse_mapper = {value: "-".join(key.split('__')[-1].split('-')[1:]) for key, value in mapper.items()}
            predictions = predictions.applymap(lambda x: reverse_mapper[x])

            predictions.to_csv(os.path.join(results_folder_path, "predictions_for_" + filename))

            # create final scores by using calculation over predictions
            scores = pd.DataFrame(columns=['Accuracy', 'f1-score'])

            # run over all models and evaluate the performance of each of them
            confusion_matrix_per_metric = []
            for model_name in predictions.drop('truth', axis=1).columns:
                # create confusion matrix for the specific model
                y_true = pd.Series(predictions['truth'], name="Truth")
                y_pred = pd.Series(predictions[model_name], name="Predicted")
                df_confusion = pd.crosstab(y_true, y_pred).reindex(columns=list(reverse_mapper.values())).reindex(
                    list(reverse_mapper.values())).fillna(0)

                # TODO: do we want to save also the partial confusion matrix?

                confusion_matrix_per_metric.append(df_confusion)

                # calculate metrics for all predictions
                accuracy = accuracy_score(predictions['truth'], predictions[model_name], normalize=True)
                # b_accuracy = balanced_accuracy_score(predictions['truth'], predictions[model_name])
                f1 = f1_score(predictions['truth'], predictions[model_name], average="weighted")
                # update the scores dataframe with the relevant results
                scores.loc[model_name] = [accuracy, f1]

            # create total confusion matrix for the relevant metric using average
            total_confusion_matrix_for_current_metric = pd.concat([
                df_confusion.stack() for df_confusion in confusion_matrix_per_metric
            ], axis=1) \
                .mean(axis=1) \
                .unstack()

            total_confusion_matrix_for_current_metric.to_csv(
                os.path.join(results_folder_path, "total_confusion_matrix_for_" + filename))

            scores.sort_values(by='Accuracy', ascending=False).to_csv(
                os.path.join(results_folder_path, "scores_for_" + filename))

            # add scores of the current metric to the scores list
            full_scores_list.append(scores.sort_index())

        # create total scores (for all metrics) using average on scores and save to csv
        total_scores = pd.concat([
            scores_df.stack() for scores_df in full_scores_list
        ], axis=1) \
            .mean(axis=1) \
            .unstack() \
            .sort_values(by='Accuracy', ascending=False)

        # save to csv
        total_scores.to_csv(os.path.join(results_folder_path, "total_scores_using_average.csv"))

        ExplainablePerformanceMetrics.accuracy(y_true=[],
                                               y_pred=[])


if __name__ == '__main__':
    ExplainablePerformancePipelineAnalyzer.run(data_folder_path="meta_table_data",
                                               sub_tables_folder_path="sub_meta_tables",
                                               results_folder_path="classification_results_path")
