# library imports
import os
import pandas as pd
import numpy as np

# project imports
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from meta_data_table_generator import MetaDataTableGenerator
from explainable_performance_metrics import ExplainablePerformanceMetrics


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

            # find column with maximal value
            sub_mdf['best'] = sub_mdf[[c for c in sub_mdf.columns if not c.startswith('x-')]].idxmax(axis="columns")

            # convert column to integers
            mapper = {value: index for index, value in enumerate(list(sub_mdf.columns))}
            sub_mdf['best'] = sub_mdf['best'].apply(lambda x: mapper[x])
            sub_mdf.to_csv(os.path.join(sub_tables_folder_path, filename))

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

        for filename in os.listdir(sub_tables_folder_path):
            sub_mdf = pd.read_csv(os.path.join(sub_tables_folder_path, filename))
            X = sub_mdf[[c for c in sub_mdf.columns if c.startswith("x-")]]
            y = sub_mdf['best']

            # clean the data and fix anomalies
            X.replace(np.inf, 1e9, inplace=True)
            X.replace(-np.inf, -1e9, inplace=True)
            y.replace(np.inf, 1e9, inplace=True)
            y.replace(-np.inf, -1e9, inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True)
            scores, predictions = clf.fit(X_train, X_test, y_train, y_test)

            scores.to_csv(os.path.join(results_folder_path, "scores_for_" + filename))
            predictions.to_csv(os.path.join(results_folder_path, "predictions_for_" + filename))

        ExplainablePerformanceMetrics.accuracy(y_true=[],
                                               y_pred=[])


if __name__ == '__main__':
    ExplainablePerformancePipelineAnalyzer.run(data_folder_path="meta_table_data",
                                               sub_tables_folder_path="sub_meta_tables",
                                               results_folder_path="classification_results_path")
