# library imports
import os
import pandas as pd

# project imports
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
            results_folder_path: str):
        """
        # TODO: add here later
        """
        # read data from meta table file
        mdf = pd.read_csv(os.path.join(data_folder_path, MetaDataTableGenerator.FILE_NAME))

        ExplainablePerformancePipelineAnalyzer.divide_mdf_by_metric(mdf=mdf,
                                                                    results_folder_path=results_folder_path)
        ExplainablePerformancePipelineAnalyzer.analyze_2(mdf=mdf,
                                                         results_folder_path=results_folder_path)
        ExplainablePerformancePipelineAnalyzer.analyze_3(mdf=mdf,
                                                         results_folder_path=results_folder_path)

    @staticmethod
    def divide_mdf_by_metric(mdf,
                             results_folder_path: str):
        """
        This method drops all irrelevant columns in the MDF and divide the
        'x' columns and the 'y' columns that starts with 'expandability_' in the dataframe.

        Afterward, it divide the dataframe to the 'x' columns and the 'y' columns with
        the results of a singular metric at a time. Each such dataframe is saved with name that
        contains the name of the metric in the 'results_folder_path' folder.

        Note: if sub-MDF file is already in place, skip it re-generation

        :return: a dictionary: keys are metric names, and the values are the obtained dataframes.
        """
        pass

    @staticmethod
    def index_all_classes_of_pairs_of_fs(mdf):
        """
        Assumption: the MDF is obtained from the 'ExplainablePerformancePipelineAnalyzer.divide_mdf_by_metric' method.
        This method add a column to the mdf with the index of the best 'y' column (starting with 0).
        """
        pass

    @staticmethod
    def train_and_test_local_classifier(mdf,
                                        results_folder_path: str):
        """
        Assumption: the MDF is obtained from the 'ExplainablePerformancePipelineAnalyzer.divide_mdf_by_metric' and
        'ExplainablePerformancePipelineAnalyzer.index_all_classes_of_pairs_of_fs' methods.
        This method train a list of classifiers, test them based on the "accuracy" metric and
        save the best model and its test results in files in the 'results_folder_path' folder.
        """
        # NOTE: you can use the 'Lazy Predict' library for this method.
        # https://lazypredict.readthedocs.io/en/latest/
        # https://towardsdatascience.com/lazy-predict-fit-and-evaluate-all-the-models-from-scikit-learn-with-a-single-line-of-code-7fe510c7281
        # TODO: this metric for the tests
        ExplainablePerformanceMetrics.accuracy(y_true=[],
                                               y_pred=[])

