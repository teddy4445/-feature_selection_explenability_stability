# library imports
import os
import pandas as pd

# project imports
from meta_data_table_generator import MetaDataTableGenerator


class StabilityTestsAnalyzer:
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
        StabilityTestsAnalyzer.first_plots(mdf=mdf,
                                           results_folder_path=results_folder_path)
        StabilityTestsAnalyzer.hierarchical_clustering(mdf=mdf,
                                                       results_folder_path=results_folder_path)
        StabilityTestsAnalyzer.prepare_dataset_for_classificator_learning(mdf=mdf,
                                                                          results_folder_path=results_folder_path)

    @staticmethod
    def first_plots(mdf,
                    results_folder_path: str):

        """
        This method plots several analysis on the meta-dataset.

        1. For each row in the MDF:
           surface plot: x-axis the FS algorithm (with names as ticks),
                         y-axis is the stability test (with names as ticks)
                         z-axis the test score for the FS sets distance metric
           *  The color map is "warmcold"
           ** The z-values needs to be normalized between 0 and 1

        2. For each row in the MDF:
           bar plot: x-axis the FS algorithm (with names as ticks) AND bar-groups according to the data stability test
                     y-axis the test score the FS sets distance metric.

        3. For each row in the MDF:
           error bar plot: x-axis the data stability test (with names as ticks)
                     y-axis the mean test score of the FS sets distance metric across the FS algorithms
                            and with an error bar corresponding to one STD.

        4. For all rows in the MDF:
           error bar plot: x-axis the FS algorithm (with names as ticks) AND bar-groups according to the data stability test
                           y-axis the mean test score the FS sets distance metric across the rows in the MDF
                                  and with an error bar corresponding to one STD.
           ** The z-values needs to be normalized between 0 and 1
        """
        pass

    @staticmethod
    def hierarchical_clustering(mdf,
                                results_folder_path: str):
        """
        This method responsible to show which datasets are similar each one on the 'x' features
        and than similar in the results for the same FS, stability test, and FS set distance metric.

        This method will generate a hierarchical clustering plot (for example: https://prnt.sc/1o4w2cq) where the samples
        are the 'x' features of each row in the MDF and a 'y' feature which is one stability-related column.
        * This process repeats for every 'y' feature which is one stability-related column in the MDF.
        """
        pass

    @staticmethod
    def prepare_dataset_for_classificator_learning(mdf,
                                                   results_folder_path: str):
        """
        This method responsible to generate a new table from the MDF for later classification problem.
        The new table is constructed from the 'x' columns of the MDF (repeating for multiple lines)
        with a column that indicates the FS algorithm used (according to some indexing of the FS algorithms)
        and a column that indicates if for this FS and dataset the data-stability (0), feature-stability (1),
        or Lyapunov-stability (2) test has the lowest value.

        The result is saved to a csv file.
        """
        pass
