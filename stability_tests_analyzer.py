# library imports
import os
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

# project imports
from meta_data_table_generator import MetaDataTableGenerator


class StabilityTestsAnalyzer:
    """
    A class that provides plots, CSV files with tables, and statistical outcomes of a meta-dataset of feature selection
    computed by the MetaDataTableGenerator.run class
    """

    # CONSTS #
    STABILITY_COL_START_NAME = "stability-"
    COL_SPLIT_CHAR = "-"
    # END - CONSTS #

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

        # prepare folder
        try:
            os.mkdir(results_folder_path)
        except:
            pass

        # run analysis in all its forms
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
        # first, find all the FS algorithms + stability tests
        fs_algorithms = {}
        stability_tests = {}
        column_names = list(mdf)
        stability_column_names = []
        for column_name in column_names:
            try:
                stability_metric, fs_filter, stability_test = StabilityTestsAnalyzer._get_column_signature(column_name=column_name)
                if fs_filter not in fs_algorithms:
                    fs_algorithms[fs_filter] = len(fs_algorithms.keys())
                if stability_test not in stability_tests:
                    stability_tests[stability_test] = len(stability_tests.keys())
                stability_column_names.append(column_name)
            except:
                pass
        # format the data as x,y,z for each row in the dataset
        for index, row in mdf.iterrows():
            x = list(range(len(list(fs_algorithms.keys()))))
            y = list(range(len(list(stability_tests.keys()))))
            # gather the data from the relevant columns
            z = [float(row[column_name]) for column_name in stability_column_names]
            # normalize the z-values to [0, 1]
            z_min = min(z)
            z_delta = max(z) - z_min
            y_size = len(y)
            x_size = len(x)
            z = np.asarray([round((z_value - z_min) / z_delta, 3) for z_value in z]).reshape(y_size, x_size)
            # make 2d grid
            x, y = np.meshgrid(x, y)
            # plot the surface.
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                                   figsize=(10, 10))
            ax.plot_wireframe(x,
                              y,
                              z,
                              rstride=1,
                              cstride=1,
                              color="black",
                              alpha=0.5,
                              cmap=cm.coolwarm)
            ax.scatter(x,
                       y,
                       z,
                       c=z,
                       s=80,
                       cmap=cm.coolwarm)
            # format the z-axis in a cooler way
            ax.set_zlim(0, 1)
            ax.zaxis.set_major_locator(LinearLocator(11))
            ax.zaxis.set_major_formatter('{x:.02f}')
            ax.set_yticks(list(range(len(list(stability_tests.keys())))))
            ax.set_yticklabels(list(stability_tests.keys()))
            ax.set_xticklabels([value.replace("_", " ") for value in list(fs_algorithms.keys())])
            ax.view_init(28, 30)
            # save the plot
            plt.savefig(os.path.join(results_folder_path, "{}.png".format(row["ds_name"])))
            # close for next plot
            plt.close()

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

    # HELPER #

    @staticmethod
    def _get_column_signature(column_name: str):
        """
        Helper function, gets the columns classification to the three objects
        """
        # if not stability column, just return error
        if StabilityTestsAnalyzer.STABILITY_COL_START_NAME not in column_name:
            raise Exception("Not a stability column")
        # break it
        column_name = column_name[len(StabilityTestsAnalyzer.STABILITY_COL_START_NAME):]
        return column_name.strip().split(StabilityTestsAnalyzer.COL_SPLIT_CHAR)

    # END - HELPER #


if __name__ == '__main__':
    StabilityTestsAnalyzer.run(
        data_folder_path=r"C:\Users\lazeb\Desktop\-feature_selection_explenability_stability\meta_table_data",
        results_folder_path=r"C:\Users\lazeb\Desktop\-feature_selection_explenability_stability\stability_results")
