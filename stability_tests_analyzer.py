# library imports
import os
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.cluster import hierarchy
from matplotlib.ticker import LinearLocator
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

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
        StabilityTestsAnalyzer.scatter_data(mdf=mdf,
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
           ** The z-values needs to be normalized between 0 and 1
        """
        # prepare inner folders
        threed_prints_folder_name = "3d_plots"
        instance_bar_prints_folder_name = "instance_bar_plots"
        stability_instance_bar_prints_folder_name = "stability_instance_bar_plots"
        inner_folders = [threed_prints_folder_name, instance_bar_prints_folder_name,
                         stability_instance_bar_prints_folder_name]
        for folder_name in inner_folders:
            try:
                os.mkdir(os.path.join(results_folder_path, folder_name))
            except:
                pass

        # first, find all the FS algorithms + stability tests
        fs_algorithms = {}
        stability_tests = {}
        column_names = list(mdf)
        stability_column_names = []
        for column_name in column_names:
            try:
                stability_metric, fs_filter, stability_test = StabilityTestsAnalyzer._get_column_signature(
                    column_name=column_name)
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
            z = [round((z_value - z_min) / z_delta, 3) for z_value in z]
            three_d_z = np.asarray(z).reshape(y_size, x_size)
            # make 2d grid
            x, y = np.meshgrid(x, y)
            # plot the surface.
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                                   figsize=(10, 10))
            ax.plot_wireframe(x,
                              y,
                              three_d_z,
                              rstride=1,
                              cstride=1,
                              color="black",
                              alpha=0.5,
                              cmap=cm.coolwarm)
            ax.scatter(x,
                       y,
                       three_d_z,
                       c=three_d_z,
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
            plt.savefig(os.path.join(results_folder_path, threed_prints_folder_name, "{}.png".format(row["ds_name"])))
            # close for next plot
            plt.close()

            # save data as a bar plot #
            x = np.arange(len(fs_algorithms))  # the label locations
            width = 0.9 / len(stability_tests)  # the width of the bars
            # prepare plot
            fig, ax = plt.subplots()
            for stability_test_index, stability_test in enumerate(stability_tests):
                rects = ax.bar(x - 0.45 + width * stability_test_index,
                               z[stability_test_index::len(stability_tests)],
                               width,
                               label=stability_test)
                # TODO: add the following line to have numbers on top
                # ax.bar_label(rects, padding=3)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel("{} [1]".format(stability_metric))
            ax.set_xticks(x)
            ax.set_yticks(np.linspace(0, 1, 11))
            ax.set_xticklabels([value.replace("_", " ") for value in list(fs_algorithms.keys())])
            ax.legend()
            plt.grid(axis="y",
                     alpha=0.5)
            plt.xticks(rotation=90)
            fig.tight_layout()
            # save the plot
            plt.savefig(os.path.join(results_folder_path, instance_bar_prints_folder_name,
                                     "{}_bar_plot.png".format(row["ds_name"])))
            # close for next plot
            plt.close()

            # save data as a bar plot with stability test as the leading #
            x = np.arange(len(stability_tests))  # the label locations
            width = 0.9
            # prepare plot
            fig, ax = plt.subplots()
            rects = ax.bar(x,
                           [np.nanmean([z[stability_test_index::len(stability_tests)]]) for stability_test_index in
                            range(len(stability_tests))],
                           yerr=[np.nanstd([z[stability_test_index::len(stability_tests)]]) for stability_test_index in
                                 range(len(stability_tests))],
                           width=width,
                           capsize=5)
            # TODO: add the following line to have numbers on top
            # ax.bar_label(rects, padding=3)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel("{} [1]".format(stability_metric))
            ax.set_xticks(x)
            ax.set_xticklabels(list(stability_tests.keys()))
            plt.grid(axis="y",
                     alpha=0.5)
            plt.xticks(rotation=0)
            fig.tight_layout()
            # save the plot
            plt.savefig(os.path.join(results_folder_path, stability_instance_bar_prints_folder_name,
                                     "{}_stability_bar_plot.png".format(row["ds_name"])))
            # close for next plot
            plt.close()

    @staticmethod
    def hierarchical_clustering(mdf,
                                results_folder_path: str):
        """
        This method responsible to show which datasets are similar each one on the 'x' features
        and than similar in the results for the same FS, stability test, and FS set distance metric.

        This method will generate a hierarchical clustering dendrogram plot (for example: https://prnt.sc/1o4w2cq) where the samples
        are the 'x' features of each row in the MDF and a 'y' feature which is one stability-related column.
        * This process repeats for every 'y' feature which is one stability-related column in the MDF.
        """
        # prepare method's inner folder
        inner_folder_name = "hierarchical_clustering_plots"
        try:
            os.mkdir(os.path.join(results_folder_path, inner_folder_name))
        except:
            pass
        # load the fs algorithms and stability tests
        fs_algorithms = {}
        stability_tests = {}
        column_names = list(mdf)
        stability_column_names = ["baseline"]
        x_column_names = []
        for column_name in column_names:
            # get the x represented columns
            if "x-" in column_name:
                x_column_names.append(column_name)
            try:
                stability_metric, fs_filter, stability_test = StabilityTestsAnalyzer._get_column_signature(
                    column_name=column_name)
                if fs_filter not in fs_algorithms:
                    fs_algorithms[fs_filter] = len(fs_algorithms.keys())
                if stability_test not in stability_tests:
                    stability_tests[stability_test] = len(stability_tests.keys())
                stability_column_names.append(column_name)
            except:
                pass

        # format for each row the representing x-vector of it
        ds_embeddings = []
        ds_labels = []
        # normlize the dataset to handle same size data
        mdf_x_data = mdf[x_column_names].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        mdf_x_data_scaled = min_max_scaler.fit_transform(mdf_x_data)
        hierarchy_mdf = pd.DataFrame(mdf_x_data_scaled, columns=mdf[x_column_names].columns)
        # prepare the rows for the clustering later
        for index, row in hierarchy_mdf.iterrows():
            ds_embeddings.append(
                [row[column_name] if not np.isnan(row[column_name]) else 0 for column_name in x_column_names])
            ds_embeddings[index].append(0)
        # find the db_names
        for index, row in mdf.iterrows():
            ds_labels.append(row["ds_name"].replace(".csv", ""))

        # plot  hierarchy dendrogram
        for stability_column in stability_column_names:
            # prepare data
            if stability_column != "baseline":
                for ds_index, ds_embedding in enumerate(ds_embeddings):
                    ds_embedding[-1] = mdf[stability_column][ds_index]
            plt.figure()
            hierarchy_z = hierarchy.linkage(ds_embeddings, method='single')
            hierarchy.dendrogram(hierarchy_z,
                                 labels=np.asarray(ds_labels),
                                 distance_sort='descending',
                                 show_leaf_counts=True)
            plt.xticks(rotation=90)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(os.path.join(results_folder_path, inner_folder_name,
                                     "hierarchical_plot_for_{}.png".format(stability_column)))
            plt.close()

    @staticmethod
    def scatter_data(mdf,
                     results_folder_path: str):
        """
        This method responsible to generate a 3d scatter plot:
        1. X-axis: DB
        2. Y-axis: FS algorithm
        3. Z-axis: score
        4. Color: stability test
        """
        # prepare inner folders
        db_view = "db_view"
        fs_filter_view = "fs_filter_view"
        inner_folders = [db_view, fs_filter_view]
        for folder_name in inner_folders:
            try:
                os.mkdir(os.path.join(results_folder_path, folder_name))
            except:
                pass

        # load the fs algorithms and stability tests
        fs_algorithms = {}
        stability_tests = {}
        stability_column_names = []
        for column_name in list(mdf):
            try:
                stability_metric, fs_filter, stability_test = StabilityTestsAnalyzer._get_column_signature(
                    column_name=column_name)
                if fs_filter not in fs_algorithms:
                    fs_algorithms[fs_filter] = len(fs_algorithms.keys())
                if stability_test not in stability_tests:
                    stability_tests[stability_test] = len(stability_tests.keys())
                stability_column_names.append(column_name)
            except:
                pass
        datasets_names = {name: index for index, name in enumerate(list(mdf["ds_name"]))}

        # gather the scatter dots
        x = []
        y = []
        z = []
        color = []
        for index, row in mdf.iterrows():
            db_name = row["ds_name"]
            for stability_column in stability_column_names:
                stability_metric, fs_filter, stability_test = StabilityTestsAnalyzer._get_column_signature(
                    column_name=stability_column)
                x.append(datasets_names[db_name])
                y.append(fs_algorithms[fs_filter])
                z.append(row[stability_column])
                color.append(stability_tests[stability_test])

        # prepare colors
        color_mapper = {0: "red", 1: "blue", 2: "green"}
        color_names = [color_mapper[value] for value in color]

        # plot scatter
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                               figsize=(10, 10))
        for color_group in set(color):
            ax.scatter([value for index, value in enumerate(x) if color[index] == color_group],
                       [value for index, value in enumerate(y) if color[index] == color_group],
                       [value for index, value in enumerate(z) if color[index] == color_group],
                       s=30,
                       marker="o",
                       c=color_mapper[color_group],
                       alpha=0.75,
                       label="{}".format(list(stability_tests.keys())[color_group]))
        ax.set_zlim(0, 1)
        ax.set_xticks(list(range(len(list(datasets_names.keys())))))
        ax.set_xticklabels(list(datasets_names.keys()))
        ax.set_yticks(list(range(len(list(fs_algorithms.keys())))))
        ax.set_yticklabels([value.replace("_", " ") for value in list(fs_algorithms.keys())])
        ax.set_zlabel("Score")
        plt.legend()
        plt.savefig(os.path.join(results_folder_path, "summary_scatter_plot.png"))
        plt.close()

        # generate view in the DB level
        datasets_names_list = list(datasets_names.keys())
        points_in_db = int(len(x) / len(datasets_names_list))
        for db_index in range(len(datasets_names_list)):
            y_db = y[db_index*points_in_db:(db_index+1)*points_in_db]
            z_db = z[db_index*points_in_db:(db_index+1)*points_in_db]
            fig, ax = plt.subplots(figsize=(8, 8))
            for color_group in set(color):
                x_final = [value for index, value in enumerate(y_db) if color[index] == color_group]
                y_final = [value for index, value in enumerate(z_db) if color[index] == color_group]
                ax.scatter(x_final,
                           y_final,
                           s=30,
                           marker="o",
                           c=color_mapper[color_group],
                           alpha=0.75,
                           label="{}".format(list(stability_tests.keys())[color_group]))
                # calc stats
                mean_y_final = np.mean(y_final)
                std_y_final = np.std(y_final)
                ax.fill_between([0, len(x_final)-1],
                                [mean_y_final + std_y_final, mean_y_final + std_y_final],
                                [mean_y_final - std_y_final, mean_y_final - std_y_final],
                                color=color_mapper[color_group],
                                alpha=0.1)
                ax.plot([0, len(x_final)-1],
                        [mean_y_final, mean_y_final],
                        color=color_mapper[color_group],
                        alpha=0.8)
            ax.set_xticks(list(range(len(list(fs_algorithms.keys())))))
            ax.set_xticklabels([value.replace("_", " ") for value in list(fs_algorithms.keys())], rotation=90)
            ax.set_yticks([value / 10 for value in list(range(11))])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_xlabel("FS algorithm")
            plt.title("Test on DB: '{}'".format(datasets_names_list[db_index]))
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder_path, db_view, "summary_scatter_plot_{}_db_view.png".format(datasets_names_list[db_index])))
            plt.close()
            print("StabilityTestsAnalyzer.scatter_data: finish DB view of {}".format(datasets_names_list[db_index]))

        # generate view in the FS level
        fs_algorithms_list = list(fs_algorithms.keys())
        points_in_fs_algorithm = int(len(x) / len(fs_algorithms_list))
        for fs_algorithm_index in range(len(fs_algorithms_list)):
            db_numbers = list(range(int(points_in_fs_algorithm/len(list(stability_tests.keys())))))
            y_fs = []
            [y_fs.extend(db_numbers) for i in range(len(list(stability_tests.keys())))]
            z_fs = z[fs_algorithm_index*points_in_fs_algorithm:(fs_algorithm_index+1)*points_in_fs_algorithm]
            fig, ax = plt.subplots(figsize=(8, 8))
            for color_group in set(color):
                x_final = [value for index, value in enumerate(y_fs) if color[index] == color_group]
                y_final = [value for index, value in enumerate(z_fs) if color[index] == color_group]
                ax.scatter(x_final,
                           y_final,
                           s=30,
                           marker="o",
                           c=color_mapper[color_group],
                           alpha=0.75,
                           label="{}".format(list(stability_tests.keys())[color_group]))
                # calc stats
                mean_y_final = np.mean(y_final)
                std_y_final = np.std(y_final)
                ax.fill_between([0, len(x_final)-1],
                                [mean_y_final + std_y_final, mean_y_final + std_y_final],
                                [mean_y_final - std_y_final, mean_y_final - std_y_final],
                                color=color_mapper[color_group],
                                alpha=0.1)
                ax.plot([0, len(x_final)-1],
                        [mean_y_final, mean_y_final],
                        color=color_mapper[color_group],
                        alpha=0.8)
            ax.set_xticks(list(range(len(list(datasets_names.keys())))))
            ax.set_xticklabels([value.replace("_", " ").replace(".csv", "") for value in list(datasets_names.keys())], rotation=90)
            ax.set_yticks([value / 10 for value in list(range(11))])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_xlabel("DB")
            plt.title("Test on FS: '{}'".format(fs_algorithms_list[fs_algorithm_index]))
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder_path, fs_filter_view, "summary_scatter_plot_{}_fs_view.png".format(fs_algorithms_list[fs_algorithm_index])))
            plt.close()
            print("StabilityTestsAnalyzer.scatter_data: finish FS view of {}".format(fs_algorithms_list[fs_algorithm_index]))

    @staticmethod
    def prepare_dataset_for_classificator_learning(mdf,
                                                   results_folder_path: str):
        """
        This method responsible to generate a new table from the MDF for later classification problem.
        The new table is constructed from the 'x' columns of the MDF (repeating for multiple lines)
        with a column that indicates the FS algorithm used (according to some indexing of the FS algorithms)
        and a column that indicates if for this FS and dataset the data-stability (0), feature-stability (1),
        or Lyapunov-stability (2) test that has the highest value.

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
