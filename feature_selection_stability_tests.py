import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression


class FeatureSelectionStabilityTests:
    """
    A list of stability tests
    """

    # PROPOSED METRICS #

    def __init__(self):
        pass

    @staticmethod
    def data_stability_test(data,
                            feature_selection_method,
                            feature_sets_differance_metrics,
                            k_folds: int = 10,
                            shuffle: bool = False,
                            save_test_plot: str = None,
                            save_test_raw_data: str = None):
        """
        increase the size of the data (like it added over time) in 'k_fold' sizes,
        run each time the FS algorithm and test the changes
        """
        if shuffle:
            # TODO: move 73 to global constant
            data = data.sample(frac=1,
                               random_state=73).reset_index(drop=True)
        chunk_size = math.floor(data.shape[0] / k_folds)
        tests_datasets = [data.iloc[:chunk_index * chunk_size] for chunk_index in range(1, k_folds)]
        tests_datasets.append(data)

        feature_sets = []
        scores = [0]
        for i in range(k_folds):

            # TODO: move this magic word outside
            x = tests_datasets[i].drop("target", axis=1)
            y = tests_datasets[i]["target"]

            feature_sets.append(list(feature_selection_method(x, y)))
            if i > 0:
                scores.append(feature_sets_differance_metrics(feature_sets[i], feature_sets[i - 1]))

        # save results if asked
        FeatureSelectionStabilityTests._save_plot_raw_data(save_test_raw_data=save_test_raw_data,
                                                           save_test_plot=save_test_plot,
                                                           scores=scores,
                                                           x_size=k_folds)
        # return the scores to the caller
        return scores

    @staticmethod
    def feature_stability_test(data,
                               feature_selection_method,
                               feature_sets_differance_metrics,
                               save_test_plot: str = None,
                               save_test_raw_data: str = None):
        """
        increase the number of features, run each time the FS algorithm and test the changes
        """
        # TODO: move this magic word outside
        x = data.drop("target", axis=1)
        y = data["target"]

        x_datasets = [x[list(x)[:up_to_feature_index]] for up_to_feature_index in range(1, x.shape[1])]

        feature_sets = []
        scores = [0]
        for i in range(len(x_datasets)):
            feature_sets.append(list(feature_selection_method(x_datasets[i], y)))
            if i > 0:
                scores.append(feature_sets_differance_metrics(feature_sets[i], feature_sets[i - 1]))

        # save results if asked
        FeatureSelectionStabilityTests._save_plot_raw_data(save_test_raw_data=save_test_raw_data,
                                                           save_test_plot=save_test_plot,
                                                           scores=scores,
                                                           x_size=len(x_datasets))
        # return the scores to the caller
        return scores

    @staticmethod
    def lyapunov_stability_test(data,
                                feature_selection_method,
                                feature_sets_differance_metrics,
                                k_folds: int = 10,
                                shuffle: bool = False,
                                save_test_plot: str = None,
                                save_test_raw_data: str = None):
        """
        increase the size of the data (like it added over time) in 'k_fold' sizes,
        run each time the FS algorithm and test the changes
        """
        if shuffle:
            data = data.sample(frac=1,
                               random_state=73).reset_index(drop=True)

        feature_sets = []
        scores = [0]

        # TODO: move this magic word outside
        x = data.drop("target", axis=1)
        y = data["target"]

        max_cut_index = min(x.shape[1], k_folds)
        chunk_size = math.floor(data.shape[0] / max_cut_index)

        for cut_index in range(max_cut_index):
            x_dataset = x[list(x)[:cut_index+1]].iloc[:(cut_index+1) * chunk_size]

            feature_sets.append(list(feature_selection_method(x_dataset,
                                                              y.iloc[:(cut_index+1) * chunk_size])))
            if cut_index > 0:
                scores.append(feature_sets_differance_metrics(feature_sets[cut_index], feature_sets[cut_index - 1]))

        # save results if asked
        FeatureSelectionStabilityTests._save_plot_raw_data(save_test_raw_data=save_test_raw_data,
                                                           save_test_plot=save_test_plot,
                                                           scores=scores,
                                                           x_size=max_cut_index)

        # return the scores to the caller
        return scores

    # END - PROPOSED METRICS #

    # HELP FUNCTIONS #

    @staticmethod
    def _save_plot_raw_data(save_test_plot: str,
                            save_test_raw_data: str,
                            scores: list,
                            x_size: int):

        # if asked, save the plot of the this test
        if save_test_plot is not None:
            fig, ax = plt.subplots()
            ax.plot(list(range(1, x_size + 1)), scores, "-o", color="k")
            ax.set(xlabel='Test index [1]', ylabel='Differance in the features set [1]')
            ax.grid()
            fig.savefig(save_test_plot)
            plt.close()

        # if asked, save the csv of the this test
        if save_test_raw_data is not None:
            with open(save_test_raw_data, "w") as raw_data_file:
                raw_data_file.write("test_index,differance\n")
                [raw_data_file.write("{},{}\n".format(index, scores[index - 1])) for index in range(1, x_size + 1)]

    @staticmethod
    def _save_plot_raw_data_2d(save_test_plot: str,
                               save_test_raw_data: str,
                               scores: list,
                               x_size: int,
                               y_size: int):

        # if asked, save the plot of the this test
        if save_test_plot is not None:
            fig, ax = plt.subplots()
            ax.plot(list(range(1, x_size + 1)), scores, "-o", color="k")
            ax.set(xlabel='Test index [1]', ylabel='Differance in the features set [1]')
            ax.grid()
            fig.savefig(save_test_plot)
            plt.close()

        # if asked, save the csv of the this test
        if save_test_raw_data is not None:
            with open(save_test_raw_data, "w") as raw_data_file:
                raw_data_file.write("test_index,differance\n")
                [raw_data_file.write("{},{}\n".format(index, scores[index - 1])) for index in range(1, x_size + 1)]

    # END - HELP FUNCTIONS #

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<FeatureSelectionStabilityTests>"
