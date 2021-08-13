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
            data = data.sample(frac=1,
                               random_state=73).reset_index(drop=True)
        chunk_size = math.floor(data.shape[0] / k_folds)
        tests_datasets = [data.iloc[:chunk_index * chunk_size] for chunk_index in range(0, data.shape[0], chunk_size)]

        feature_sets = []
        scores = [0]
        for i in range(k_folds):
            feature_sets.append(feature_selection_method(tests_datasets[i]))
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
        tests_datasets = [data[list(data)[:up_to_feature_index]] for up_to_feature_index in range(1, data.shape[1])]

        feature_sets = []
        scores = [0]
        for i in range(len(tests_datasets)):
            feature_sets.append(feature_selection_method(tests_datasets[i]))
            if i > 0:
                scores.append(feature_sets_differance_metrics(feature_sets[i], feature_sets[i - 1]))

        # save results if asked
        FeatureSelectionStabilityTests._save_plot_raw_data(save_test_raw_data=save_test_raw_data,
                                                           save_test_plot=save_test_plot,
                                                           scores=scores,
                                                           x_size=len(tests_datasets))
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
        chunk_size = math.floor(data.shape[0] / k_folds)

        feature_sets = []
        scores = [0]

        for up_to_feature_index in range(1, data.shape[1]):
            for chunk_index in range(0, data.shape[0], chunk_size):
                test_dataset = data[list(data)[:up_to_feature_index]].iloc[:chunk_index * chunk_size]
                feature_sets.append(feature_selection_method(test_dataset))

        # save results if asked
        FeatureSelectionStabilityTests._save_plot_raw_data(save_test_raw_data=save_test_raw_data,
                                                           save_test_plot=save_test_plot,
                                                           scores=scores,
                                                           x_size=len(data))

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
