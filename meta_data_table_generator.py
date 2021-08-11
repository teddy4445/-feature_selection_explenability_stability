# library imports
import os
import glob
import pandas as pd

# project imports
from feature_selection_algorithms import FeatureSelectionAlgorithms
from dataset_properties_measurements import DatasetPropertiesMeasurements
from feature_selection_stability_tests import FeatureSelectionStabilityTests
from feature_selection_sets_differences import FeatureSelectionSetsDifferences


class MetaDataTableGenerator:
    """
    A class responsible to run a set of experiments, generating a dataset to study explainable
    methods and the stability of these methods.
    """

    FILE_NAME = "meta_dataset.csv"

    METRICS = ["rosenfeld_f_metric_{}_harmonic_mean".format(value) for value in ["accuracy", "recall", "precision", "r2"]]
    METRICS.extend(["rosenfeld_f_metric_{}_mean".format(value) for value in ["accuracy", "recall", "precision", "r2"]])

    STABILITY_TESTS = ["data",
                       "features",
                       "lyapunov"]

    STABILITY_METRICS = ["iou"]

    FS_FILTER = ["chi_square",
                 "symmetrical_uncertainty",
                 "information_gain",
                 "pearson_correlation",
                 "spearman_correlation",
                 "remove_low_variance",
                 "missing_value_ratio",
                 "fishers_score",
                 "permutation_feature_importance",
                 "support_vector_machines_recursive_feature_elimination"]

    FS_EMBEDDING = ["dt",
                    "lasso",
                    "linearSVC"]

    def __init__(self):
        pass

    @staticmethod
    def run(data_folder_path: str,
            results_folder_path: str):
        """
        data_folder_path: is the path to the folder with all the data
        answer_folder_path: is the path to the folder we wish to write our results to
        """
        # start by making sure we have the answer folder
        try:
            os.mkdir(results_folder_path)
        except:
            pass
        # create empty dataset we will populate during the function
        answer_df = pd.DataFrame(data=None, columns=MetaDataTableGenerator.prepare_columns())
        # if we wish to see debugging
        DatasetPropertiesMeasurements.IS_DEBUG = True
        for path in glob.glob(os.path.join(data_folder_path, "*")):
            # print the file name if debugging mood
            if DatasetPropertiesMeasurements.IS_DEBUG:
                print("\n\nMetaDataTableGenerator: Working on file {}".format(path))
            # get file's data
            dataset = pd.read_csv(path)
            # get its name
            dataset_name = os.path.basename(path)

            """
            # save this dataset raw results in its folder
            try:
                os.mkdir(os.path.join(answer_folder_path, dataset_name))
            except:
                pass
            """

            # calculate the data's vector
            dataset_summary_results = MetaDataTableGenerator.calculate_single_dataset_vector(dataset=dataset,
                                                                                             dataset_name=dataset_name)

            # add to the global dataframe
            answer_df.append(dataset_summary_results)
        # save the results to the basic folder
        answer_df.to_csv(os.path.join(results_folder_path, MetaDataTableGenerator.FILE_NAME))

    # HELP FUNCTIONS #

    @staticmethod
    def prepare_columns():
        """
        Prepare the columns of the meta-data table
        """
        columns = ["ds_name"]
        # get the "X" columns
        columns.extend(["x_{}".format(col_name) for col_name in DatasetPropertiesMeasurements.get_columns()])
        # get the FS filter + FS embedding + metrics
        for metric in MetaDataTableGenerator.METRICS:
            for fs_filter in MetaDataTableGenerator.FS_FILTER:
                for fs_embedding in MetaDataTableGenerator.FS_EMBEDDING:
                    columns.append("expandability_{}_{}_{}".format(metric,
                                                                   fs_filter,
                                                                   fs_embedding))
        # get the FS filter + stability test + stability metric
        for stability_metric in MetaDataTableGenerator.STABILITY_METRICS:
            for fs_filter in MetaDataTableGenerator.FS_FILTER:
                for stability_test in MetaDataTableGenerator.STABILITY_TESTS:
                    columns.append("stability_{}_{}_{}".format(stability_metric,
                                                               fs_filter,
                                                               stability_test))
        return columns

    @staticmethod
    def calculate_single_dataset_vector(dataset,
                                        dataset_name: str):
        """
        Calculate the row corosponding to each dataset in the meta-data table
        """
        # NOTE: we assume that all the datasets are classification tasks and that the last column is the target column #
        dataset_summary_results = [dataset_name]
        # perform all the needed tests and experiments on this dataset
        dataset_summary_results.extend(DatasetPropertiesMeasurements.get_dataset_profile_vector(dataset=dataset))

        # columns for the expandability feature selection
        for metric in MetaDataTableGenerator.METRICS:
            for fs_filter in MetaDataTableGenerator.FS_FILTER:
                for fs_embedding in MetaDataTableGenerator.FS_EMBEDDING:
                    # TODO: finish here
                    dataset_summary_results.append(1)

        # columns for the stability feature selection
        for stability_metric in MetaDataTableGenerator.STABILITY_METRICS:
            for fs_filter in MetaDataTableGenerator.FS_FILTER:
                for stability_test in MetaDataTableGenerator.STABILITY_TESTS:
                    # TODO: finish here
                    dataset_summary_results.append(1)
        # return answer
        return dataset_summary_results

    # END - HELP FUNCTIONS #

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<MetaDataTableGenerator>"


if __name__ == '__main__':
    MetaDataTableGenerator.run(data_folder_path=os.path.join(os.path.dirname(__file__), "data_fixed"),
                               results_folder_path=os.path.join(os.path.dirname(__file__), "meta_table_data"))
