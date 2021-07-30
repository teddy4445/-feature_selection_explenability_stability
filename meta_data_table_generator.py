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

    @staticmethod
    def run(data_folder_path: str,
            answer_folder_path: str):
        """
        data_folder_path: is the path to the folder with all the data
        answer_folder_path: is the path to the folder we wish to write our results to
        """
        # start by making sure we have the answer folder
        try:
            os.mkdir(answer_folder_path)
        except:
            pass
        # create empty dataset we will populate during the function
        answer_df = pd.DataFrame(data=None, columns=["ds_name",
                                                     "ds_rows",
                                                     "ds_coloums",
                                                     "ds_other_feature_1",
                                                     "ds_other_feature_2",
                                                     "ds_other_feature_3",
                                                     "ds_other_feature_4",
                                                     "ds_other_feature_5",
                                                     "ds_other_feature_6",
                                                     "ds_other_feature_7",
                                                     "ds_other_feature_8",
                                                     "ds_other_feature_9",
                                                     "ds_other_feature_10",
                                                     "stability_feature_1",
                                                     "stability_feature_2",
                                                     "stability_feature_3",
                                                     "stability_feature_4",
                                                     "stability_feature_5",
                                                     "stability_feature_6",
                                                     "stability_feature_7",
                                                     "expandability_feature_1",
                                                     "expandability_feature_2",
                                                     "expandability_feature_3",
                                                     "expandability_feature_4",
                                                     "expandability_feature_5",
                                                     "expandability_feature_6",
                                                     "expandability_feature_7"])
        for path in glob.glob(os.path.join(data_folder_path, "*")):
            dataset = pd.read_csv(path)
            # get its name
            dataset_name = os.path.basename(path)
            # NOTE: we assume that all the datasets are classification tasks and that the last column is the target column #
            dataset_summary_results = []

            # save this dataset raw results in its folder
            try:
                os.mkdir(os.path.join(answer_folder_path, dataset_name))
            except:
                pass

            # perform all the needed tests and experiments on this dataset
            # TODO: finish it later

            # add to the global dataframe
            answer_df.append(dataset_summary_results)
        # save the results to the basic folder
        answer_df.to_csv(os.path.join(data_folder_path, "meta_dataset.csv"))

    # HELP FUNCTION #

    @staticmethod
    def help():
        pass

    # END - HELP FUNCTION #
