# library imports
import pandas as pd

class DatasetPropertiesMeasurements:
    """
    A collection of analysis for a tabular dataset
    """

    @staticmethod
    def get_dataset_profile(dataset: pd.DataFrame):
        """
        Run all the measurements and summary them up in a dict
        """
        return {
            "row_count": DatasetPropertiesMeasurements.row_count(dataset=dataset),
            "col_count": DatasetPropertiesMeasurements.col_count(dataset=dataset),
            "classes_count": DatasetPropertiesMeasurements.classes_count(dataset=dataset),
            "average_pearson_linearly": DatasetPropertiesMeasurements.average_pearson_linearly(dataset=dataset),
            "average_linearly_to_target": DatasetPropertiesMeasurements.average_linearly_to_target(dataset=dataset)
        }

    @staticmethod
    def row_count(dataset: pd.DataFrame):
        return dataset.shape[0]

    @staticmethod
    def col_count(dataset: pd.DataFrame):
        return dataset.shape[1]

    @staticmethod
    def classes_count(dataset: pd.DataFrame):
        return dataset["target"].nunique()

    @staticmethod
    def average_pearson_linearly(dataset: pd.DataFrame):
        """
        :return: the average person correlation in the matrix
        """
        pass

    @staticmethod
    def average_linearly_to_target(dataset: pd.DataFrame):
        """
        :return: the average R^2 between the explainable features and the target feature
        """
        pass
