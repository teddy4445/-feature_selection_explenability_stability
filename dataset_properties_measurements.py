class DatasetPropertiesMeasurements:
    """
    A collection of analysis for a tabular dataset
    """

    @staticmethod
    def get_dataset_profile(dataset):
        """
        Run all the measurements and summary them up in a dict
        """
        return {
            "row_count": DatasetPropertiesMeasurements.row_count(dataset=dataset),
            "col_count": DatasetPropertiesMeasurements.col_count(dataset=dataset)
        }

    @staticmethod
    def row_count(dataset):
        pass

    @staticmethod
    def col_count(dataset):
        pass
