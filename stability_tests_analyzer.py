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
        StabilityTestsAnalyzer.analyze_1(mdf=mdf,
                                         results_folder_path=results_folder_path)
        StabilityTestsAnalyzer.analyze_2(mdf=mdf,
                                         results_folder_path=results_folder_path)
        StabilityTestsAnalyzer.analyze_3(mdf=mdf,
                                         results_folder_path=results_folder_path)

    @staticmethod
    def analyze_1(mdf,
                  results_folder_path: str):
        pass

    @staticmethod
    def analyze_2(mdf,
                  results_folder_path: str):
        pass

    @staticmethod
    def analyze_3(mdf,
                  results_folder_path: str):
        pass
