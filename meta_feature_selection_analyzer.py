# library imports
import os
import pandas as pd


class MetaFeatureSelectionAnalyzer:
    """
    A class that provides plots, CSV files with tables, and statistical outcomes of a meta-dataset of feature selection
    computed by the MetaDataTableGenerator.run class
    """

    @staticmethod
    def all_compute(summary_file_results_path: str,
                    answers_folder_path: str):
        mdf = pd.read_csv(summary_file_results_path)
        MetaFeatureSelectionAnalyzer.analyze_1(mdf=mdf,
                                               answers_folder_path=answers_folder_path)
        MetaFeatureSelectionAnalyzer.analyze_2(mdf=mdf,
                                               answers_folder_path=answers_folder_path)
        MetaFeatureSelectionAnalyzer.analyze_3(mdf=mdf,
                                               answers_folder_path=answers_folder_path)

    @staticmethod
    def analyze_1(mdf,
                  answers_folder_path: str):
        pass

    @staticmethod
    def analyze_2(mdf,
                  answers_folder_path: str):
        pass

    @staticmethod
    def analyze_3(mdf,
                  answers_folder_path: str):
        pass
