# library imports
import os
import pandas as pd


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
        mdf = pd.read_csv(summary_file_results_path)
        ExplainablePerformancePipelineAnalyzer.analyze_1(mdf=mdf,
                                                         answers_folder_path=answers_folder_path)
        ExplainablePerformancePipelineAnalyzer.analyze_2(mdf=mdf,
                                                         answers_folder_path=answers_folder_path)
        ExplainablePerformancePipelineAnalyzer.analyze_3(mdf=mdf,
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
