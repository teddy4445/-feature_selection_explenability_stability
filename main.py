# library imports
import os
import math

# project imports
from meta_data_table_generator import MetaDataTableGenerator
from explainable_performance_pipeline_analyzer import ExplainablePerformancePipelineAnalyzer


class Main:
    """
    Analyze the meta-data set of FS algorithms
    """

    # consts #
    RESULTS_FOLDER_NAME = "results"
    DATA_FOLDER_NAME = "data_fixed"

    DATA_FOLDER = os.path.join(os.path.dirname(__file__), DATA_FOLDER_NAME)
    RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME)
    RESULTS_EXP_FOLDER = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME, "expandability")
    RESULTS_STABILITY_FOLDER = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME, "stability")
    # end - consts #

    def __init__(self):
        pass

    @staticmethod
    def run_all_project(data_folder_path: str,
                        results_folder_path: str):
        Main.prepare_meta_data_set(data_folder_path=data_folder_path,
                                   results_folder_path=results_folder_path)
        Main.prepare_meta_data_set(data_folder_path=data_folder_path,
                                   results_folder_path=results_folder_path)
        Main.analyze_stability_test(data_folder_path=data_folder_path,
                                    results_folder_path=results_folder_path)

    @staticmethod
    def prepare_meta_data_set(data_folder_path: str,
                              results_folder_path: str):
        MetaDataTableGenerator.run(data_folder_path=data_folder_path,
                                   results_folder_path=results_folder_path)

    @staticmethod
    def analyze_stability_test(data_folder_path: str,
                               results_folder_path: str):
        StabilityTestsAnalyzer.run(data_folder_path=data_folder_path,
                                   results_folder_path=results_folder_path)

    @staticmethod
    def analyze_expandability_performance_pipeline(data_folder_path: str,
                                                   results_folder_path: str):
        ExplainablePerformancePipelineAnalyzer.run(data_folder_path=data_folder_path,
                                                   results_folder_path=results_folder_path)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Main>"


if __name__ == '__main__':
    Main.run_all_project(data_folder_path=Main.DATA_FOLDER,
                         results_folder_path=Main.RESULTS_FOLDER)
