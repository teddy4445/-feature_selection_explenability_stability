# library imports
import os
import math


# project imports


class Main:
    """
    Analyze the meta-data set of FS algorithms
    """

    RESULTS_FOLDER_NAME = "results"
    DATA_FOLDER_NAME = "data"

    DATA_FOLDER = os.path.join(os.path.dirname(__file__), DATA_FOLDER_NAME)
    RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME)
    RESULTS_EXP_FOLDER = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME, "expandability")
    RESULTS_STABILITY_FOLDER = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME, "stability")

    def __init__(self):
        pass


    @staticmethod
    def analyze_stability(data_folder_path: str,
                          results_folder_path: str):
        pass

    @staticmethod
    def analyze_expandability(data_folder_path: str,
                              results_folder_path: str):
        pass

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Main>"


if __name__ == '__main__':
    Main.analyze_stability(data_folder_path=Main.DATA_FOLDER,
                           results_folder_path=Main.RESULTS_FOLDER)
    Main.analyze_expandability(data_folder_path=Main.DATA_FOLDER,
                               results_folder_path=Main.RESULTS_FOLDER)
