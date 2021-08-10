import os
import numpy as np
import pandas as pd
from glob import glob


class DataSetPrepare:
    """
    A class that runs once to prepare the datasets into the same style we can work with
    """

    # CONSTS #
    MAX_RATIO = 0.05
    MAX_UNIQUE_VALUES = 20
    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(data_sets_source_folder_path: str,
            data_sets_target_folder_path: str):
        """
        Preparing the data set (single entry point of the class)
        :param data_sets_source_folder_path: where the original data is located
        :param data_sets_target_folder_path: where the fixed data is saved
        """
        # make sure we have target folder
        try:
            os.makedirs(data_sets_target_folder_path)
        except:
            pass
        # run on each file in the source and save it
        for file_path in glob(os.path.join(data_sets_source_folder_path, "*.csv")):
            # alert
            if os.path.exists(os.path.join(data_sets_target_folder_path, os.path.basename(file_path))):
                print("Skipping {}".format(os.path.basename(file_path)))
                continue
            print("Starting with {}".format(os.path.basename(file_path)))
            # read the data
            df = pd.read_csv(file_path)
            # move target column to the right (last column)
            target_col = df.pop('target')
            df['target'] = target_col
            # fix columns
            col_to_remove = []
            for col in list(df):
                unique_count = df[col].nunique()
                if df[col].dtype != np.number:
                    if unique_count / df[col].size > DataSetPrepare.MAX_RATIO or unique_count > DataSetPrepare.MAX_UNIQUE_VALUES:
                        col_to_remove.append(col)
                    else:
                        # TODO: maybe use one hot encoding instead
                        mapper = {value: index for index, value in enumerate(list(df[col].unique()))}
                        df[col] = df[col].apply(lambda x: mapper[x])
            # remove what we do not need
            df.drop(col_to_remove, axis=1, inplace=True)
            # Finally, when we have only the data we need, remove lines with nan
            df.dropna(inplace=True)
            # save the result
            df.to_csv(os.path.join(data_sets_target_folder_path, os.path.basename(file_path)))


if __name__ == '__main__':
    DataSetPrepare.run(data_sets_source_folder_path=os.path.join(os.path.dirname(__file__), "data"),
                       data_sets_target_folder_path=os.path.join(os.path.dirname(__file__), "data_fixed"))
