import os
import numpy as np
import pandas as pd
from glob import glob

from pandas.core.dtypes.common import is_numeric_dtype


class DataSetPrepare:
    """
    A class that runs once to prepare the datasets into the same style we can work with
    """

    # CONSTS #
    MAX_RATIO = 0.2
    MAX_UNIQUE_VALUES = 30
    MAX_DATA_POINTS = 10000
    MAX_FEATURES = 50
    MAX_ROWS = 1000
    MIN_FEATURES_TO_SAVE = 2
    MIN_ROWS_TO_SAVE = 20
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
        # run on each file in the source and save it
        for file_path in glob(os.path.join(data_sets_source_folder_path, "*.csv")):
            # alert
            if os.path.exists(os.path.join(data_sets_target_folder_path, os.path.basename(file_path))):
                print("Skipping {}".format(os.path.basename(file_path)))
                continue
            print("Working on {}".format(os.path.basename(file_path)))
            # read the data
            df = pd.read_csv(file_path)

            # we do not want to handle too large feature space
            if df.shape[1] > DataSetPrepare.MAX_FEATURES:
                df = df.iloc[:, -1 * DataSetPrepare.MAX_FEATURES:]
            # if the dataset is too large, calc the number of rows needed to be in order to obtain max dataset size
            if df.shape[0] * df.shape[1] > DataSetPrepare.MAX_DATA_POINTS:
                max_rows = round(DataSetPrepare.MAX_DATA_POINTS / df.shape[1])
                df = df.iloc[:max_rows, :]

            # move target column to the right (last column)
            target_col = df.pop('target')

            # fix columns
            col_to_remove = []
            for col in list(df):
                unique_count = df[col].nunique()
                if not is_numeric_dtype(df[col]):
                    if unique_count / df[col].size > DataSetPrepare.MAX_RATIO or unique_count > DataSetPrepare.MAX_UNIQUE_VALUES:
                        col_to_remove.append(col)
                    else:
                        # TODO: maybe use one hot encoding instead
                        mapper = {value: index for index, value in enumerate(list(df[col].unique()))}
                        df[col] = df[col].apply(lambda x: mapper[x])
            # remove what we do not need
            df.drop(col_to_remove, axis=1, inplace=True)

            # add targets to the right
            df['target'] = target_col

            #  when we have only the data we need, remove lines with nan
            df.dropna(inplace=True)

            # convert target to integers
            mapper = {value: index for index, value in enumerate(list(df['target'].unique()))}
            df['target'] = df['target'].apply(lambda x: mapper[x])

            # if after this is ready we have too much rows, reduce it to manageable size
            if df.shape[0] > DataSetPrepare.MAX_ROWS:
                df = df.iloc[:DataSetPrepare.MAX_ROWS,:]

            # if the df is not reduced too much after fixing
            if df.shape[0] > DataSetPrepare.MIN_ROWS_TO_SAVE and df.shape[1] > DataSetPrepare.MIN_FEATURES_TO_SAVE:
                # save the result
                df.to_csv(os.path.join(data_sets_target_folder_path, os.path.basename(file_path)))
