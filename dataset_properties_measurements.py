# library imports
import scipy
import numpy as np
import pandas as pd


class DatasetPropertiesMeasurements:
    """
    A collection of analysis for a tabular dataset
    A lot of ideas from: Reif, M., Shafait, F., Dengel, A., Meta-learning for evolutionary parameter optimization of classifiers. Machine Learning. 2012, 87:357-380. ---> (https://link.springer.com/content/pdf/10.1007/s10994-012-5286-7.pdf)

    Statistical features from: (Gama and Brazdil 1995; Vilalta et al. 2004; Brazdil et al. 1994; King et al. 1995)

    More ideas from: Shen, Z., Chen, X., Garibaldi, J. M., A Novel Meta Learning Framework for Feature Selection using Data Synthesis and Fuzzy Similarity. IEEE World Congress on Computational Intelligence. 2020. ----> (https://arxiv.org/pdf/2005.09856.pdf)
    """

    @staticmethod
    def get_dataset_profile(dataset: pd.DataFrame):
        """
        Run all the measurements and summary them up in a dict
        """
        return {
            "row_count": DatasetPropertiesMeasurements.row_count(dataset=dataset),
            "col_count": DatasetPropertiesMeasurements.col_count(dataset=dataset),
            "col_numerical_count": DatasetPropertiesMeasurements.col_numerical_count(dataset=dataset),
            "col_categorical_count": DatasetPropertiesMeasurements.col_categorical_count(dataset=dataset),
            "classes_count": DatasetPropertiesMeasurements.classes_count(dataset=dataset),
            "average_pearson_linearly": DatasetPropertiesMeasurements.average_pearson_linearly(dataset=dataset),
            "average_linearly_to_target": DatasetPropertiesMeasurements.average_linearly_to_target(dataset=dataset)
        }

    @staticmethod
    def row_count(dataset: pd.DataFrame):
        # Idea from (Engels and Theusinger 1998)
        return dataset.shape[0]

    @staticmethod
    def col_count(dataset: pd.DataFrame):
        # Idea from (Engels and Theusinger 1998)
        return dataset.shape[1]

    @staticmethod
    def col_numerical_count(dataset: pd.DataFrame,
                            max_values: int = 20):
        """
        :param dataset: the data set
        :param max_values: max number of values to be considered categorical
        :return: the number of categorical colums
        """
        # Idea from (Engels and Theusinger 1998)
        data_size = len(list(dataset.iloc[0]))
        return len([1 for column in list(dataset) if dataset[column].nunique() > max_values])

    @staticmethod
    def col_categorical_count(dataset: pd.DataFrame,
                              max_values: int = 20):
        """
        :param dataset: the data set
        :param max_values: max number of values to be considered categorical
        :return: the number of categorical colums
        """
        # Idea from (Engels and Theusinger 1998)
        return len(list(dataset)) - DatasetPropertiesMeasurements.col_numerical_count(dataset=dataset,
                                                                                      max_values=max_values)

    @staticmethod
    def classes_count(dataset: pd.DataFrame):
        return dataset["target"].nunique()

    @staticmethod
    def average_linearly_to_target(dataset: pd.DataFrame):
        """
        :return: the average R^2 between the explainable features and the target feature
        """
        pass

    @staticmethod
    def cancor_1(dataset: pd.DataFrame):
        """
        :return: canonical correlation for the best single combination of features
        """
        pass

    @staticmethod
    def cancor_2(dataset: pd.DataFrame):
        """
        :return: canonical correlation for the best single combination of features orthogonal to cancor_1
        """
        pass

    @staticmethod
    def fract_1(dataset: pd.DataFrame):
        """
        :return: first normalized eigenvalues of canonical discriminant matrix
        """
        pass

    @staticmethod
    def fract_2(dataset: pd.DataFrame):
        """
        :return: second normalized eigenvalues of canonical discriminant matrix
        """
        pass

    @staticmethod
    def skewness(dataset: pd.DataFrame):
        """
        :return: mean asymmetry of the probability distributions of the features (nonnormality)
        """
        pass

    @staticmethod
    def kurtosis(dataset: pd.DataFrame):
        """
        :return: mean peakedness of the probability distributions of the features
        """
        pass

    @staticmethod
    def average_asymmetry_of_features(dataset: pd.DataFrame):
        """
        :return: It measures the average value of the Pearson’s asymmetry coefficient.
        """
        # idea from (Shen et al., 2020)
        return 3 * sum([(np.mean(dataset[column]) - np.median(dataset[column]))/np.std(dataset[column]) for column in list(dataset)]) / len(list(dataset))

    @staticmethod
    def average_correlation_between_features(dataset: pd.DataFrame):
        """
        :return: It measures the average value of Pearson’s correlation coefficient between different features.
        """
        # idea from (Shen et al., 2020)
        n = len(list(dataset))
        cols = list(dataset)
        return 2 * sum([sum([scipy.stats.pearsonr(dataset[cols[column_i]],
                                                  dataset[cols[column_j]])
                             for column_j in range(column_i + 1, len(cols))])
                        for column_i in range(len(cols)-1)]) / (n * (n-1))

    @staticmethod
    def average_coefficient_of_variation_of_feature(dataset: pd.DataFrame):
        """
        :return: It measures the average coefficient of variation by the ratio of the standard deviation and the mean of the feature values.
        """
        # idea from (Shen et al., 2020)
        n = len(list(dataset))
        return sum([np.std(dataset[column]) / np.mean(dataset[column]) for column in list(dataset)]) / len(list(dataset))

    @staticmethod
    def average_coefficient_of_anomaly(dataset: pd.DataFrame):
        """
        :return: It measures the average coefficient of anomaly by the ratio of the mean and the standard deviation of the feature values.
        """
        # idea from (Shen et al., 2020)
        n = len(list(dataset))
        return sum([np.mean(dataset[column]) / np.std(dataset[column]) for column in list(dataset)]) / len(list(dataset))

    @staticmethod
    def average_entropy_of_features(dataset: pd.DataFrame):
        """
        :return: It measures the average coefficient of anomaly by the ratio of the mean and the standard deviation of the feature values.
        """
        # idea from (Shen et al., 2020)
        n = len(list(dataset))
        return sum([scipy.stats.entropy(dataset[column]) for column in list(dataset)]) / len(list(dataset))
