# library imports
import numpy as np
import pandas as pd
from typing import Union
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import chi2, SelectFromModel, RFE, VarianceThreshold
from ITMO_FS.filters.univariate import su_measure, information_gain, pearson_corr, spearman_corr, f_ratio_measure


class FeatureSelectionAlgorithms:
    """
    A collection of feature selection algorithms
    """

    # CONSTS #
    RANDOM_STATE = 73
    # END - CONSTS #

    ### EMBEDDING ###

    def __init__(self):
        pass

    @staticmethod
    def decision_tree(x: pd.DataFrame,
                      y: Union[pd.DataFrame, pd.Series],
                      return_model: bool = True):
        """
        # TODO: add here later
        """
        model = DecisionTreeClassifier().fit(x, y)
        cols = SelectFromModel(model, prefit=True).transform(x)
        if return_model:
            return cols, model
        return cols

    @staticmethod
    def lasso(x: pd.DataFrame,
              y: Union[pd.DataFrame, pd.Series],
              return_model: bool = True):
        """
        # TODO: add here later
        """
        features = list(x)
        model = Lasso()
        model.fit(x, y)
        coefficients = model.named_steps['model'].coef_
        importance = np.abs(coefficients)
        cols = list(np.array(features)[importance > 0])
        if return_model:
            return cols, model
        return cols


    @staticmethod
    def linear_svc(x: pd.DataFrame,
                   y: Union[pd.DataFrame, pd.Series],
                   return_model: bool = True):
        """
        # TODO: add here later
        """
        lsvc = LinearSVC(penalty="l1", dual=False).fit(x, y)
        model = SelectFromModel(lsvc, prefit=True)
        cols = list(model.transform(x))
        if return_model:
            return cols, model
        return cols

    ### END - EMBEDDING ###

    ### FILTER ###

    @staticmethod
    def chi_square(x: pd.DataFrame,
                   y: Union[pd.DataFrame, pd.Series],
                   alpha: float = 0.05,
                   is_ranking: bool = False):
        """
        Return all columns that pass the chi2 test with p-value of less than alpha
        :param x: feature vars
        :param y: target var
        :param alpha: threshold for using the feature var
        :param is_ranking: define if returning the subset with normalized p-values or not
        :return: sub set of 'x' (and scores as the normalized p-values)
        """
        p_values = list(chi2(x, y)[1])
        pass_columns = [index for index in p_values if p_values[index] < alpha]
        pass_columns_p = [p_values[index] for index in p_values if p_values[index] < alpha]
        answer = x.iloc[:, pass_columns]
        if is_ranking:
            p_value_sum = sum(pass_columns_p)
            return answer, [value / p_value_sum if p_value_sum > 0 else 0 for value in pass_columns_p]
        return answer

    @staticmethod
    def symmetrical_uncertainty(x: pd.DataFrame,
                                y: Union[pd.DataFrame, pd.Series],
                                threshold: float = 0):
        """
        Return only columns with symmetrical uncertainty (related to the targets) greater than "threshold".
        :param x: feature vars
        :param y: target var
        :param threshold: threshold for using the feature var
        """
        # get scores for SU
        scores = su_measure(x, y)
        # return only columns above threshold
        return x[[col for index, col in enumerate(x.columns) if scores[index] >= threshold]]

    @staticmethod
    def information_gain(x: pd.DataFrame,
                         y: Union[pd.DataFrame, pd.Series],
                         threshold: float = 0):
        """
        Return only columns with information gain score (related to the targets) greater than "threshold".
        :param x: feature vars
        :param y: target var
        :param threshold: threshold for using the feature var
        """
        # get scores for IG
        scores = information_gain(x, y)
        # return only columns above threshold
        return x[[col for index, col in enumerate(x.columns) if scores[idx] >= threshold]]

    @staticmethod
    def pearson_correlation(x: pd.DataFrame,
                            y: Union[pd.DataFrame, pd.Series],
                            threshold: float = 0):
        """
        Return only the k features with the highest Pearson Correlation (related to the targets).
        :param x: feature vars
        :param y: target var
        :param threshold: threshold for using the feature
        """
        # get Pearson correlation scores
        scores = pearson_corr(x, y)
        # return only top_k features (above threshold)
        return x[[col for index, col in enumerate(x.columns) if scores[index] >= threshold]]

    @staticmethod
    def spearman_correlation(x: pd.DataFrame,
                             y: Union[pd.DataFrame, pd.Series],
                             threshold: float = 0):
        """
        Return only the k features with the highest Spearman Correlation (related to the targets).
        :param x: feature vars
        :param y: target var
        :param threshold: threshold for using the feature
        """
        # get Pearson correlation scores
        scores = spearman_corr(x, y)
        # return only top_k features (above threshold)
        return x[[col for index, col in enumerate(x.columns) if scores[index] >= threshold]]

    @staticmethod
    def remove_low_variance(x: pd.DataFrame,
                            y: Union[pd.DataFrame, pd.Series],
                            remove_threshold: float,
                            is_normalized: bool = True):
        """
        Returns only features with variance greater than remove_threshold.
        :param x: feature vars
        :param y: target var
        :param remove_threshold: minimal variance required
        :param is_normalized: use StandardScaler prior to calculation
        """
        # Initialize estimator with required threshold
        vt = VarianceThreshold(threshold=remove_threshold)
        # Normalize if required
        if is_normalized:
            x_scaled = preprocessing.StandardScaler().fit_transform(x)
            vt.fit(x_scaled)
        else:
            vt.fit(x)
        # Return features with variance equal or greater than remove_threshold
        mask = vt.get_support()
        selected = x.loc[:, mask]
        return selected

    @staticmethod
    def missing_value_ratio(x: pd.DataFrame,
                            y: Union[pd.DataFrame, pd.Series],
                            remove_threshold: float = 0.05):
        """
        Returns only features with number of missing values lower than remove_threshold.
        :param x: feature vars
        :param y: target var
        :param remove_threshold: maximal ratio of missing values [0 to 1] (default = 0.05)
        """
        missing_value_ratio = x.isna().mean(axis=0)
        return x[[col for index, col in enumerate(x.columns) if missing_value_ratio[index] <= remove_threshold]]

    @staticmethod
    def fishers_score(x: pd.DataFrame,
                      y: Union[pd.DataFrame, pd.Series],
                      threshold: float = 0):
        """
        Return only columns with Fisher's score (related to the targets) greater than "threshold".
        :param x: feature vars
        :param y: target var
        :param threshold: threshold for using the feature var
        """
        # get Fisher's scores
        scores = f_ratio_measure(x, y)
        # return only columns above threshold
        return x[[col for index, col in enumerate(x.columns) if scores[index] >= threshold]]

    @staticmethod
    def permutation_feature_importance(x: pd.DataFrame,
                                       y: Union[pd.DataFrame, pd.Series],
                                       model,
                                       n_repeats: int,
                                       mean_remove_threshold: float,
                                       std_remove_threshold: float):
        """
        # TODO: add here later
        """
        # Fit model
        x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=FeatureSelectionAlgorithms.RANDOM_STATE)
        fitted_model = model.fit(x_train, y_train)

        # Get importances mean and std
        scores = permutation_importance(fitted_model,
                                        x,
                                        y,
                                        scoring=None,
                                        n_repeats=n_repeats,
                                        random_state=FeatureSelectionAlgorithms.RANDOM_STATE,
                                        sample_weight=None)
        return x[[col for index, col in enumerate(x.columns) if (scores.importances_mean[index] >= mean_remove_threshold and
                                                                 scores.importances_std[index] < std_remove_threshold)]]

    @staticmethod
    def support_vector_machines_recursive_feature_elimination(x: pd.DataFrame,
                                                              y: Union[pd.DataFrame, pd.Series],
                                                              top_k: int = 10,
                                                              kernel="linear"):
        """
        From: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2451-4
        Sanz, H., Valim, C., Vegas, E., Oller, J. M., Reverter, F., SVM-RFE: selection and visualization of the most relevant features through non-linear kernels, BMC bioinformatics, 2018.
        Return the top_k important features after performing SVM recursively.
        :param x: feature vars
        :param y: target var
        :param top_k: number of features to return
        :param kernel: kernel to use in the SVM
        """
        return RFE(SVC(kernel=kernel), n_features_to_select=top_k, step=1).transform(x)

    # SAME NAME CALLS #

    @staticmethod
    def sym(x, y):
        return FeatureSelectionAlgorithms.symmetrical_uncertainty(x=x,
                                                                  y=y)

    @staticmethod
    def ig(x, y):
        return FeatureSelectionAlgorithms.information_gain(x=x,
                                                           y=y)

    @staticmethod
    def svmrfe(x, y):
        return FeatureSelectionAlgorithms.support_vector_machines_recursive_feature_elimination(x=x,
                                                                                                y=y)

    # END - SAME NAME CALLS #

    ### END - FILTER ###

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<FeatureSelectionAlgorithms>"
