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
        try:
            model = DecisionTreeClassifier().fit(x, y)
            select_model = SelectFromModel(model, prefit=True)
            data = select_model.transform(x)
            feature_idx = select_model.get_support()
            feature_name = x.columns[feature_idx]
            cols = pd.DataFrame(data=data,
                                columns=feature_name)
            re_fitted_model = DecisionTreeClassifier()
            if len(list(cols)) > 0:
                re_fitted_model.fit(cols, y)
            else:
                re_fitted_model = None
            if return_model:
                return cols, re_fitted_model
            return cols
        except:
            return [], None

    @staticmethod
    def lasso(x: pd.DataFrame,
              y: Union[pd.DataFrame, pd.Series],
              return_model: bool = True):
        """
        # TODO: add here later
        """
        try:
            features = list(x)
            model = Lasso()
            model.fit(x, y)
            coefficients = model.coef_
            importance = np.abs(coefficients)
            cols = x[list(np.array(features)[importance > 0])]
            re_fitted_model = Lasso()
            if len(list(cols)) > 0:
                re_fitted_model.fit(cols, y)
            else:
                re_fitted_model = None
            if return_model:
                return cols, re_fitted_model
            return cols
        except:
            return [], None

    @staticmethod
    def linear_svc(x: pd.DataFrame,
                   y: Union[pd.DataFrame, pd.Series],
                   return_model: bool = True):
        """
        # TODO: add here later
        """
        try:
            model = LinearSVC(penalty="l1", dual=False).fit(x, y)
            select_model = SelectFromModel(model, prefit=True)
            data = select_model.transform(x)
            feature_idx = select_model.get_support()
            feature_name = x.columns[feature_idx]
            cols = pd.DataFrame(data=data,
                                columns=feature_name)
            re_fitted_model = LinearSVC(penalty="l1", dual=False)
            if len(list(cols)) > 0:
                re_fitted_model.fit(cols, y)
            else:
                re_fitted_model = None
            if return_model:
                return cols, re_fitted_model
            return cols
        except:
            return [], None

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
        try:
            p_values = list(chi2(x, y)[1])
            pass_columns = [index for index in range(len(p_values)) if p_values[index] < alpha]
            pass_columns_p = [p_values[index] for index in range(len(p_values)) if p_values[index] < alpha]
            answer = x.iloc[:, pass_columns]
            if is_ranking:
                p_value_sum = sum(pass_columns_p)
                return answer, [value / p_value_sum if p_value_sum > 0 else 0 for value in pass_columns_p]
            return answer
        except Exception as error:
            print("Error at FeatureSelectionAlgorithms.chi_square, saying: {}".format(error))
            return x

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
        try:
            # get scores for SU
            scores = su_measure(x, y)
            # return only columns above threshold
            return x[[col for index, col in enumerate(x.columns) if scores[index] >= threshold]]
        except Exception as error:
            print("Error at FeatureSelectionAlgorithms.symmetrical_uncertainty, saying: {}".format(error))
            return x

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
        try:
            # get scores for IG
            scores = information_gain(x, y)
            # return only columns above threshold
            return x[[col for index, col in enumerate(x.columns) if scores[index] >= threshold]]
        except Exception as error:
            print("Error at FeatureSelectionAlgorithms.information_gain, saying: {}".format(error))
            return x

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
        try:
            # get Pearson correlation scores
            scores = pearson_corr(x, y)
            # return only top_k features (above threshold)
            return x[[col for index, col in enumerate(x.columns) if scores[index] >= threshold]]
        except Exception as error:
            print("Error at FeatureSelectionAlgorithms.pearson_correlation, saying: {}".format(error))
            return x

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
        try:
            # get Pearson correlation scores
            scores = spearman_corr(x, y)
            # return only top_k features (above threshold)
            return x[[col for index, col in enumerate(x.columns) if scores[index] >= threshold]]
        except Exception as error:
            print("Error at FeatureSelectionAlgorithms.spearman_correlation, saying: {}".format(error))
            return x

    @staticmethod
    def remove_low_variance(x: pd.DataFrame,
                            y: Union[pd.DataFrame, pd.Series],
                            remove_threshold: float = 1,
                            is_normalized: bool = True):
        """
        Returns only features with variance greater than remove_threshold.
        :param x: feature vars
        :param y: target var
        :param remove_threshold: minimal variance required
        :param is_normalized: use StandardScaler prior to calculation
        """
        try:
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
        except Exception as error:
            print("Error at FeatureSelectionAlgorithms.remove_low_variance, saying: {}".format(error))
            return x

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
        try:
            missing_value_ratio = x.isna().mean(axis=0)
            return x[[col for index, col in enumerate(x.columns) if missing_value_ratio[index] <= remove_threshold]]
        except Exception as error:
            print("Error at FeatureSelectionAlgorithms.missing_value_ratio, saying: {}".format(error))
            return x

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
        try:
            # get Fisher's scores
            scores = f_ratio_measure(x, y)
            # return only columns above threshold
            return x[[col for index, col in enumerate(x.columns) if scores[index] >= threshold]]
        except Exception as error:
            print("Error at FeatureSelectionAlgorithms.fishers_score, saying: {}".format(error))
            return x

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
        try:
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
        except Exception as error:
            print("Error at FeatureSelectionAlgorithms.fishers_score, saying: {}".format(error))
            return x

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
        # TODO: replace the top_k by some threshold logic
        try:
            model = SVC(kernel=kernel)
            model.fit(x,
                      y)
            return RFE(model, n_features_to_select=top_k, step=1).transform(x)
        except Exception as error:
            print("Error at FeatureSelectionAlgorithms.support_vector_machines_recursive_feature_elimination, saying: {}".format(error))
            return x

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
