# library imports
from typing import Union

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2, SelectFromModel, RFE, VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
from ITMO_FS.filters.univariate import su_measure, information_gain, pearson_corr, spearman_corr, f_ratio_measure
from sklearn.model_selection import train_test_split


class FeatureSelectionAlgorithms:
    """
    A collection of feature selection algorithms
    """

    ### EMBEDDING ###

    @staticmethod
    def decision_tree(x: pd.DataFrame,
                      y: Union[pd.DataFrame, pd.Series]):
        return SelectFromModel(DecisionTreeClassifier().fit(x, y), prefit=True).transform(x)

    @staticmethod
    def lasso(x: pd.DataFrame,
              y: Union[pd.DataFrame, pd.Series]):
        features = list(x)
        model = Lasso()
        model.fit(x, y)
        coefficients = model.named_steps['model'].coef_
        importance = np.abs(coefficients)
        return list(np.array(features)[importance > 0])

    @staticmethod
    def linear_svc(x: pd.DataFrame,
                   y: Union[pd.DataFrame, pd.Series]):
        # TODO: think on a good definition here
        lsvc = LinearSVC(penalty="l1", dual=False).fit(x, y)
        model = SelectFromModel(lsvc, prefit=True)
        return list(model.transform(x))

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
        return x[[col for idx, col in enumerate(x.columns) if scores[idx] >= threshold]]

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
        return x[[col for idx, col in enumerate(x.columns) if scores[idx] >= threshold]]

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

        # set threshold according to the top_k parameter
        # threshold = sorted(scores, reverse=True)[min(top_k - 1, len(scores) - 1)]

        # return only top_k features (above threshold)
        return x[[col for idx, col in enumerate(x.columns) if scores[idx] >= threshold]]

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

        # set threshold according to the top_k parameter
        # threshold = sorted(scores, reverse=True)[min(top_k - 1, len(scores) - 1)]

        # return only top_k features (above threshold)
        return x[[col for idx, col in enumerate(x.columns) if scores[idx] >= threshold]]

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
                            remove_threshold: float):
        """
        Returns only features with number of missing values lower than remove_threshold.
        :param x: feature vars
        :param y: target var
        :param remove_threshold: maximal ratio of missing values [0 to 1]
        """
        missing_value_ratio = x.isna().mean(axis=0)
        return x[[col for idx, col in enumerate(x.columns) if missing_value_ratio[idx] <= remove_threshold]]

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
        return x[[col for idx, col in enumerate(x.columns) if scores[idx] >= threshold]]

    @staticmethod
    def permutation_feature_importance(x: pd.DataFrame,
                                       y: Union[pd.DataFrame, pd.Series],
                                       model,
                                       n_repeats: int,
                                       mean_remove_threshold: float,
                                       # std_remove_threshold: float
                                       ):
        """
       Return only columns with feature importance greater than "threshold".
       :param x: feature vars
       :param y: target var
       :param threshold: threshold for using the feature var
       """
        # TODO: learn more at: https://scikit-learn.org/stable/modules/permutation_importance.html
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
        model = model.fit(X_train, y_train)
        r = permutation_importance(model, X_val, y_val, n_repeats=n_repeats, random_state=0)
        scores = r.importances_mean

        # return only columns above threshold
        return x[[col for idx, col in enumerate(x.columns) if scores[idx] >= mean_remove_threshold]]

    @staticmethod
    def relief(x, y):
        # TODO: do you mean STIR algorithm? reliefF?

        stir = STIR(top_k)
        trX = stir.fit_transform(x, y)
        # TODO: trX to pd.DataFrame?
        return None

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
        estimator = SVC(kernel=kernel)
        selector = RFE(estimator, n_features_to_select=top_k, step=1)
        return selector.transform(x)

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
