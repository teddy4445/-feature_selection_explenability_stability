# library imports
import pandas as pd
from sklearn.feature_selection import chi2


class FeatureSelectionAlgorithms:
    """
    A collection of feature selection algorithms
    """

    @staticmethod
    def chi_square(x: pd.DataFrame,
                   y: pd.DataFrame,
                   alpha: float = 0.05,
                   is_ranking: bool = False):
        """
        Return all colums that pass the chi2 test with p-value of less than alpha
        :param x: feature vars
        :param y: target var
        :param alpha: trashold for using the feature var
        :param is_ranking: define if returning the subset with normalized p-values or not
        :return: sub set of 'x' (and scores as the normalized p-values)
        """
        p_values = list(chi2(x, y)[1])
        pass_columns = [index for index in p_values if p_values[index] < alpha]
        pass_columns_p = [p_values[index] for index in p_values if p_values[index] < alpha]
        answer = x.iloc[:, pass_columns]
        if is_ranking:
            p_value_sum = sum(pass_columns_p)
            return answer, [value/p_value_sum if p_value_sum > 0 else 0 for value in pass_columns_p]
        return answer

    @staticmethod
    def symmetrical_uncertainty(x, y):
        pass

    @staticmethod
    def relief(x, y):
        pass

    @staticmethod
    def information_gain(x, y):
        pass

    @staticmethod
    def support_vector_machines_recursive_feature_elimination(x, y):
        """
        From: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2451-4
        Sanz, H., Valim, C., Vegas, E., Oller, J. M., Reverter, F., SVM-RFE: selection and visualization of the most relevant features through non-linear kernels, BMC bioinformatics, 2018.
        """
        pass

    @staticmethod
    def pearson_correlation(x, y,
                            top_k: int = 10):
        pass

    @staticmethod
    def remove_low_variance(x, y,
                            remove_threshold: float):
        pass

    @staticmethod
    def missing_value_ratio(x, y,
                            remove_threshold: float):
        pass

    @staticmethod
    def fishers_score(x, y):
        pass

    @staticmethod
    def mutual_information(x, y):
        pass

    @staticmethod
    def permutation_feature_importance(x, y,
                                       model,
                                       n_repeats: int,
                                       mean_remove_threshold: float,
                                       std_remove_threshold: float):
        pass
        # TODO: learn more at: https://scikit-learn.org/stable/modules/permutation_importance.html

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
