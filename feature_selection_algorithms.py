class FeatureSelectionAlgorithms:
    """
    A collection of feature selection algorithms
    """

    @staticmethod
    def chi_square(data):
        pass

    @staticmethod
    def symmetrical_uncertainty(data):
        pass

    @staticmethod
    def relief(data):
        pass

    @staticmethod
    def information_gain(data):
        pass

    @staticmethod
    def support_vector_machines_recursive_feature_elimination(data):
        """
        From: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2451-4
        Sanz, H., Valim, C., Vegas, E., Oller, J. M., Reverter, F., SVM-RFE: selection and visualization of the most relevant features through non-linear kernels, BMC bioinformatics, 2018.
        """
        pass

    @staticmethod
    def pearson_correlation(data,
                            top_k: int = 10):
        pass

    @staticmethod
    def remove_low_variance(data,
                            remove_threshold: float):
        pass

    @staticmethod
    def missing_value_ratio(data,
                            remove_threshold: float):
        pass

    @staticmethod
    def fishers_score(data):
        pass

    @staticmethod
    def mutual_information(data):
        pass

    @staticmethod
    def permutation_feature_importance(data,
                                       model,
                                       n_repeats: int,
                                       mean_remove_threshold: float,
                                       std_remove_threshold: float):
        pass
        # TODO: learn more at: https://scikit-learn.org/stable/modules/permutation_importance.html

    # SAME NAME CALLS #

    @staticmethod
    def sym(data):
        return FeatureSelectionAlgorithms.symmetrical_uncertainty(data=data)

    @staticmethod
    def ig(data):
        return FeatureSelectionAlgorithms.information_gain(data=data)

    @staticmethod
    def svmrfe(data):
        return FeatureSelectionAlgorithms.support_vector_machines_recursive_feature_elimination(data=data)

    # END - SAME NAME CALLS #