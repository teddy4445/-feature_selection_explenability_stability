# library imprts
import numpy
from scipy import stats


class FeatureSelectionSetsDifferences:
    """
    A collection of feature sets distances
    """

    def __init__(self):
        pass

    # Generalized to list of features rather than pair #

    @staticmethod
    def generalize_to_list(metric, feature_set_list):
        scores = [[metric(feature_set_list[i], feature_set_list[j])
                   for j in range(i + 1, len(feature_set_list))]
                  for i in range(len(feature_set_list) - 1)]
        return numpy.nanmean(scores), numpy.nanstd(scores)

    # end - Generalized to list of features rather than pair #

    @staticmethod
    def iou(original_set, new_set):
        """
        The classical intersection over union metric
        """
        if len(original_set) == len(new_set) and len(new_set) == 0:
            raise Exception("Both original and new feature sets are empty, cannot compute the IOU metric")
        return len(list(set(original_set).intersection(set(new_set)))) / len(
            list(set(original_set).union(set(new_set))))

    # ideas from Kalousis, A., Prados, J., Hilario, M., Stability of Feature Selection Algorithms: a study on high dimensional spaces, Knowledge and Information Systems, 2007.

    @staticmethod
    def weighted_features_sets(original_set_weights, new_set_weights):
        """
        Normalized pearson correlation between the two sets
        """
        if len(original_set_weights) != len(new_set_weights):
            raise Exception(
                "The original and new feature sets have different length, make sure they are both in the same size")
        return (numpy.cov(original_set_weights, new_set_weights) + 1) / 2

    @staticmethod
    def ranked_features_sets(original_set_rank,
                             new_set_rank):
        """
        Normalized pearsman correlation between the two sets
        """
        if len(original_set_rank) != len(new_set_rank):
            raise Exception(
                "The original and new feature sets have different length, make sure they are both in the same size")
        return (stats.spearmanr(original_set_rank, new_set_rank, nan_policy="raise") + 1) / 2

    @staticmethod
    def tanimoto_distance(original_set,
                          new_set):
        """
        The classical intersection over union metric
        """
        if len(original_set) == len(new_set) and len(new_set) == 0:
            raise Exception("Both original and new feature sets are empty, cannot compute the tanimoto distance metric")
        return (len(original_set) + len(new_set) - 2 * len(list(set(original_set).intersection(set(new_set))))) \
               / (len(original_set) + len(new_set) - len(list(set(original_set).union(set(new_set)))))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<FeatureSelectionSetsDifferences>"
