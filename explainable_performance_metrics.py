import math
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, r2_score


class ExplainablePerformanceMetrics:
    """
    A list of explainable metrics
    """

    def __init__(self):
        pass

    # COMBINED METRICS #

    @staticmethod
    def harmonic_mean(data,
                      y_true,
                      y_pred,
                      performance_metric,
                      explainability_metric,
                      beta: float = 1):
        """
        F_{beta} score of the explainability and performance metric
        """
        exp_score = explainability_metric(data)
        per_score = performance_metric(y_true, y_pred)
        try:
            return (1 + beta * beta) * per_score * exp_score / (beta * beta * per_score + exp_score)
        except:
            return 0

    @staticmethod
    def mean(data,
             y_true,
             y_pred,
             performance_metric,
             explainability_metric,
             beta: float = 0.5):
        """
        Weighted average of the explainability and performance metric
        """
        return beta * explainability_metric(data) + (1-beta) * performance_metric(y_true, y_pred)

    # END - COMBINED METRICS #

    # EXPLAINABILITY METRICS #

    # ideas taken from: Rosenfeld, A., Better Metrics for Evaluating Explainable Artificial Intelligence, AAMAS. 2021.

    @staticmethod
    def rosenfeld_f_metric(data):
        return len(list(data))

    @staticmethod
    def rosenfeld_d_metric(y_true,
                           y_pred_black_box,
                           y_pred_white_box,
                           performance_metric):
        return performance_metric(y_true, y_pred_black_box) - performance_metric(y_true, y_pred_white_box)

    @staticmethod
    def rosenfeld_r_metric(model):
        if type(model) in [DecisionTreeRegressor, DecisionTreeClassifier]:
            return ExplainablePerformanceMetrics._get_tree_rules_count(tree=model)
        elif type(model) in [KNeighborsClassifier]:
            return model.n_neighbors
        elif type(model) in [LogisticRegression, LinearRegression]:
            return 1
        elif False:
            pass
            # TODO: add all the SKLEARN models here (check out lazy_predict module for the list of models)
        else:
            raise Exception("We do not support rosenfeld_r_metric for the provided model. "
                            "Make sure it is possible to count the rules in this model")

    @staticmethod
    def rosenfeld_s_metric(features_original, features_after_boosting):
        return len(list(set(features_original).intersection(set(features_after_boosting)))) / len(list(set(features_original).union(set(features_after_boosting))))

    # TODO: think on ideas with https://github.com/TeamHG-Memex/eli5

    # END - EXPLAINBABILITY METRICS #

    # PERFORMANCE METRICS #

    @staticmethod
    def accuracy(y_true,
                 y_pred):
        try:
            return accuracy_score(y_true, y_pred)
        except:
            try:
                return accuracy_score([round(val) for val in y_true],
                                      [round(val) for val in y_pred])
            except:
                return 0

    @staticmethod
    def recall(y_true,
               y_pred):
        try:
            return recall_score(y_true, y_pred)
        except:
            try:
                return recall_score([round(val) for val in y_true],
                                    [round(val) for val in y_pred])
            except:
                return 0

    @staticmethod
    def precision(y_true,
                  y_pred):
        try:
            return precision_score(y_true, y_pred)
        except:
            try:
                return precision_score([round(val) for val in y_true],
                                       [round(val) for val in y_pred])
            except:
                return 0

    @staticmethod
    def r2(y_true,
           y_pred):
        try:
            return r2_score(y_true, y_pred)
        except:
            try:
                return r2_score([round(val) for val in y_true],
                                [round(val) for val in y_pred])
            except:
                return 0

    # END - PERFORMANCE METRICS #

    # HELP FUNCTIONS #

    @staticmethod
    def _get_tree_rules_count(tree):
        """
        Rules are the number of nodes without the leaves nodes
        as the leaves nodes are the classes and not classification rules
        """
        return tree.tree_.node_count - tree.tree_.n_leaves

    # END - HELP FUNCTIONS #

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<ExplainablePerformanceMetrics>"
