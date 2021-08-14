# library imports
import pandas as pd
from sklearn.model_selection import train_test_split

# project import
from feature_selection_algorithms import FeatureSelectionAlgorithms
from dataset_properties_measurements import DatasetPropertiesMeasurements
from explainable_performance_metrics import ExplainablePerformanceMetrics
from feature_selection_stability_tests import FeatureSelectionStabilityTests
from feature_selection_sets_differences import FeatureSelectionSetsDifferences


class FunctionsMapper:
    """
    This class responsible to map strings to functions in the project and run them if needed
    """

    # operative consts #
    RANDOM_STATE = 73
    TEST_SIZE = 0.2
    # end - operative consts #

    # CONSTS #

    METRICS = {"accuracy": ExplainablePerformanceMetrics.accuracy,
               "recall": ExplainablePerformanceMetrics.recall,
               "precision": ExplainablePerformanceMetrics.precision,
               "r2": ExplainablePerformanceMetrics.r2,
               "rosenfeld_f_metric": ExplainablePerformanceMetrics.rosenfeld_f_metric,
               "rosenfeld_d_metric": ExplainablePerformanceMetrics.rosenfeld_d_metric,
               "rosenfeld_r_metric": ExplainablePerformanceMetrics.rosenfeld_r_metric,
               "rosenfeld_s_metric": ExplainablePerformanceMetrics.rosenfeld_s_metric,
               "harmonic_mean": ExplainablePerformanceMetrics.harmonic_mean,
               "mean": ExplainablePerformanceMetrics.mean}

    STABILITY_TESTS = {"data": FeatureSelectionStabilityTests.data_stability_test,
                       "features": FeatureSelectionStabilityTests.feature_stability_test,
                       "lyapunov": FeatureSelectionStabilityTests.lyapunov_stability_test}

    STABILITY_METRICS = {"iou": FeatureSelectionSetsDifferences.iou,
                         "weighted_features_sets": FeatureSelectionSetsDifferences.weighted_features_sets,
                         "tanimoto_distance": FeatureSelectionSetsDifferences.tanimoto_distance,
                         "ranked_features_sets": FeatureSelectionSetsDifferences.ranked_features_sets}

    FS_FILTER = {"chi_square": FeatureSelectionAlgorithms.chi_square,
                 "symmetrical_uncertainty": FeatureSelectionAlgorithms.symmetrical_uncertainty,
                 "information_gain": FeatureSelectionAlgorithms.information_gain,
                 "pearson_correlation": FeatureSelectionAlgorithms.pearson_correlation,
                 "spearman_correlation": FeatureSelectionAlgorithms.spearman_correlation,
                 "remove_low_variance": FeatureSelectionAlgorithms.remove_low_variance,
                 "missing_value_ratio": FeatureSelectionAlgorithms.missing_value_ratio,
                 "fishers_score": FeatureSelectionAlgorithms.fishers_score,
                 "permutation_feature_importance": FeatureSelectionAlgorithms.permutation_feature_importance,
                 "support_vector_machines_recursive_feature_elimination": FeatureSelectionAlgorithms.support_vector_machines_recursive_feature_elimination}

    FS_EMBEDDING = {"decision_tree": FeatureSelectionAlgorithms.decision_tree,
                    "lasso": FeatureSelectionAlgorithms.lasso,
                    "linear_svc": FeatureSelectionAlgorithms.linear_svc}

    # END - CONSTS #

    def __init__(self):
        pass

    # runners #

    @staticmethod
    def run_explainablity_column(metric: str,
                                 fs_filter: str,
                                 fs_embedding: str,
                                 x: pd.DataFrame,
                                 y: pd.DataFrame):
        """
        Get the properties (metric, FS filter, and FS embedding) and a dataset (x,y)
        and return the value of the corresponding column after computing the right pipeline
        """
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=FunctionsMapper.TEST_SIZE,
                                                            random_state=FunctionsMapper.RANDOM_STATE)

        # TODO: the current code assumes we get complex performance and explainability metric, this won't work for the other options
        # TODO: and only for the 'rosenfeld_f_metric' explainability metric.
        # get from the MDF the combination
        metrics_components = FunctionsMapper.get_complex_metric(metric=metric)
        # run filter FS
        filter_x = FunctionsMapper.FS_FILTER[fs_filter](x=x,
                                                        y=y)
        # run embedding FS
        filter_embbeding_x, model = FunctionsMapper.FS_EMBEDDING[fs_embedding](x=filter_x,
                                                                               y=y)
        # compute the y_pred for the performance metric
        y_pred = model.predict(x_test[list(filter_embbeding_x)])

        # compute the overall score of the model
        return metrics_components[0](data=filter_embbeding_x,
                                     y_true=y_test,
                                     y_pred=y_pred,
                                     performance_metric=metrics_components[1],
                                     explainability_metric=metrics_components[2])

    @staticmethod
    def run_stability_column(stability_test: str,
                             fs_filter: str,
                             stability_metric: str,
                             x: pd.DataFrame,
                             y: pd.DataFrame):
        """
        Get the properties (metric, FS filter, and FS embedding) and a dataset (x,y)
        and return the value of the corresponding column after computing the right pipeline
        """
        return FunctionsMapper.STABILITY_TESTS[stability_test](data=x,
                                                              feature_selection_method=FunctionsMapper.FS_FILTER[fs_filter],
                                                              feature_sets_differance_metrics=FunctionsMapper.STABILITY_METRICS[stability_metric])

    # end - runners #

    # complex metric function #

    @staticmethod
    def get_complex_metric(metric: str):
        """
        Get a column name and return the metric function represented by this column
        as a list of the functions (combine, performance, explainablity)
        """
        # splitting element which is constant in the 'MetaDataTableGenerator' class
        SPLIT_ELEMENT = "__"
        # split string to elements
        elements = metric.strip().split(SPLIT_ELEMENT)
        # return answer
        return [FunctionsMapper.METRICS[element] for element in elements[::-1]]

    # end - complex metric function #
