# library imports
import json
import os
import pandas as pd
import numpy as np

# project imports
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from meta_data_table_generator import MetaDataTableGenerator
from explainable_performance_metrics import ExplainablePerformanceMetrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


class ExplainablePerformancePipelineAnalyzer:
    """
    A class that provides plots, CSV files with tables, and statistical outcomes of a meta-dataset of feature selection
    computed by the MetaDataTableGenerator.run class
    """

    # CONSTS #
    MAPPER_FILE_NAME = "mapper.json"
    RANDOM_STATE_VAL = 73
    K_FOLD = 5

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(data_folder_path: str,
            sub_tables_folder_path: str,
            results_folder_path: str):
        """
        # TODO: add here later
        """
        # read data from meta table file
        mdf = pd.read_csv(os.path.join(data_folder_path, MetaDataTableGenerator.FILE_NAME))

        ExplainablePerformancePipelineAnalyzer.divide_mdf_by_metric(mdf=mdf,
                                                                    sub_tables_folder_path=sub_tables_folder_path)
        ExplainablePerformancePipelineAnalyzer.index_all_classes_of_pairs_of_fs(
            sub_tables_folder_path=sub_tables_folder_path)

        ExplainablePerformancePipelineAnalyzer.train_and_test_local_classifier(
            sub_tables_folder_path=sub_tables_folder_path,
            results_folder_path=results_folder_path)

        for y_col_name in ['best_filter', 'best_embedding']:
            ExplainablePerformancePipelineAnalyzer.train_and_test_local_classifier_for_single_case(
                sub_tables_folder_path=sub_tables_folder_path,
                results_folder_path=results_folder_path,
                y_col_name=y_col_name)

    @staticmethod
    def divide_mdf_by_metric(mdf,
                             sub_tables_folder_path: str):
        """
        This method drops all irrelevant columns in the MDF and divide the
        'x' columns and the 'y' columns that starts with 'expandability-' in the dataframe.

        Afterward, it divide the dataframe to the 'x' columns and the 'y' columns with
        the results of a singular metric at a time. Each such dataframe is saved with name that
        contains the name of the metric in the 'results_folder_path' folder.

        Note: if sub-MDF file is already in place, skip it re-generation

        :return: a dictionary: keys are metric names, and the values are the obtained dataframes.
        """
        # make sure we have target folder
        try:
            os.makedirs(sub_tables_folder_path)
        except:
            pass

        # create sub-table for every metric from MetaDataTableGenerator.METRICS
        for metric in MetaDataTableGenerator.METRICS:
            # create sub_mdf using the columns with x_ and with the relevant metric
            sub_mdf = mdf[[c for c in mdf.columns if
                           c.startswith("x-") or (c.startswith("expandability-") and metric in c)]]
            sub_mdf.to_csv(os.path.join(sub_tables_folder_path, "sub_table_for_{}.csv".format(metric)))

    @staticmethod
    def index_all_classes_of_pairs_of_fs(sub_tables_folder_path):
        # TODO: fix documentation
        """
        Assumption: the MDF is obtained from the 'ExplainablePerformancePipelineAnalyzer.divide_mdf_by_metric' method.
        This method add a column to the mdf with the index of the best 'y' column (starting with 0).
        """
        # read sub_mdf
        for filename in os.listdir(sub_tables_folder_path):
            sub_mdf = pd.read_csv(os.path.join(sub_tables_folder_path, filename))
            try:
                sub_mdf.drop('Unnamed: 0', axis=1, inplace=True)
            except:
                pass

            # convert column to integers
            mapper = {value: index for index, value in
                      enumerate(list(sub_mdf[[c for c in sub_mdf.columns if not c.startswith('x-')]].columns))}
            mapper_filter = {value: index for index, value in enumerate(MetaDataTableGenerator.FS_FILTER)}
            mapper_embedding = {value: index for index, value in enumerate(MetaDataTableGenerator.FS_EMBEDDING)}
            # save mapper for later usage
            try:
                with open(os.path.join(os.path.dirname(sub_tables_folder_path),
                                       ExplainablePerformancePipelineAnalyzer.MAPPER_FILE_NAME), 'w') as mapper_file:
                    json.dump(mapper, mapper_file, indent=2)
            except:
                os.makedirs(sub_tables_folder_path)
                with open(os.path.join(os.path.dirname(sub_tables_folder_path),
                                       ExplainablePerformancePipelineAnalyzer.MAPPER_FILE_NAME), 'w') as mapper_file:
                    json.dump(mapper, mapper_file, indent=2)

            # find top3 columns
            sub_mdf['top3'] = sub_mdf[[c for c in sub_mdf.columns if not c.startswith('x-')]].apply(
                lambda s: s.nlargest(3).index.to_list(), axis=1)

            # find column with maximal value
            sub_mdf['best'] = sub_mdf[[c for c in sub_mdf.columns if not c.startswith('x-')]].drop('top3',
                                                                                                   axis=1).idxmax(
                axis="columns")

            # take the index of the column instead of the name of the pipeline
            sub_mdf['best_filter'] = sub_mdf['best'].apply(lambda x: mapper_filter[x.split("-")[-2]])
            sub_mdf['best_embedding'] = sub_mdf['best'].apply(lambda x: mapper_embedding[x.split("-")[-1]])
            sub_mdf['best'] = sub_mdf['best'].apply(lambda x: mapper[x])
            sub_mdf['top3'] = sub_mdf['top3'].apply(lambda l: [mapper[x] for x in l])
            sub_mdf.drop([c for c in sub_mdf.columns if c.startswith('expandability-')], axis=1).to_csv(
                os.path.join(sub_tables_folder_path, filename))

    @staticmethod
    def train_and_test_local_classifier(sub_tables_folder_path: str,
                                        results_folder_path: str):

        """
        Assumption: the MDF is obtained from the 'ExplainablePerformancePipelineAnalyzer.divide_mdf_by_metric' and
        'ExplainablePerformancePipelineAnalyzer.index_all_classes_of_pairs_of_fs' methods.
        This method train a list of classifiers, test them based on the "accuracy" metric and
        save the best model and its test results in files in the 'results_folder_path' folder.
        """
        try:
            os.makedirs(results_folder_path)
        except:
            pass

        full_scores_list = []
        full_confusion_matrix_list = []
        for filename in os.listdir(sub_tables_folder_path):
            sub_mdf = pd.read_csv(os.path.join(sub_tables_folder_path, filename))
            try:
                sub_mdf.drop('Unnamed: 0', axis=1, inplace=True)
            except:
                pass
            X = sub_mdf[[c for c in sub_mdf.columns if c.startswith("x-")]]
            y = sub_mdf['best']
            top3 = sub_mdf['top3'].apply(lambda s: ExplainablePerformancePipelineAnalyzer.string_to_list(s))

            # clean the data and fix anomalies
            X.replace(np.inf, np.nan, inplace=True)
            X.replace(-np.inf, np.nan, inplace=True)
            y.replace(np.inf, np.nan, inplace=True)
            y.replace(-np.inf, np.nan, inplace=True)

            cv_predictions = []
            X = X.sample(frac=1, random_state=ExplainablePerformancePipelineAnalyzer.RANDOM_STATE_VAL)
            y = y[X.index]
            top3 = top3[X.index]

            for i in range(ExplainablePerformancePipelineAnalyzer.K_FOLD):
                test_size = int(X.shape[0] / ExplainablePerformancePipelineAnalyzer.K_FOLD)
                # save index of test set
                test_index = X.iloc[(i * test_size):((i + 1) * test_size), :].index

                # define X_test, y_test
                X_test = X.loc[test_index]
                y_test = y.loc[test_index]
                # define X_train, y_train
                X_train = X.loc[~X.index.isin(test_index)]
                y_train = y.loc[~y.index.isin(test_index)]

                # run classifiers
                clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True)
                _, predictions = clf.fit(X_train, X_test, y_train, y_test)

                # set index of predictions to be test_index
                predictions.index = test_index

                cv_predictions.append(predictions)

            # create final prediction by concatenating predictions
            predictions = pd.concat(cv_predictions, axis=0)
            predictions['truth'] = y[predictions.index]

            # convert back to class names
            with open(os.path.join(os.path.dirname(sub_tables_folder_path),
                                   ExplainablePerformancePipelineAnalyzer.MAPPER_FILE_NAME), 'r') as f:
                mapper = json.load(f)

            reverse_mapper = {value: "-".join(key.split('__')[-1].split('-')[1:]) for key, value in mapper.items()}
            try:
                predictions = predictions.applymap(lambda x: reverse_mapper[x])
            except:
                predictions.dropna(inplace=True)
                predictions = predictions.applymap(lambda x: reverse_mapper[x])

            # add top3 column for prediction in order to create more metrics
            predictions['top3'] = top3[predictions.index]
            predictions['top3'] = predictions['top3'].apply(lambda l: [reverse_mapper[x] for x in l])

            predictions.to_csv(os.path.join(results_folder_path, "predictions_for_" + filename))

            # create final scores by using calculation over predictions
            scores = pd.DataFrame(columns=['Accuracy', 'f1-score', 'top3_accuracy'])

            # run over all models and evaluate the performance of each of them
            confusion_matrix_per_metric = []
            for model_name in predictions.drop(['truth', 'top3'], axis=1).columns:
                # create confusion matrix for the specific model
                y_true = pd.Series(predictions['truth'], name="Truth")
                y_pred = pd.Series(predictions[model_name], name="Predicted")
                df_confusion = pd.crosstab(y_true, y_pred).reindex(columns=list(reverse_mapper.values())).reindex(
                    list(reverse_mapper.values())).fillna(0)

                confusion_matrix_per_metric.append(df_confusion)

                # calculate metrics for all predictions
                accuracy = accuracy_score(predictions['truth'], predictions[model_name], normalize=True)
                # b_accuracy = balanced_accuracy_score(predictions['truth'], predictions[model_name])
                f1 = f1_score(predictions['truth'], predictions[model_name], average="weighted")

                # add accuracy for top3
                check_targets_top3 = predictions.apply(lambda x: x[model_name] in x['top3'], axis=1)
                top3_accuracy = check_targets_top3.mean()

                # update the scores dataframe with the relevant results
                scores.loc[model_name] = [accuracy, f1, top3_accuracy]

            # create total confusion matrix for the relevant metric using average
            total_confusion_matrix_for_current_metric = pd.concat([
                df_confusion.stack() for df_confusion in confusion_matrix_per_metric
            ], axis=1) \
                .mean(axis=1) \
                .unstack()

            total_confusion_matrix_for_current_metric.to_csv(
                os.path.join(results_folder_path, "total_confusion_matrix_for_" + filename))

            scores.sort_values(by='Accuracy', ascending=False).to_csv(
                os.path.join(results_folder_path, "scores_for_" + filename))

            # add scores of the current metric to the scores list
            full_scores_list.append(scores.sort_index())

        # create total scores (for all metrics) using average on scores and save to csv
        total_scores = pd.concat([
            scores_df.stack() for scores_df in full_scores_list
        ], axis=1) \
            .mean(axis=1) \
            .unstack() \
            .sort_values(by='Accuracy', ascending=False)

        # save to csv
        total_scores.to_csv(os.path.join(results_folder_path, "total_scores_using_average.csv"))

        ExplainablePerformanceMetrics.accuracy(y_true=[],
                                               y_pred=[])

    @staticmethod
    def train_and_test_local_classifier_for_single_case(sub_tables_folder_path: str,
                                                        results_folder_path: str,
                                                        y_col_name: str):

        """
        Assumption: the MDF is obtained from the 'ExplainablePerformancePipelineAnalyzer.divide_mdf_by_metric' and
        'ExplainablePerformancePipelineAnalyzer.index_all_classes_of_pairs_of_fs' methods.
        This method train a list of classifiers, test them based on the "accuracy" metric and
        save the best model and its test results in files in the 'results_folder_path' folder.
        """
        try:
            os.makedirs(results_folder_path)
        except:
            pass

        full_scores_list = []
        full_confusion_matrix_list = []
        for filename in os.listdir(sub_tables_folder_path):
            sub_mdf = pd.read_csv(os.path.join(sub_tables_folder_path, filename))
            try:
                sub_mdf.drop('Unnamed: 0', axis=1, inplace=True)
            except:
                pass
            X = sub_mdf[[c for c in sub_mdf.columns if c.startswith("x-")]]
            y = sub_mdf[y_col_name]

            # clean the data and fix anomalies
            X.replace(np.inf, np.nan, inplace=True)
            X.replace(-np.inf, np.nan, inplace=True)
            y.replace(np.inf, np.nan, inplace=True)
            y.replace(-np.inf, np.nan, inplace=True)

            cv_predictions = []
            X = X.sample(frac=1, random_state=ExplainablePerformancePipelineAnalyzer.RANDOM_STATE_VAL)
            y = y[X.index]

            for i in range(ExplainablePerformancePipelineAnalyzer.K_FOLD):
                test_size = int(X.shape[0] / ExplainablePerformancePipelineAnalyzer.K_FOLD)
                # save index of test set
                test_index = X.iloc[(i * test_size):((i + 1) * test_size), :].index

                # define X_test, y_test
                X_test = X.loc[test_index]
                y_test = y.loc[test_index]
                # define X_train, y_train
                X_train = X.loc[~X.index.isin(test_index)]
                y_train = y.loc[~y.index.isin(test_index)]

                # run classifiers
                clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True)
                _, predictions = clf.fit(X_train, X_test, y_train, y_test)

                # set index of predictions to be test_index
                predictions.index = test_index

                cv_predictions.append(predictions)

            # create final prediction by concatenating predictions
            predictions = pd.concat(cv_predictions, axis=0)
            predictions['truth'] = y[predictions.index]

            # convert back to class names
            with open(os.path.join(os.path.dirname(sub_tables_folder_path),
                                   ExplainablePerformancePipelineAnalyzer.MAPPER_FILE_NAME), 'r') as f:
                mapper = json.load(f)

            reverse_mapper = {value: "-".join(key.split('__')[-1].split('-')[1:]) for key, value in mapper.items()}
            try:
                predictions = predictions.applymap(lambda x: reverse_mapper[x])
            except:
                predictions.dropna(inplace=True)
                predictions = predictions.applymap(lambda x: reverse_mapper[x])

            predictions.to_csv(os.path.join(results_folder_path, "predictions_for_{}_{}".format(filename, y_col_name)))

            # create final scores by using calculation over predictions
            scores = pd.DataFrame(columns=['Accuracy', 'f1-score'])

            # run over all models and evaluate the performance of each of them
            confusion_matrix_per_metric = []
            for model_name in predictions.drop(['truth'], axis=1).columns:
                # create confusion matrix for the specific model
                y_true = pd.Series(predictions['truth'], name="Truth")
                y_pred = pd.Series(predictions[model_name], name="Predicted")
                df_confusion = pd.crosstab(y_true, y_pred).reindex(columns=list(reverse_mapper.values())).reindex(
                    list(reverse_mapper.values())).fillna(0)

                confusion_matrix_per_metric.append(df_confusion)

                # calculate metrics for all predictions
                accuracy = accuracy_score(predictions['truth'], predictions[model_name], normalize=True)
                # b_accuracy = balanced_accuracy_score(predictions['truth'], predictions[model_name])
                f1 = f1_score(predictions['truth'], predictions[model_name], average="weighted")

                # update the scores dataframe with the relevant results
                scores.loc[model_name] = [accuracy, f1]

            # create total confusion matrix for the relevant metric using average
            total_confusion_matrix_for_current_metric = pd.concat([
                df_confusion.stack() for df_confusion in confusion_matrix_per_metric
            ], axis=1) \
                .mean(axis=1) \
                .unstack()

            total_confusion_matrix_for_current_metric.to_csv(
                os.path.join(results_folder_path, "total_confusion_matrix_for_{}_{}".format(filename, y_col_name)))

            scores.sort_values(by='Accuracy', ascending=False).to_csv(
                os.path.join(results_folder_path, "scores_for_{}_{}".format(filename, y_col_name)))

            # add scores of the current metric to the scores list
            full_scores_list.append(scores.sort_index())

        # create total scores (for all metrics) using average on scores and save to csv
        total_scores = pd.concat([
            scores_df.stack() for scores_df in full_scores_list
        ], axis=1) \
            .mean(axis=1) \
            .unstack() \
            .sort_values(by='Accuracy', ascending=False)

        # save to csv
        total_scores.to_csv(os.path.join(results_folder_path, "total_scores_using_average_{}.csv".format(y_col_name)))

        ExplainablePerformanceMetrics.accuracy(y_true=[],
                                               y_pred=[])

    @staticmethod
    def string_to_list(s):
        return [int(x) for x in s[1:-1].replace(' ', '').split(",")]


if __name__ == '__main__':
    ExplainablePerformancePipelineAnalyzer.run(data_folder_path="meta_table_data",
                                               sub_tables_folder_path="sub_meta_tables",
                                               results_folder_path="classification_results")
