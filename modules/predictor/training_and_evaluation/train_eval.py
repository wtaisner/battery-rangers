"""Cross-validation pipeline."""
import copy
import os
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from modules.predictor.data.utils import prepare_data_for_regressors
from modules.predictor.training_and_evaluation.evaluation_metrics import EvalMetrics
from modules.predictor.training_and_evaluation.model_factory import Models
from modules.predictor.training_and_evaluation.train_utils import feature_search, param_search, train_and_save_model

# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes


class CrossValidationPipeline:
    """
    Cross-validation pipeline class.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        numerical_features: list,
        categorical_features: list,
        folds: list,
        metrics: list,
        save_dir: str,
        data_name: str,
        hyperparam_opt: tuple[bool, Literal["grid_search", "bayesian_search"]] | None = None,
        feature_selection: tuple[bool, list] | None = None,
        verbose: bool = False,
    ):
        """
        Initialize the cross-validation pipeline.
        :param X: dataframe with features.
        :param y: dataframe with target variable.
        :param numerical_features: list with numerical features.
        :param categorical_features: list with categorical features.
        :param folds: list with cross-validation folds.
        :param metrics: list with metrics to evaluate.
        :param save_scores_path: path to save scores.
        :param data_name: name of the dataset.
        :param hyperparam_opt: tuple of type of hyperparameter optimization and parameter grid, None if none optimization should be performed.
        :param feature_selection: tuple with boolean value whether to perform feature selection and list of features to keep.
        :param verbose: whether to print model scores.
        """
        self.X = X
        self.y = y
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.folds = folds
        self.data_name = data_name
        self.save_dir = save_dir
        self.hyperparam_opt = hyperparam_opt
        self.feature_selection = feature_selection
        self.verbose = verbose
        self.metrics = metrics
        self.scores = None

    def preprocess_data(self, X_train: pd.DataFrame | None, X_test: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the data.
        :param X_train: training data.
        :param X_test: test data.
        :return: preprocessed data.
        """
        if X_train is not None and "smiles" in X_train.columns:
            X_train.drop(columns=["smiles"], inplace=True)

        if X_test is not None and "smiles" in X_test.columns:
            X_test.drop(columns=["smiles"], inplace=True)

        if len(self.categorical_features) > 0 or len(self.numerical_features) > 0:
            if X_test is not None and X_train is not None:
                X_test_copy = copy.deepcopy(X_test)
                X_test = prepare_data_for_regressors(X_test, (X_train, X_test_copy), self.numerical_features, self.categorical_features)
                X_train = prepare_data_for_regressors(X_train, (X_train, X_test_copy), self.numerical_features, self.categorical_features)
            elif X_train is not None:
                X_train = prepare_data_for_regressors(X_train, (X_train, X_train), self.numerical_features, self.categorical_features)
        return X_train, X_test

    def select_features(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, model: object) -> tuple:
        """
        Perform feature selection.
        :param X_train: training data.
        :param y_train: target variable data.
        :param X_test: test data.
        :param model: prediction model.
        """
        if self.feature_selection is not None:
            fixed_features = self.feature_selection[1]
            selected_features = feature_search(model, X_train, y_train, 0.6, fixed_features)
            X_train = X_train.loc[:, selected_features]
            X_test = X_test.loc[:, selected_features]
            selected_features = [f for f in selected_features if f not in fixed_features]
        else:
            selected_features = None
        return X_train, X_test, selected_features

    def tune_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model: object, param_grid: dict | None) -> object:
        """
        Perform model tuning.
        :param X_train: training data.
        :param y_train: target variable data.
        :param model: prediction model.
        :param param_grid: dictionary with parameter grid.
        :return: model with optimized parameters.
        """
        if self.hyperparam_opt is not None:
            perform_opt, type_opt = self.hyperparam_opt
            if perform_opt and param_grid is not None:
                best_score, best_params = param_search(model, X_train, y_train, 0.6, param_grid, type_opt)
                model.set_params(**best_params)
                if self.verbose:
                    print(f"Best score: {best_score}\n Best params: {best_params}")
        return model

    def eval_model(self, y_pred: np.array, y_test: np.array) -> dict:
        """
        Evaluate the model.
        :param y_pred: predicted values.
        :param y_test: true values.
        :return: dictionary with evaluation metrics.
        """
        partial_scores = {}
        for metric in self.metrics:
            partial_scores[metric] = EvalMetrics().evaluate(metric, y_test, y_pred)
        return partial_scores

    def init_scores(self):
        """
        Initialize scores dictionary.
        """
        scores = {}
        for metric in self.metrics:
            scores[metric] = []
            scores[f"baseline_mean_{metric}"] = []
            scores[f"baseline_median_{metric}"] = []
        if self.feature_selection is not None:
            if self.feature_selection[0]:
                scores["selected_features"] = []
        self.scores = scores

    def update_scores(self, model_scores: dict, baseline_mean_scores: dict, baseline_median_scores: dict, selected_features: list | None = None):
        """
        Update scores dictionary.
        :param model_scores: dictionary with model scores.
        :param baseline_mean_scores: dictionary with baseline mean scores.
        :param baseline_median_scores: dictionary with baseline median scores.
        :param selected_features: list with selected features.
        """
        for metric in self.metrics:
            self.scores[metric].append(model_scores[metric])
            self.scores[f"baseline_mean_{metric}"].append(baseline_mean_scores[metric])
            self.scores[f"baseline_median_{metric}"].append(baseline_median_scores[metric])
        if self.feature_selection is not None:
            selected_features = "(" + ",".join(sorted(selected_features)) + ")"
            self.scores["selected_features"].append(selected_features)

    def aggregate_scores(self):
        """
        Aggregate scores.
        :return: aggregated scores.
        """
        results = {}
        for metric in self.metrics:
            metric_scores = self.scores[metric]
            results[metric] = round(sum(metric_scores) / len(metric_scores), 4)
        if self.feature_selection is not None:
            results["selected_features"] = "-".join(self.scores["selected_features"])
        else:
            results["selected_features"] = "all"
        return results

    def save_results(self, results: dict, model_name: str, model_params: Any):
        """
        Save results to a file.
        :param results: dictionary with results.
        :param model_name: name of the model.
        :param model_params: model parameters.
        """
        save_scores_path = os.path.join(self.save_dir, f"results_{model_name}.txt")
        metric_lines = [f"{metric}: {score}\n" for metric, score in results.items()]
        lines = [
            "\n******************************************************\n",
            f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n",
            f"{model_name}\n" f"Model parameters: {model_params}\n",
            f"Training data: {self.data_name}\n",
            *metric_lines,
            "******************************************************\n",
        ]
        if self.verbose:
            for l in lines:
                print(l)
        with open(save_scores_path, "a", encoding="utf-8") as f:
            f.writelines(lines)

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> dict:
        """
        Train the model.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: dictionary with results.
        """
        proper_model_name, model, param_grid = Models().get_model(model_name, model_path=model_path)
        self.init_scores()

        if self.verbose:
            print(f"Training model {proper_model_name}")

        for fold in self.folds:
            train_idx, test_idx = fold

            # train-test split
            X_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)  # pylint: disable=invalid-name
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)  # pylint: disable=invalid-name
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)

            # data preprocessing
            X_train, X_test = self.preprocess_data(X_train, X_test)

            # feature selection
            X_train, X_test, selected_features = self.select_features(X_train, y_train, X_test, model)

            # model tuning
            model = self.tune_model(X_train, y_train, model, param_grid)

            # model training
            model.fit(X_train, y_train[y_train.columns[0]])
            y_pred = model.predict(X_test).flatten()

            # model eval
            y_test_numpy = y_test.to_numpy().flatten()
            baseline_median = np.array([y_train.median()] * len(y_test_numpy)).flatten()
            baseline_mean = np.array([y_train.mean()] * len(y_test_numpy)).flatten()
            y_pred_eval = self.eval_model(y_pred, y_test_numpy)
            baseline_median_eval = self.eval_model(baseline_median, y_test_numpy)
            baseline_mean_eval = self.eval_model(baseline_mean, y_test_numpy)

            self.update_scores(y_pred_eval, baseline_mean_eval, baseline_median_eval, selected_features)

        results = self.aggregate_scores()
        self.save_results(results, proper_model_name, model.get_params())

        return results

    def batch_train_and_eval(self, models: list) -> dict:
        """
        Train and evaluate models.
        :param models: list with models' names.
        :return: models' results.
        """
        model_results = {}
        for model_name in models:
            results = self.train_pipeline(model_name)
            model_results[model_name] = results

            X = copy.deepcopy(self.X)
            y = copy.deepcopy(self.y)
            date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

            _, model, param_grid = Models().get_model(model_name)

            X, _ = self.preprocess_data(X, None)
            model = self.tune_model(X, y, model, param_grid)
            train_and_save_model(model, X, y, os.path.join(self.save_dir, f"{model_name}_{date}.pkl"), verbose=True)
            print("=======================================================================")
        return model_results
