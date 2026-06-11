"""Cross-validation pipeline."""

import copy
import os
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from modules.predictor.data.oversampling import smoter
from modules.predictor.data.utils import prepare_data_for_regressors
from modules.predictor.training_and_evaluation.evaluation_metrics import EvalMetrics
from modules.predictor.training_and_evaluation.explanations import explain_model
from modules.predictor.training_and_evaluation.model_factory import Models
from modules.predictor.training_and_evaluation.train_utils import (
    feature_search,
    param_search,
    train_and_save_model,
)

# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes


class CrossValidationPipeline:
    """Cross-validation pipeline class."""

    def __init__(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        numerical_features: list,
        categorical_features: list,
        folds: list,
        metrics: list,
        save_dir: str,
        data_name: str,
        oversampling: bool = False,
        explainability: bool = False,
        hyperparam_opt: tuple[bool, Literal["grid_search", "bayesian_search"]] | None = None,
        feature_selection: tuple[bool, list] | None = None,
        verbose: bool = False,
    ):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        if oversampling:
            self.X, _ = self.preprocess_data(x, None, num_features=self.numerical_features)
        else:
            self.X, _ = self.preprocess_data(
                x,
                None,
                num_features=self.numerical_features,
                cat_features=self.categorical_features,
            )
        self.X = x
        self.oversampling = oversampling
        self.y = y
        self.folds = folds
        self.data_name = data_name
        self.save_dir = save_dir
        self.hyperparam_opt = hyperparam_opt
        self.feature_selection = feature_selection
        self.verbose = verbose
        self.metrics = metrics
        self.explainability = explainability
        self.scores: dict[str, list[Any]] = {}
        self.shap_values: list[Any] = []

    def preprocess_data(
        self,
        x_train: pd.DataFrame | None,
        x_test: pd.DataFrame | None,
        cat_features: list | None = None,
        num_features: list | None = None,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Preprocess the data.
        :param num_features: numerical features.
        :param cat_features: categorical features
        :param X_train: training data.
        :param X_test: test data.
        :return: preprocessed data.
        """
        if x_train is not None and "smiles" in x_train.columns:
            x_train.drop(columns=["smiles"], inplace=True)

        if x_test is not None and "smiles" in x_test.columns:
            x_test.drop(columns=["smiles"], inplace=True)

        cat_features = cat_features if cat_features is not None else []
        num_features = num_features if num_features is not None else []

        if len(cat_features) > 0 or len(num_features) > 0:
            if x_test is not None and x_train is not None:
                x_test_copy = copy.deepcopy(x_test)
                x_test = prepare_data_for_regressors(
                    x_test,
                    (x_train, x_test_copy),
                    numerical_features=num_features,
                    categorical_features=cat_features,
                )
                x_train = prepare_data_for_regressors(x_train, (x_train, x_test_copy), num_features, cat_features)
            elif x_train is not None:
                x_train = prepare_data_for_regressors(x_train, (x_train, x_train), num_features, cat_features)
        return x_train, x_test

    def oversampler(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform oversampling.
        :param X_train: training data.
        :param y_train: target variable data.
        :return: oversampled data.
        """
        if self.oversampling:
            try:
                numeric_cols = [i for i, col in enumerate(x_train.columns) if col in self.numerical_features]
                x_train_oversampled, y_train_oversampled = smoter(
                    x_train.values,
                    y_train.values.flatten(),
                    te=0.5,
                    o=300,
                    k=5,
                    numeric_cols=numeric_cols,
                    oversampling_type="SMOTER",
                )
                x_train = pd.DataFrame(x_train_oversampled, columns=x_train.columns)
                y_train = pd.DataFrame(y_train_oversampled, columns=y_train.columns)
            except ValueError:
                print("Oversampling failed")
        return x_train, y_train

    def select_features(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        model: object,
    ) -> tuple:
        """Perform feature selection.
        :param X_train: training data.
        :param y_train: target variable data.
        :param X_test: test data.
        :param model: prediction model.
        """
        if self.feature_selection is not None and self.feature_selection[0]:
            fixed_features = self.feature_selection[1]
            selected_features = feature_search(model, x_train, y_train, 0.6, fixed_features)
            x_train = x_train.loc[:, selected_features]
            x_test = x_test.loc[:, selected_features]
            selected_features = [f for f in selected_features if f not in fixed_features]
        else:
            selected_features = None
        return x_train, x_test, selected_features

    def tune_model(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        model: object,
        param_grid: dict | None,
    ) -> object:
        """Perform model tuning.
        :param X_train: training data.
        :param y_train: target variable data.
        :param model: prediction model.
        :param param_grid: dictionary with parameter grid.
        :return: model with optimized parameters.
        """
        if self.hyperparam_opt is not None:
            perform_opt, type_opt = self.hyperparam_opt
            if perform_opt and param_grid is not None:
                best_score, best_params = param_search(model, x_train, y_train, 0.6, param_grid, type_opt)
                model.set_params(**best_params)
                if self.verbose:
                    print(f"Best score: {best_score}\n Best params: {best_params}")
        return model

    def eval_model(self, y_pred: np.array, y_test: np.array) -> dict:
        """Evaluate the model.
        :param y_pred: predicted values.
        :param y_test: true values.
        :return: dictionary with evaluation metrics.
        """
        partial_scores = {}
        for metric in self.metrics:
            partial_scores[metric] = EvalMetrics().evaluate(metric, y_test, y_pred)
        return partial_scores

    def init_scores(self):
        """Initialize scores dictionary."""
        scores = {}
        for metric in self.metrics:
            scores[metric] = []
            scores[f"baseline_mean_{metric}"] = []
            scores[f"baseline_median_{metric}"] = []
        if self.feature_selection is not None:
            if self.feature_selection[0]:
                scores["selected_features"] = []
        self.scores = scores
        self.shap_values = []

    def update_scores(
        self,
        model_scores: dict,
        baseline_mean_scores: dict,
        baseline_median_scores: dict,
        selected_features: list | None = None,
    ):
        """Update scores dictionary.
        :param model_scores: dictionary with model scores.
        :param baseline_mean_scores: dictionary with baseline mean scores.
        :param baseline_median_scores: dictionary with baseline median scores.
        :param selected_features: list with selected features.
        """
        for metric in self.metrics:
            self.scores[metric].append(model_scores[metric])
            self.scores[f"baseline_mean_{metric}"].append(baseline_mean_scores[metric])
            self.scores[f"baseline_median_{metric}"].append(baseline_median_scores[metric])
        if self.feature_selection is not None and selected_features is not None:
            selected_features = "(" + ",".join(sorted(selected_features)) + ")"
            self.scores["selected_features"].append(selected_features)

    def aggregate_scores(self):
        """Aggregate scores.
        :return: aggregated scores.
        """
        results = {}
        for metric in self.metrics:
            metric_scores = self.scores[metric]
            baseline_mean = self.scores[f"baseline_mean_{metric}"]
            baseline_median = self.scores[f"baseline_median_{metric}"]
            results[metric] = round(sum(metric_scores) / len(metric_scores), 4)
            results[f"baseline_mean_{metric}"] = round(sum(baseline_mean) / len(baseline_mean), 4)
            results[f"baseline_median_{metric}"] = round(sum(baseline_median) / len(baseline_median), 4)
        if self.feature_selection is not None:
            results["selected_features"] = "-".join(self.scores.get("selected_features", []))
        else:
            results["selected_features"] = "all"
        return results

    def save_results(self, results: dict, model_name: str, model_params: Any):
        """Save results to a file.
        :param results: dictionary with results.
        :param model_name: name of the model.
        :param model_params: model parameters.
        """
        save_scores_path = os.path.join(self.save_dir, f"results_{model_name}.txt")
        metric_lines = [f"{metric}: {score}\n" for metric, score in results.items()]
        lines = [
            "\n******************************************************\n",
            f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n",
            f"{model_name}\nModel parameters: {model_params}\n",
            f"Training data: {self.data_name}\n",
            *metric_lines,
            "******************************************************\n",
        ]
        if self.verbose:
            for line in lines:
                print(line)
        with open(save_scores_path, "a", encoding="utf-8") as f:
            f.writelines(lines)

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> dict:
        """Train the model.
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
            x_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            x_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)

            # data preprocessing
            if self.oversampling:
                x_train, x_test = self.preprocess_data(x_train, x_test, cat_features=self.categorical_features)

            # feature selection
            x_train, x_test, selected_features = self.select_features(x_train, y_train, x_test, model)

            # model tuning
            model = self.tune_model(x_train, y_train, model, param_grid)

            if self.oversampling:
                x_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
                y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
                x_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
                y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)
                x_train, y_train = self.oversampler(x_train, y_train)
                x_train, x_test = self.preprocess_data(x_train, x_test, cat_features=self.categorical_features)
                x_train = x_train.astype(float)
                x_test = x_test.astype(float)

            # model training
            model.fit(x_train, y_train[y_train.columns[0]])

            if self.explainability:
                shap_values = explain_model(model, x_test)
                self.shap_values.append((shap_values, test_idx))

            y_pred = model.predict(x_test).flatten()

            # model eval
            y_test_numpy = y_test.to_numpy().flatten()
            baseline_median = np.array([y_train.median()] * len(y_test_numpy)).flatten()
            baseline_mean = np.array([y_train.mean()] * len(y_test_numpy)).flatten()
            y_pred_eval = self.eval_model(y_pred, y_test_numpy)
            baseline_median_eval = self.eval_model(baseline_median, y_test_numpy)
            baseline_mean_eval = self.eval_model(baseline_mean, y_test_numpy)

            self.update_scores(y_pred_eval, baseline_mean_eval, baseline_median_eval, selected_features)

        results = self.aggregate_scores()

        if len(self.save_dir) > 0:
            self.save_results(results, proper_model_name, model.get_params())

        return results

    def batch_train_and_eval(self, models: list) -> dict:
        """Train and evaluate models.
        :param models: list with models' names.
        :return: models' results.
        """
        model_results = {}
        for model_name in models:
            results = self.train_pipeline(model_name)
            model_results[model_name] = results

            x = copy.deepcopy(self.X)
            y = copy.deepcopy(self.y)
            date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

            _, model, param_grid = Models().get_model(model_name)

            if self.oversampling:
                x, _ = self.preprocess_data(x, None, cat_features=self.categorical_features)

            model = self.tune_model(x, y, model, param_grid)
            train_and_save_model(
                model,
                x,
                y,
                os.path.join(self.save_dir, f"{model_name}_{date}.pkl"),
                verbose=self.verbose,
            )
            print("=======================================================================")
        return model_results
