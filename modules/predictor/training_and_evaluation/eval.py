"""Functions for model evaluation"""
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from modules.predictor.data.utils import custom_data_split
from modules.predictor.training_and_evaluation.evaluation_metrics import average_ranking_score, mape, ndcg_score, rmse
from modules.predictor.training_and_evaluation.train import param_search


def cv_eval(
    model: object,
    model_name: str,
    folds: list,
    df: pd.DataFrame,
    features: list,
    target: str,
    df_name: str,
    save_scores_path: str,
    hyperparam_opt: tuple[Literal["grid_search", "bayesian_search"], dict] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Perform cross-validation evaluation, saves the results to a given file.
    :param model: prediction model.
    :param model_name: name of the model.
    :param folds: list with k-folds.
    :param df: dataframe with features and targets.
    :param features: list of features' names.
    :param target: name of the target variable.
    :param df_name: name of the dataset
    :param save_scores_path: path to save scores.
    :param hyperparam_opt: tuple of type of hyperparameter optimization and parameter grid, None if none optimization should be performed
    :param verbose: whether to print model scores.
    :return: dictionary with results
    """
    metric_mapping = {
        "accuracy": average_ranking_score,
        "NDCG": ndcg_score,
        "MAPE": mape,
        "RMSE": rmse,
    }
    scores = {}
    for metric in metric_mapping:
        scores[metric] = []
        scores["baseline_mean_{metric}"] = []
        scores["baseline_median_{metric}"] = []

    for fold in folds:
        train_idx, test_idx = fold
        X_train = df.loc[train_idx, features]  # pylint: disable=invalid-name
        y_train = df.loc[train_idx, target]
        X_test = df.loc[test_idx, features]  # pylint: disable=invalid-name
        y_test = df.loc[test_idx, target]

        if hyperparam_opt is not None:
            type_opt, param_grid = hyperparam_opt
            opt_split = custom_data_split(df.loc[train_idx, :].reset_index(), target, train_size=0.6)
            best_score, best_params = param_search(model, df.loc[train_idx, :].reset_index(), features, target, opt_split, param_grid, type_opt)
            model.set_params(**best_params)
            if verbose:
                print(f"Best score: {best_score}\n Best params: {best_params}")

        # Model fitting
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        baseline_median = np.array([y_train.median()] * len(y_test))
        baseline_mean = np.array([y_train.mean()] * len(y_test))
        for metric, func in metric_mapping.items():
            scores[metric].append(func(y_test, y_pred))
            scores[f"baseline_mean_{metric}"].append(func(y_test, baseline_mean))
            scores[f"baseline_median_{metric}"].append(func(y_test, baseline_median))

    metric_lines = [f"{metric}: {round(sum(score) / len(score), 4)}\n" for (metric, score) in scores.items()]
    lines = [
        "\n******************************************************\n",
        f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n",
        f"{model_name}\n" f"Model parameters: {model.get_params()}\n",
        f"Training data: {df_name}\n",
        *metric_lines,
        "******************************************************\n",
    ]
    results = {metric: round(sum(score) / len(score), 4) for (metric, score) in scores.items()}
    if verbose:
        for l in lines:
            print(l)

    with open(save_scores_path, "a", encoding="utf-8") as f:
        f.writelines(lines)

    return results
