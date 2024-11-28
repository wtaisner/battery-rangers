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
) -> None:
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
    :return: None
    """
    ndcg_scores = []
    mape_scores = []
    accuracy_score = []
    rmse_scores = []
    baseline_mean_rmse = []
    baseline_median_rmse = []

    for fold in folds:
        train_idx, test_idx = fold
        if hyperparam_opt is not None:
            type_opt, param_grid = hyperparam_opt
            opt_split = custom_data_split(df.loc[train_idx, :].reset_index(), target, train_size=0.6)
            best_score, best_params = param_search(model, df.loc[train_idx, :].reset_index(), features, target, opt_split, param_grid, type_opt)
            model.set_params(**best_params)
            if verbose:
                print(f"Best score: {best_score}\n Best params: {best_params}")
        # Data split
        X_train = df.loc[train_idx, features]  # pylint: disable=invalid-name
        y_train = df.loc[train_idx, target]
        X_test = df.loc[test_idx, features]  # pylint: disable=invalid-name
        y_test = df.loc[test_idx, target]

        # Model fitting
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy_score.append(average_ranking_score(y_test, y_pred))
        ndcg_scores.append(ndcg_score(y_test, y_pred))
        rmse_scores.append(rmse(y_test, y_pred))
        mape_scores.append(mape(y_test, y_pred))
        baseline_mean_rmse.append(rmse(np.array([y_test.mean()] * len(y_test)), y_test))
        baseline_median_rmse.append(rmse(np.array([y_test.median()] * len(y_test)), y_test))

    lines = [
        "\n******************************************************\n",
        f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n",
        f"{model_name}\n" f"Model parameters: {model.get_params()}\n",
        f"Training data: {df_name}\n",
        f"NDCG score: {round(sum(ndcg_scores) / len(ndcg_scores), 2)}\n",
        f"Accuracy score: {round(sum(accuracy_score) / len(accuracy_score), 2)}\n",
        f"Mape score: {round(sum(mape_scores) / len(mape_scores), 2)}\n",
        f"RMSE score: {round(sum(rmse_scores) / len(rmse_scores), 2)}\n",
        f"Baseline RMSE mean score: {round(sum(baseline_mean_rmse) / len(baseline_mean_rmse), 2)}\n",
        f"Baseline RMSE median score: {round(sum(baseline_median_rmse) / len(baseline_median_rmse), 2)}\n" "******************************************************\n",
    ]
    if verbose:
        for l in lines:
            print(l)

    with open(save_scores_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
