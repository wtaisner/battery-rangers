"""Functions for model training"""
import pickle
from typing import Literal

import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

from modules.predictor.training_and_evaluation.evaluation_metrics import rmse


def param_search(model: object, df: pd.DataFrame, features: list, target: str, folds: list, param_grid: dict, opt_method: Literal["grid_search", "bayesian_search"]) -> tuple:
    """
    Performs hyperparameter optimization using the provided parameter grid (optimization of RMSE).
    :param model: prediction model.
    :param df: data for prediction.
    :param features: list of features' names.
    :param target: name of the target variable.
    :param folds: list with folds.
    :param param_grid: dictionary with parameter grid.
    :param opt_method: optimization method, either grid_search or bayesian_search
    :return: best score and best parameters.
    """
    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    if opt_method == "grid_search":
        opt = GridSearchCV(estimator=model, param_grid=param_grid, cv=folds, scoring=rmse_scorer, refit=True, n_jobs=-1, return_train_score=True)
    else:
        opt = BayesSearchCV(estimator=model, search_spaces=param_grid, cv=folds, scoring=rmse_scorer, refit=True, n_jobs=-1, return_train_score=True)

    X = df.loc[:, features]  # pylint: disable=invalid-name
    y = df.loc[:, target]

    opt.fit(X, y)
    return opt.best_score_, opt.best_params_


def train_and_save_model(model: object, df_train: pd.DataFrame, features: list, target: str, model_save_path: str, verbose: bool = False) -> object:
    """
    Trains and saves trained model.
    :param model: prediction model.
    :param df_train: data for prediction.
    :param features: list with features' names.
    :param target: name of the target variable.
    :param model_save_path: path to save trained model.
    :param verbose: whether to print model scores after training
    :return: trained model
    """
    X = df_train.loc[:, features]  # pylint: disable=invalid-name
    y = df_train.loc[:, target]

    model.fit(X, y)
    if verbose:
        print(model.score(X, y))
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    return model
