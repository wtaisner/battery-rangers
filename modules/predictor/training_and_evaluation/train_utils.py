"""Functions for model training"""
import pickle
from typing import Literal

import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

from modules.predictor.data.utils import custom_data_split
from modules.predictor.training_and_evaluation.evaluation_metrics import mape

# pylint: disable=invalid-name


def feature_search(model: object, X: pd.DataFrame, y: pd.DataFrame, train_size: float, fixed_features: list) -> list:
    """
    Performs feature selection using the provided list of features (optimization of RMSE).
    :param model: prediction model.
    :param X: dataframe with features.
    :param y: dataframe with target variable.
    :param train_size: size of the training set (proportion).
    :param fixed_features: list with features to keep.
    :return: best score and best features.
    """
    scorer = make_scorer(mape, greater_is_better=False)
    folds = custom_data_split(X, y, train_size=train_size)
    fixed_features_ids = tuple(X.columns.get_loc(f) for f in fixed_features)

    sfs = SequentialFeatureSelector(model, k_features=(len(fixed_features), len(X.columns)), forward=True, floating=False, scoring=scorer, cv=folds, n_jobs=-1, fixed_features=fixed_features_ids)
    sfs.fit(X, y[y.columns[0]])
    return sfs.k_feature_names_


def param_search(model: object, X: pd.DataFrame, y: pd.DataFrame, train_size: float, param_grid: dict, opt_method: Literal["grid_search", "bayesian_search"]) -> tuple:
    """
    Performs hyperparameter optimization using the provided parameter grid (optimization of RMSE).
    :param model: prediction model.
    :param X: dataframe with features.
    :param y: dataframe with target variable.
    :param train_size: size of the training set (proportion).
    :param param_grid: dictionary with parameter grid.
    :param opt_method: optimization method, either grid_search or bayesian_search
    :return: best score and best parameters.
    """
    scorer = make_scorer(mape, greater_is_better=False)
    folds = custom_data_split(X, y, train_size=train_size)
    if opt_method == "grid_search":
        opt = GridSearchCV(estimator=model, param_grid=param_grid, cv=folds, scoring=scorer, refit=True, n_jobs=-1, return_train_score=True)
    else:
        opt = BayesSearchCV(estimator=model, search_spaces=param_grid, cv=folds, scoring=scorer, refit=True, n_jobs=-1, return_train_score=True)

    opt.fit(X, y[y.columns[0]])
    return opt.best_score_, opt.best_params_


def train_and_save_model(model: object, X: pd.DataFrame, y: pd.DataFrame, model_save_path: str, verbose: bool = False) -> object:
    """
    Trains and saves trained model.
    :param model: prediction model.
    :param X: data for prediction.
    :param y: list with features' names.
    :param model_save_path: path to save trained model.
    :param verbose: whether to print model scores after training
    :return: trained model
    """
    model.fit(X, y[y.columns[0]])
    if verbose:
        print(model.score(X, y[y.columns[0]]))
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    return model
