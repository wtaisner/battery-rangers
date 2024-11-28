"""Functions for train and eval."""
import os
from datetime import datetime
from typing import Literal

import pandas as pd

from modules.predictor.training_and_evaluation.eval import cv_eval
from modules.predictor.training_and_evaluation.model_factory import get_model
from modules.predictor.training_and_evaluation.train import param_search, train_and_save_model


def batch_train_and_eval(df: pd.DataFrame, df_name: str, folds: list, target: str, features: list, opt_type: Literal["grid_search", "bayesian_search"], save_dir: str) -> None:
    """
    Performs parameter tuning, cross-validation and model trainng on full-data. Save the model.
    :param df: dataframe with generated features
    :param df_name: data name
    :param folds: list with cross-validation folds
    :param target: name of the target variable
    :param features: list of features' names
    :param opt_type: type of hyperparameter search
    :param save_dir: directory to save model and results
    :return: None
    """
    models = ["knn", "xgboost", "random_forest"]

    for m in models:
        print(f"Training model {m}")
        model_name, model, param_grid = get_model(m)

        cv_eval(model, model_name, folds, df, features, target, df_name, os.path.join(save_dir, f"results_{model_name}.txt"), hyperparam_opt=(opt_type, param_grid), verbose=True)

        date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        best_score, best_params = param_search(model, df, features, target, folds, param_grid, opt_type)
        print(f"Best score: {best_score}")
        print(f"Best params: {best_params}")
        model.set_params(**best_params)
        train_and_save_model(model, df, features, target, os.path.join(save_dir, f"{m}_{date}.pkl"), verbose=True)
        print("=======================================================================")
