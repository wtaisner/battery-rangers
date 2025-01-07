"""Functions for train and eval."""
import os
from datetime import datetime
from typing import Literal

import pandas as pd

from modules.predictor.data.utils import prepare_data_for_regressors
from modules.predictor.training_and_evaluation.eval import cv_eval
from modules.predictor.training_and_evaluation.model_factory import get_model
from modules.predictor.training_and_evaluation.train import param_search, train_and_save_model


def batch_train_and_eval(df: pd.DataFrame, df_name: str, folds: list, target: str, features: list, opt_type: Literal["grid_search", "bayesian_search"], save_dir: str) -> dict:
    """
    Performs parameter tuning, cross-validation and model trainng on full-data. Save the model.
    :param df: dataframe with generated features
    :param df_name: data name
    :param folds: list with cross-validation folds
    :param target: name of the target variable
    :param features: list of features' names
    :param opt_type: type of hyperparameter search
    :param save_dir: directory to save model and results
    :return: dictionary with results
    """
    models = ["knn", "xgboost", "random_forest"]
    model_results = dict(zip(models, [] * len(models)))

    for m in models:
        print(f"Training model {m}")
        model_name, model, param_grid = get_model(m)

        results = cv_eval(model, model_name, folds, df, features, target, df_name, os.path.join(save_dir, f"results_{model_name}.txt"), hyperparam_opt=(opt_type, param_grid), verbose=True)
        model_results[m] = results
        date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        best_score, best_params = param_search(model, df, features, target, folds, param_grid, opt_type)
        print(f"Best score: {best_score}")
        print(f"Best params: {best_params}")
        model.set_params(**best_params)
        train_and_save_model(model, df, features, target, os.path.join(save_dir, f"{m}_{date}.pkl"), verbose=True)
        print("=======================================================================")
    return model_results


def dataset_preprocess_and_train(dataset: pd.DataFrame, dataset_name: str, cat_features: list, num_features: list, folds: list, target: str) -> dict:
    """
    Preprocess the dataset and train the models.
    :param dataset: dataframe with the dataset.
    :param dataset_name: name of the dataset
    :param cat_features: list with categorical features.
    :param num_features: list with numerical features.
    :param folds: list with cross-validation folds.
    :param target: name of the target column.
    :return: dictionary with results
    """
    if "smiles" in dataset.columns:
        dataset.drop(columns=["smiles"], inplace=True)

    if len(cat_features) > 0 or len(num_features) > 0:
        dataset = prepare_data_for_regressors(dataset, num_features, cat_features)
    features = [f for f in dataset if f not in [target]]

    date = datetime.today().strftime("%d-%m-%Y")
    save_dir = f"../../../results/{dataset_name}/{date}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    results = batch_train_and_eval(dataset, dataset_name, folds, target, features, "grid_search", save_dir)

    return results
