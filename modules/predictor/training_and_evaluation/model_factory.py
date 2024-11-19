"""Definitions of models and models' parameters."""
import pickle
from typing import Literal

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


def get_trained_model(model_path: str) -> object:
    """
    Loads trained model saved as a pickle file.
    :param model_path: path to the saved model (pickle format)
    :return: trained model
    """
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


def _get_knn() -> tuple:
    """
    :return: knn model and its parameter grid
    """
    knn = KNeighborsRegressor(n_jobs=-1)
    knn_params = {"n_neighbors": [1, 3, 5]}
    return knn, knn_params


def _get_xgboost() -> tuple:
    """
    :return: xgboost model and its parameter grid
    """
    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    xgb_params = {
        "learning_rate": (0.05, 0.10, 0.15),
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.2],
        "colsample_bytree": [0.3, 0.4],
    }
    return xgb, xgb_params


def _get_rf() -> tuple:
    """
    :return: random forest model and its parameter grid
    """
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_params = {"n_estimators": [10, 15, 25, 40, 50, 75, 100], "max_depth": [None, 1, 2, 3, 5, 10], "min_samples_split": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 4], "bootstrap": [True, False]}
    return rf, rf_params


def get_model(name: Literal["knn", "xgboost", "random_forest"], model_path: str | None = None) -> tuple:
    """
    Loads a selected model.
    :param name: name of the model, one of 'knn', 'xgboost', 'random_forest'
    :param model_path: path to the saved model, if necessary (if this path is given, only a trained mdoel and a name of the model will be returned)
    :return: "proper" model name, model and a parameter grid
    """
    dict_name = {"knn": "KNN Regressor", "xgboost": "XGBoost Regressor", "random_forest": "Random Forest Regressor"}
    dict_function = {"knn": _get_knn, "xgboost": _get_xgboost, "random_forest": _get_rf}
    if model_path is not None:
        model = get_trained_model(model_path)
        return dict_name[name], model, None
    model, params = dict_function[name]()
    return dict_name[name], model, params
