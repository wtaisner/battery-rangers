"""Definitions of models and models' parameters."""
import pickle
from typing import Callable

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class Models:
    """
    Class for ML models.
    """

    _models: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Register an ML model with a given name.

        This method is used as a decorator to register an ML model
        under a specified name. The registered function can later be retrieved
        and used to retrieve and train the model.

        :param name: The name to register the ML model under.
        :return: A decorator that registers the ML model.
        """

        def decorator(func: Callable) -> Callable:
            cls._models[name] = func
            return func

        return decorator

    def get_model(self, name: str, model_path: str | None = None) -> tuple:
        """
        Loads a selected model.
        :param name: name of the model, registered in the class.
        :param model_path: path to the saved model, if necessary (if this path is given, only a trained model and a name of the model will be returned).
        :return: "proper" model name, model and a parameter grid
        """
        if name in self._models:
            if model_path is not None:
                model = get_trained_model(model_path)
                _, _, dict_name = self._models[name]()
                return dict_name, model, None
            model, params, dict_name = self._models[name]()
            return dict_name, model, params
        raise ValueError(f"Model '{name}' is not defined.")


@Models.register("knn")
def _get_knn() -> tuple:
    """
    :return: knn model, its parameter grid and name
    """
    knn = KNeighborsRegressor(n_jobs=-1)
    knn_params = {"n_neighbors": [3, 5, 7]}
    return knn, knn_params, "KNN Regressor"


@Models.register("xgboost")
def _get_xgboost() -> tuple:
    """
    :return: xgboost model, its parameter grid and name
    """
    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    xgb_params = {
        "n_estimators": [5, 15, 25, 50],
        "learning_rate": [0.01, 0.05, 0.15, 0.2],
        "max_depth": [3, 5, 8, None],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.2],
        "colsample_bytree": [0.1, 0.2, 0.3],
    }
    return xgb, xgb_params, "XGBoost Regressor"


@Models.register("random_forest")
def _get_rf() -> tuple:
    """
    :return: random forest model, its parameter grid and name
    """
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_params = {"n_estimators": [5, 15, 25, 50], "max_depth": [None, 3, 5, 8], "min_samples_split": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 4], "bootstrap": [True, False]}
    return rf, rf_params, "Random Forest Regressor"


@Models.register("lasso")
def _get_lasso() -> tuple:
    """
    :return: lasso model, its parameter grid and name
    """
    lasso = Lasso(random_state=42)
    lasso_params = {"alpha": [0.1, 0.5, 1, 2, 5, 10, 20]}
    return lasso, lasso_params, "Lasso Regressor"


@Models.register("adaboost")
def _get_adaboost() -> tuple:
    """
    :return: AdaBoost model, its parameter grid and name
    """

    adaboost = AdaBoostRegressor(random_state=42)
    adaboost_params = {
        "n_estimators": [5, 15, 25, 50],
        "learning_rate": [0.01, 0.05, 0.15, 0.2],
        "estimator": [
            DecisionTreeRegressor(max_depth=3, random_state=42),
            DecisionTreeRegressor(max_depth=5, random_state=42),
            DecisionTreeRegressor(max_depth=8, random_state=42),
            DecisionTreeRegressor(max_depth=None, random_state=42),
        ],
    }
    return adaboost, adaboost_params, "AdaBoost Regressor"


@Models.register("svm")
def _get_svm() -> tuple:
    """
    :return: SVM model, its parameter grid and name
    """
    svm = SVR(kernel="rbf")
    svm_params = {"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.1, 0.5, 1], "gamma": ["scale", "auto"]}
    return svm, svm_params, "Support Vector Regressor"


def get_trained_model(model_path: str) -> object:
    """
    Loads trained model saved as a pickle file.
    :param model_path: path to the saved model (pickle format)
    :return: trained model
    """
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model
