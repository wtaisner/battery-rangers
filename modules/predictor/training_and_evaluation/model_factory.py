"""Definitions of models and models' parameters."""
import pickle
from typing import Callable

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Lasso, QuantileRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from tabpfn_extensions import AutoTabPFNRegressor
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
            model, params, dict_name, shap_version = self._models[name]()
            return dict_name, model, params, shap_version
        raise ValueError(f"Model '{name}' is not defined.")


@Models.register("knn")
def _get_knn() -> tuple:
    """
    :return: knn model, its parameter grid and name
    """
    knn = KNeighborsRegressor(n_jobs=-1)
    knn_params = lambda: {"n_neighbors": [3, 5, 7]}
    return knn, knn_params, "KNN Regressor", "kernel"


@Models.register("xgboost")
def _get_xgboost() -> tuple:
    """
    :return: xgboost model, its parameter grid and name
    """
    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    xgb_params = lambda: {
        "n_estimators": (5, 100),
        "learning_rate": [0.05, 0.10, 0.15],
        "max_depth": list(range(5, 16)) + [None],
        "min_child_weight": [3, 5, 7],
        "gamma": [0.0, 0.1, 0.2],
        "colsample_bytree": [0.2, 0.3, 0.4],
        "objective": ["rank:pairwise"],
    }
    return xgb, xgb_params, "XGBoost Regressor", "tree"


@Models.register("random_forest")
def _get_rf() -> tuple:
    """
    :return: random forest model, its parameter grid and name
    """
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_params = lambda: {
        "n_estimators": (5, 100),
        "max_depth": list(range(5, 16)) + [None],
        "min_samples_split": (2, 5),
        "min_samples_leaf": (1, 5),
        "bootstrap": [True, False],
    }
    return rf, rf_params, "Random Forest Regressor", "tree"


@Models.register("lasso")
def _get_lasso() -> tuple:
    """
    :return: lasso model, its parameter grid and name
    """
    lasso = Lasso(random_state=42)
    lasso_params = lambda: {
        "alpha": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [1000, 5000, 10000],
    }
    return lasso, lasso_params, "Lasso Regressor", "coefficients"


@Models.register("quantile")
def _get_quantile() -> tuple:
    """
    :return: quantile regression model, its parameter grid and name
    """
    quantile = QuantileRegressor()
    quantile_params = lambda: {"alpha": [0.01, 0.1, 1.0, 10.0]}
    return quantile, quantile_params, "Quantile Regressor", "coefficients"


@Models.register("bayes")
def _get_bayes() -> tuple:
    """
    :return:
    """
    bayes = BayesianRidge()
    bayes_params = lambda: {}
    return bayes, bayes_params, "Bayesian Ridge Regressor", "coefficients"


@Models.register("mlp")
def _get_mlp() -> tuple:
    """
    :return: mlp model, its parameter grid and name
    """
    mlp = MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)
    mlp_params = lambda: {
        "loss": ["squared_error", "poisson"],
        "activation": ["relu", "tanh"],
        "learning_rate": ["constant", "adaptive"],
        "learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
        "hidden_layer_sizes": ["(8,)", "(16,)", "(32,)", "(64,)", "(8, 8)", "(16, 8)", "(16, 16)", "(32, 16)", "(64, 32)", "(8, 8, 8)", "(16, 16, 16)", "(32, 32, 32)"],
        "solver": ["sgd", "adam"],
        "early_stopping": [True, False],
        "max_iter": [200, 500, 1000, 2000],
    }
    return mlp, mlp_params, "MLP Regressor", "kernel"


@Models.register("tabpfn")
def _get_tabfn() -> tuple:
    """
    :return: tabpfn model, its parameter grid and name
    """
    model = AutoTabPFNRegressor(max_time=120, device="cuda")
    return model, None, "TabPFN Regressor", "kernel"


def get_trained_model(model_path: str) -> object:
    """
    Loads trained model saved as a pickle file.
    :param model_path: path to the saved model (pickle format)
    :return: trained model
    """
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model
