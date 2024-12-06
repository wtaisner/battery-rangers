"""Functions for evaluating model performance."""
import numpy as np
import pandas as pd
import sklearn


def average_ranking_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates average ranking score given predictions and ground truth labels.
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :return: Average ranking score.
    """
    df = pd.DataFrame({"Test": y_true, "Pred": y_pred})
    sorted_rank = df.sort_values(by="Test", ascending=False).reset_index(drop=True)
    sorted_rank["rank"] = sorted_rank.index + 1

    cartesian_df = sorted_rank.merge(sorted_rank, how="cross", suffixes=("_1", "_2"))
    cartesian_df = cartesian_df.loc[cartesian_df["Test_1"] != cartesian_df["Test_2"]]
    cartesian_df["target_test"] = cartesian_df["Test_1"] > cartesian_df["Test_2"]
    cartesian_df["target_pred"] = cartesian_df["Pred_1"] > cartesian_df["Pred_2"]
    cartesian_df["result"] = cartesian_df["target_test"] == cartesian_df["target_pred"]

    return cartesian_df["result"].mean()


def ndcg_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates normalized discounted cumulative gain given predictions and ground truth labels.
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :return: Normalized discounted cumulative gain.
    """
    y_true_rank = (
        y_true.rank(ascending=True)
        .to_numpy()
        .reshape(
            -1,
        )
    )
    y_pred_rank = (
        pd.DataFrame(y_pred, columns=["target"])
        .rank(ascending=True)
        .to_numpy()
        .reshape(
            -1,
        )
    )

    return sklearn.metrics.ndcg_score([y_true_rank], [y_pred_rank])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates root mean squared error given predictions and ground truth labels.
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :return: Root mean squared error.
    """
    return np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates mean absolute percentage error given predictions and ground truth labels.
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :return: Mean absolute percentage error.
    """
    return sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred)
