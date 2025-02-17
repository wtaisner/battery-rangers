"""Explanation of the model."""
from typing import Iterable

import pandas as pd
import shap

# pylint: disable=invalid-name


def explain_model(model: object, X_test: pd.DataFrame) -> Iterable:
    """
    Explain the model.
    :param model: prediction model.
    :param X_test: test data.
    :return: shap values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values
