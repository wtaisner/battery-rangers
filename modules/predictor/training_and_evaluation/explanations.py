"""Explanation of the model."""

from typing import Iterable

import pandas as pd
import shap

# pylint: disable=invalid-name


def explain_model(model: object, x_test: pd.DataFrame) -> Iterable:
    """Explain the model.
    :param model: prediction model.
    :param x_test: test data.
    :return: shap values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    return shap_values
