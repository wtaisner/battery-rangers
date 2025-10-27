"""TabPFN training pipeline module."""
import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from tabpfn import load_fitted_tabpfn_model, save_fitted_tabpfn_model
from tabpfn_extensions import interpretability

from modules.predictor.training_and_evaluation.model_factory import Models
from modules.predictor.training_and_evaluation.training_pipeline import ModelTrainingPipeline

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TabPFNTrainingPipeline(ModelTrainingPipeline):
    """
    Cross-validation pipeline class for sklearn models.
    """

    def tune_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model: object, param_grid: dict | None) -> object:
        return None

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> tuple:
        """
        Train the model.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: tuple with results, scores, explanations, model parameters.
        """
        proper_model_name, model, param_grid, f_importance_method = Models().get_model(model_name, model_path=model_path)
        self.init_scores()

        if self.verbose:
            print(f"Training model {proper_model_name}")

        model_params = []

        for i, fold in enumerate(self.folds):
            train_idx, test_idx = fold

            # train-test split
            X_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)

            model.fit(X_train.to_numpy(), y_train[y_train.columns[0]].to_numpy())

            y_pred = model.predict(X_test.to_numpy()).flatten()

            f_imp = self.calculate_f_importance(model, f_importance_method, X_test, X_train)

            # model eval
            y_test_numpy = y_test.to_numpy().flatten()
            y_pred_eval = self.eval_model(y_pred, y_test_numpy)

            baseline = np.median(y_train[y_train.columns[0]].to_numpy()) * np.ones_like(y_test_numpy)
            baselines = self.eval_model(baseline, y_test_numpy)

            self.update_scores(y_pred_eval, baselines, f_imp)
            if len(self.save_dir) > 0:
                self.save_model(model, fold_num=i)
            model_params.append(model.get_params())

        results = self.aggregate_scores()
        self.save_results(results, proper_model_name, model_params)
        return results, self.scores, self.feature_importance, model_params

    def save_model(self, model: object, fold_num: int = None) -> object:
        model_save_dir = os.path.join(self.save_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        save_model_path = os.path.join(model_save_dir, f"model_{fold_num}.tabpfn_fit")
        model.best_model_.device = "cpu"
        save_fitted_tabpfn_model(model.best_model_, save_model_path)

    def load_model(self, fold_num: int = None) -> object:
        model_save_dir = os.path.join(self.save_dir, "models")
        save_model_path = os.path.join(model_save_dir, f"model_{fold_num}.tabpfn_fit")
        reg_cpu = load_fitted_tabpfn_model(save_model_path, device="cpu")
        return reg_cpu

    def calculate_f_importance(self, model: object, method: str, X_test: pd.DataFrame, X_train: pd.DataFrame) -> tuple:
        """
        Calculate feature importance using SHAP.
        :param model: model to be explained.
        :param method: type of SHAP explainer.
        :param X_test: test data.
        :param X_train: train data.
        :return: tuple with method and SHAP values.
        """
        shap_values = interpretability.shap.get_shap_values(
            estimator=model,
            test_x=X_test.to_numpy(),
            attribute_names=X_test.columns.tolist(),
            background=X_train.to_numpy(),
        )
        return method, shap_values
