"""TabPFN training pipeline module."""
import copy

import numpy as np
import pandas as pd
import shap

from modules.predictor.training_and_evaluation.model_factory import Models
from modules.predictor.training_and_evaluation.training_pipeline import ModelTrainingPipeline


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
        :return: tuple with results, scores, explanations.
        """
        proper_model_name, model, param_grid, f_importance_method = Models().get_model(model_name, model_path=model_path)
        self.init_scores()

        if self.verbose:
            print(f"Training model {proper_model_name}")

        for i, fold in enumerate(self.folds):
            train_idx, test_idx = fold

            # train-test split
            X_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)

            model.fit(X_train.to_numpy(), y_train[y_train.columns[0]].to_numpy())

            y_pred = model.predict(X_test.to_numpy()).flatten()

            f_imp = self.calculate_f_importance(model, f_importance_method, X_train, X_test)

            # model eval
            y_test_numpy = y_test.to_numpy().flatten()
            y_pred_eval = self.eval_model(y_pred, y_test_numpy)

            baseline = np.median(y_train[y_train.columns[0]].to_numpy()) * np.ones_like(y_test_numpy)
            baselines = self.eval_model(baseline, y_test_numpy)

            self.update_scores(y_pred_eval, baselines, f_imp)
            if len(self.save_dir) > 0:
                self.save_model(model, fold_num=i)

        results = self.aggregate_scores()
        model_params = model.get_params()
        self.save_results(results, proper_model_name, model_params)
        return results, self.scores, self.feature_importance

    def calculate_f_importance(self, model: object, method: str, X_test: pd.DataFrame, X_train: pd.DataFrame) -> tuple:
        """
        Calculate feature importance using SHAP.
        :param model: model to be explained.
        :param method: type of SHAP explainer.
        :param X_test: test data.
        :param X_train: train data.
        :return: tuple with method and SHAP values.
        """
        explainer = shap.KernelExplainer(model.predict, X_train.to_numpy())
        shap_values = explainer.shap_values(X_test.to_numpy())
        return method, shap_values
