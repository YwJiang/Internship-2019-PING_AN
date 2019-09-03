from xgboost import XGBClassifier
import numpy as np
import logging
import xgboost as xgb
import pandas as pd
import pickle
from src.utils import singleton
from src.config import XgboostConfig
from sklearn.model_selection import GridSearchCV


@ singleton
class Xgboost():
    def __init__(self):
        self.params = {}
        self.classif_xgboost = xgb.XGBClassifier()
        self.logger = logging.getLogger('Xgboost')

    def set_parameters(self, params):
        self.classif_xgboost.set_params(**params)
        self.logger.info('set Xgboost parameters SUCCESS! ')

    def gird_search(self, X_train, y_train, param_grid):
        xgboost = GridSearchCV(self.classif_xgboost, param_grid)
        xgboost.fit(X_train, y_train)
        self.params = xgboost.best_params_
        return self.params

    def classif_xgboost_train(self, X_train, y_train):
        self.classif_xgboost.fit(X_train, y_train)

    def classif_xgboost_predict(self, X_values):
        y_prob = self.classif_xgboost.predict_proba(X_values)
        return y_prob

    def save_model(self, model_file):
        with open(model_file, 'wb') as file:
            pickle.dump(self.classif_xgboost, file)

    def load_model(self, model_file):
        with open(model_file, 'rb') as file:
            self.classif_xgboost = pickle.load(file)


def init_xgboost(xgboost_config: XgboostConfig):
    logger = logging.getLogger('init_xgboost')
    lgbm_model = Xgboost()
    lgbm_model.load_model(xgboost_config.model_file)
    logger.info('init init_xgboost SUCCESS!')


def xgboost_predict(X):
    xgboost_model = Xgboost()
    predict_value = xgboost_model.classif_xgboost_predict(X)
    return predict_value

