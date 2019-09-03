import lightgbm as lgbm
from sklearn.model_selection import GridSearchCV
import logging
import logging.config
import pickle
from src.utils import singleton
from src.config import LightGBMConfig


logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


@singleton
class LightGBM:
    def __init__(self):
        self.params = {}
        self.classif_lgbm = lgbm.LGBMClassifier(objective='multiclass',
                                                num_leaves=31,
                                                learning_rate=0.05,
                                                n_estimators=40)
        self.logger = logging.getLogger('LightGBM')

    def set_parameters(self, params):
        self.classif_lgbm.set_params(**params)
        self.logger.info('set lightGBM parameters SUCCESS !')

    def grid_search(self, X_train, y_train, param_grid):
        gbm = GridSearchCV(self.classif_lgbm, param_grid)
        gbm.fit(X_train, y_train)
        self.logger.info('grid search finished SUCCESS !')
        self.params = gbm.best_params_
        return self.params

    def classif_lgbm_train(
            self,
            X_train,
            y_train,
            X_test,
            y_test,
            eval_metric='multi_logloss',
            early_stop_r=3):
        self.classif_lgbm.fit(X_train, y_train,
                              eval_set=[(X_test, y_test)],
                              eval_metric=eval_metric,
                              early_stopping_rounds=early_stop_r)
        self.logger.info('train lgm classifier SUCCESS !')

    def classif_lgbm_predict(self, X_values):
        y_values = self.classif_lgbm.predict(
            X_values, num_iteration=self.classif_lgbm.best_iteration_)
        self.logger.info('lgbm classifier predict SUCCESS !')
        return y_values

    def classif_lgbm_predic_prob(self, X_values):
        y_prob = self.classif_lgbm.predict_proba(X_values,
                                                 raw_score=False,
                                                 num_iteration=None,
                                                 pred_leaf=False,
                                                 pred_contrib=False)
        self.logger.info('lgbm classifier predict prob SUCCESS !')
        return y_prob

    def save_model(self, model_file):
        with open(model_file, 'wb') as file:
            pickle.dump(self.classif_lgbm, file)
        self.logger.info('Save model SUCCESS! ')

    def load_model(self, model_file):
        with open(model_file, 'rb') as file:
            self.classif_lgbm = pickle.load(file)
            self.logger.info('load model SUCCESS! ')
        return self.classif_lgbm


def init_light_gbm(lgbm_config: LightGBMConfig):
    logger = logging.getLogger('init_light_gbm')
    lgbm_model = LightGBM()
    lgbm_model.load_model(lgbm_config.model_file)
    logger.info('init light gbm SUCCESS !')


def light_gbm_predict(X):
    lgbm_model = LightGBM()
    predict_value = lgbm_model.classif_lgbm_predic_prob(X)
    return predict_value[:, 1]
