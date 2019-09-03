# encoding: utf-8
from abc import abstractmethod, ABCMeta
from configparser import ConfigParser
from src.utils import singleton
import logging.config
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


class BaseConfig(metaclass=ABCMeta):
    def get_member_vars(self):
        return vars(self).keys()

    @abstractmethod
    def load(self, conf, section):
        pass


@singleton
class ServerConfig(BaseConfig):
    def __init__(self):
        self.ip = '0.0.0.0'
        self.port = '5000'

    def load(self, conf, section):
        for item in self.get_member_vars():
            if item in conf.options(section):
                self.__dict__[item] = conf.get(section, item)


@singleton
class EsConfig(BaseConfig):
    def __init__(self):
        self.ip = '127.0.0.1'
        self.port = '9200'
        self.es_file = 'faq_vec.index'

    def load(self, conf, section):
        for item in self.get_member_vars():
            if item in conf.options(section):
                self.__dict__[item] = conf.get(section, item)


@singleton
class AnnoyConfig(BaseConfig):
    def __init__(self):
        self.annoy_file = 'test.annoy'
        self.vec_dim = 100

    def load(self, conf, section):
        for item in self.get_member_vars():
            if item in conf.options(section):
                if item == 'annoy_file':
                    self.__dict__[item] = conf.get(section, item)
                else:
                    self.__dict__[item] = conf.getint(section, item)


# @singleton
# class TfidfTransformerConfig(BaseConfig):
#     def __init__(self):
#         self.model_file = 'tfidftransformer.pkl'
#         self.max_feature = 256
#         self.feature_dims = 100
#
#     def load(self, conf, section):
#         for item in self.get_member_vars():
#             if item in conf.options(section):
#                 if item == 'model_file':
#                     self.__dict__[item] = conf.get(section, item)
#                 else:
#                     self.__dict__[item] = conf.getint(section, item)

@singleton
class SkipConfig(BaseConfig):
    def __init__(self):
        self.model_file = './sentence_embedding/saved_models/skip-best'
        self.dict_file = './sentence_embedding/data/faq.txt.pkl'
        self.vec_dim = 120

    def load(self, conf, section):
        for item in self.get_member_vars():
            if item in conf.options(section):
                if item == 'model_file' or item == 'dict_file':
                    self.__dict__[item] = conf.get(section, item)
                else:
                    self.__dict__[item] = conf.getint(section, item)


@singleton
class LightGBMConfig(BaseConfig):
    def __init__(self):
        self.model_file = 'lightgbm_Model.pkl'

    def load(self, conf, section):
        for item in self.get_member_vars():
            if item in conf.options(section):
                if item == 'model_file':
                    self.__dict__[item] = conf.get(section, item)
                else:
                    self.__dict__[item] = conf.getint(section, item)


@singleton
class TermRetrievalConfig(BaseConfig):
    def __init__(self):
        self.top_n = 20
        self.threshold = 2.0

    def load(self, conf, section):
        for item in self.get_member_vars():
            if item == 'threshold':
                self.__dict__[item] = conf.getfloat(section, item)
            else:
                self.__dict__[item] = conf.getint(section, item)


@singleton
class SemanticRetrievalConfig(BaseConfig):
    def __init__(self):
        self.top_n = 20

    def load(self, conf, section):
        for item in self.get_member_vars():
            if item in conf.options(section):
                self.__dict__[item] = conf.getint(section, item)


@singleton
class RankConfig(BaseConfig):
    def __init__(self):
        self.top_n = 5
        self.threshold = 0.5

    def load(self, conf, section):
        for item in self.get_member_vars():
            if item == 'threshold':
                self.__dict__[item] = conf.getfloat(section, item)
            else:
                self.__dict__[item] = conf.getint(section, item)

@singleton
class AbcnnConfig(BaseConfig):
    def __init__(self):
        self.model_file = 'abcnn2.ckpt'

    def load(self, conf, section):
        for item in self.get_member_vars():
            if item in conf.options(section):
                if item == 'model_file':
                    self.__dict__[item] = conf.get(section, item)
                else:
                    self.__dict__[item] = conf.getint(section, item)


@singleton
class XgboostConfig(BaseConfig):
    def __init__(self):
        self.model_file = 'Xgboost_train_Model_abcnn_zi.pkl'

    def load(self, conf, section):
        for item in self.get_member_vars():
            if item in conf.options(section):
                if item == 'model_file':
                    self.__dict__[item] = conf.get(section, item)
                else:
                    self.__dict__[item] = conf.getint(section, item)



@singleton
class FaqConfig:
    def __init__(self, config_file):
        self.logger = logging.getLogger('Config')
        self.conf = ConfigParser()
        self.config_file = config_file
        self.conf.read(self.config_file)

        self.server = ServerConfig()
        self.elastic_search = EsConfig()
        self.annoy_search = AnnoyConfig()
        self.skip_embedding = SkipConfig()
        # self.tfidf_transformer = TfidfTransformerConfig()
        self.term_retrieval = TermRetrievalConfig()
        self.semantic_retrieval = SemanticRetrievalConfig()
        self.rank = RankConfig()
        self.lightgbm = LightGBMConfig()
        self.abcnn = AbcnnConfig()
        self.xgboost = XgboostConfig()

    def get_member_vars(self):
        return vars(self).keys()

    def load(self):
        sections = self.conf.sections()
        for item in self.get_member_vars():
            if item in ['logger', 'conf', 'config_file']:
                continue
            if item in sections:
                self.__dict__[item].load(self.conf, item)
        self.logger.info('load faq config SUCCESS !')

    def save(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            self.conf.write(f)
        self.logger.info('save faq config SUCCESS !')

    def add(self, section, option, value):
        if section not in self.conf.sections():
            self.conf.add_section(section)
        self.conf.set(section, option, value)

    def set(self, section, option, value):
        if section not in self.conf.sections():
            self.logger.error('ERROR: NO ' + section + ' in sections !')
        else:
            self.conf.set(section, option, value)

    def remove(self, section, option=None):
        if section not in self.conf.sections():
            self.logger.error('ERROR: NO ' + section + ' in sections !')
        else:
            if option is None:
                self.conf.remove_section(section)
            else:
                self.conf.remove_option(section, option)

    def check_annoy_search(self):
        if self.annoy_search.vec_dim != self.skip_embedding.vec_dim:
            raise IndexError(
                'dim of annoy vector DOES NOT match dim of embedding model !')


def init_faq_config(config_file):
    logger = logging.getLogger('init_faq_config')
    faq_config = FaqConfig(config_file)
    faq_config.load()
    logger.info('init faq config SUCCESS !')
    return faq_config


if __name__ == '__main__':

    cf = FaqConfig('faq.config')
    cf.load()
    cf.set('rank', 'top_n', '5')
    cf.save()
