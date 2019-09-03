# encoding: utf-8

from src.config import FaqConfig, TfidfTransformerConfig
import json
import pickle
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import Cutter
from src.utils import singleton
import logging.config
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


@singleton
class TfidfTransformer:
    def __init__(self, max_features=256):
        self.tfidf_transformer = TfidfVectorizer(
            max_features=max_features, min_df=0.1, token_pattern=r'(?u)\b\w+\b')
        self.cutter = Cutter()
        self.logger = logging.getLogger('TfidfTransformer')

    def train(self, json_file):
        train_data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line, encoding='utf-8')
                question = data['question']
                q_tokens = self.cutter.cut_all(question)
                q_str = ' '.join(q_tokens)
                train_data.append(q_str)

        self.tfidf_transformer.fit_transform(train_data)
        self.logger.info('train tfidf transformer SUCCESS !')

    def predict(self, text):
        t_tokens = self.cutter.cut_all(text)
        t_str = ' '.join(t_tokens)
        prd = self.tfidf_transformer.transform([t_str])
        # self.logger.info('predict tfidf transformer SUCCESS !')
        return list(prd.toarray()[0])

    def save_model(self, model_file):
        with open(model_file, 'wb') as f:
            pickle.dump(self.tfidf_transformer, f)
        self.logger.info(
            'save tfidf transformer model ' +
            model_file +
            ' SUCCESS !')

    def load_model(self, model_file):
        with open(model_file, 'rb') as f:
            self.tfidf_transformer = pickle.load(f)
        self.logger.info(
            'load tfidf transformer model ' +
            model_file +
            ' SUCCESS !')

    def get_feature_dims(self):
        dim_num = len(self.tfidf_transformer.get_feature_names())
        self.logger.debug(
            'feature dims of tfidf transformer is: ' +
            str(dim_num))
        return dim_num

    def get_feature_names(self):
        names = self.tfidf_transformer.get_feature_names()
        self.logger.debug(
            'feature names of tfidf transformer is: ' +
            str(names))
        return names


def check_tfidf_transformer(tfidf_transformer: TfidfTransformer, faq_config: FaqConfig):
    dims = tfidf_transformer.get_feature_dims()
    if faq_config.tfidf_transformer.feature_dims != dims:
        faq_config.tfidf_transformer.feature_dims = dims
        faq_config.set('tfidf_transformer', 'feature_dims', str(dims))
        faq_config.save()


def init_tfidf_transformer(faq_config: FaqConfig):
    tfidf_transformer = TfidfTransformer(
        max_features=faq_config.tfidf_transformer.max_feature)
    tfidf_transformer.load_model(faq_config.tfidf_transformer.model_file)
    check_tfidf_transformer(tfidf_transformer, faq_config)
    logger = logging.getLogger('init_tfidf_transformer')
    logger.info('init tfidf transformer SUCCESS !')


def get_embedding_dims(tf_config: TfidfTransformerConfig):
    tfidf_transformer = TfidfTransformer(max_features=tf_config.max_feature)
    return tfidf_transformer.get_feature_dims()


def get_feature_dims(tf_config: TfidfTransformerConfig):
    tfidf_transformer = TfidfTransformer(max_features=tf_config.max_feature)
    return tfidf_transformer.get_feature_dims()


def generate_embedding(text, tf_config: TfidfTransformerConfig):
    tfidf_transformer = TfidfTransformer(max_features=tf_config.max_feature)
    try:
        emb_res = tfidf_transformer.predict(text)
    except NotFittedError:
        emb_res = []
    return emb_res


if __name__ == '__main__':
    from src.config import init_faq_config

    faq_config = init_faq_config('faq.config')
    tt = TfidfTransformer(
        max_features=faq_config.tfidf_transformer.max_feature)
    tt.train('../faq_vec.index')
    tt.save_model('tfidftransformer.pkl')
    print(tt.get_feature_names())
    print(tt.get_feature_dims())
    tt.load_model('tfidftransformer.pkl')

    print(tt.predict('你是谁？'))
    print(tt.predict('what are you ding today? 你好呀, do you know who am i ?'))
