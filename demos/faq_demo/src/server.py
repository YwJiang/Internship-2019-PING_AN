# encoding: utf-8

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.config import FaqConfig, init_faq_config
import json
import logging.config
from flask import Flask, request
from src.strategy import strategy, strategy_analysis_retrieval
from src.utils import Cutter
from src.light_gbm import init_light_gbm
# from src.tfidf_transformer import init_tfidf_transformer
from src.skip_embedding import init_skip_embedding
from src.annoy_search import init_annoy_search
from src.elastic_search import init_elastic_search
from src.abcnn.abcnn_model import init_abcnn


logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


server = Flask(__name__)


faq_config_file = 'faq.config'


@server.route('/search', methods=['POST', 'GET'])
def faq():
    logger = logging.getLogger('faq')
    if request.method == 'POST':
        data = request.get_data()
        data_json = json.loads(data, encoding='utf-8')
        query = data_json.get('query')
    else:
        query = request.args.get('query')

    try:
        logger.info('origin query: ' + query)
        faq_config = FaqConfig(faq_config_file)
        rsp = strategy(query, faq_config)
        return rsp
    except TypeError:
        logger.error('ERROR: NO query in request !', exc_info=True)
        return 'There is something wrong with request: NO query !'


@server.route('/retrieval', methods=['POST', 'GET'])
def retrieval_faq():
    logger = logging.getLogger('faq')
    if request.method == 'POST':
        data = request.get_data( )
        data_json = json.loads(data, encoding='utf-8')
        query = data_json.get('query')
    else:
        query = request.args.get('query')

    try:
        logger.info('origin query: ' + query)
        faq_config = FaqConfig(faq_config_file)
        rsp = strategy_analysis_retrieval(query, faq_config)
        return rsp
    except TypeError:
        logger.error('ERROR: NO query in request !', exc_info=True)
        return 'There is something wrong with request: NO query !'


def server_init(config_file):
    server_logger = logging.getLogger('server_init')

    faq_config = init_faq_config(config_file)
    server_logger.info('init faq config instance SUCCESS !')
    server_logger.info(faq_config.abcnn.model_file)

    Cutter()
    server_logger.info('init text cut tool instance SUCCESS !')

    init_elastic_search(faq_config.elastic_search)
    server_logger.info('init elastic search instance SUCCESS !')

    # init_tfidf_transformer(faq_config)
    # server_logger.info('init embedding model instance SUCCESS !')

    init_skip_embedding(faq_config.skip_embedding)
    server_logger.info('init embedding model instance SUCCESS !')

    faq_config.check_annoy_search()
    init_annoy_search(faq_config.annoy_search)
    server_logger.info('init annoy search instance SUCCESS !')
    faq_config.check_annoy_search()

    init_light_gbm(faq_config.lightgbm)
    server_logger.info('init light gbm instance SUCCESS !')

    init_abcnn(faq_config.abcnn)
    server_logger.info('init abcnn instance SUCCESS !')


def main(debug=False):
    server_logger = logging.getLogger('server')

    # faq_config_file = 'faq.config'

    server_init(faq_config_file)
    server_logger.info('init server SUCCESS !')
    faq_config = FaqConfig(faq_config_file)
    server.run(
        debug=debug,
        host=faq_config.server.ip,
        port=faq_config.server.port)


if __name__ == '__main__':

    main(debug=True)
