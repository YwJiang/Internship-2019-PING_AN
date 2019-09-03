# encoding: utf-8

from src.config import FaqConfig
import time
import json
from src.utils import query_item_to_dict
from src.utils import faq_items_to_list
from src.rank import rank
from src.match import match
from src.retrieval import retrieval
from src.analysis import analysis
import logging.config
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


def strategy(query, faq_config: FaqConfig):
    start = time.process_time()
    logger = logging.getLogger('strategy')
    query_item = analysis(query, faq_config)
    # end1 = time.process_time()
    # cost_time = (end1 - start) * 1000
    # logger.info('analysis cost time: ' + str(cost_time) + ' ms.')
    faq_items = retrieval(query_item, faq_config)
    # end2 = time.process_time()
    # cost_time = (end2 - end1) * 1000
    # logger.info('retrieval cost time: ' + str(cost_time) + ' ms.')
    faq_items = match(query_item, faq_items)
    # end3 = time.process_time()
    # cost_time = (end3 - end2) * 1000
    # logger.info('match cost time: ' + str(cost_time) + ' ms.')
    faq_items = rank(faq_items, faq_config)
    # end4 = time.process_time()
    # cost_time = (end4 - end3) * 1000
    # logger.info('rank cost time: ' + str(cost_time) + ' ms.')
    rsp = {}
    rsp['request'] = query_item_to_dict(query_item)
    rsp['response'] = faq_items_to_list(faq_items)
    print(rsp)
    rsp_json = json.dumps(rsp, ensure_ascii=False)
    logger.info('do strategy SUCCESS !')
    logger.info(rsp_json)
    end = time.process_time()
    cost_time = (end - start) * 1000
    logger.debug('total cost time: ' + str(cost_time) + ' ms.')
    return rsp_json


def strategy_analysis_retrieval(query, faq_config: FaqConfig):
    start = time.process_time()
    logger = logging.getLogger('strategy')
    query_item = analysis(query, faq_config)
    # end1 = time.process_time()
    # cost_time = (end1 - start) * 1000
    # logger.info('analysis cost time: ' + str(cost_time) + ' ms.')
    faq_items = retrieval(query_item, faq_config)
    # end2 = time.process_time()
    # cost_time = (end2 - end1) * 1000
    # logger.info('retrieval cost time: ' + str(cost_time) + ' ms.')
    # faq_items = match(query_item, faq_items)
    # end3 = time.process_time()
    # cost_time = (end3 - end2) * 1000
    # logger.info('match cost time: ' + str(cost_time) + ' ms.')
    # faq_items = rank(faq_items, faq_config)
    # end4 = time.process_time()
    # cost_time = (end4 - end3) * 1000
    # logger.info('rank cost time: ' + str(cost_time) + ' ms.')
    rsp = {}
    rsp['request'] = query_item_to_dict(query_item)
    rsp['response'] = faq_items_to_list(faq_items)
    rsp_json = json.dumps(rsp, ensure_ascii=False)
    logger.info('do strategy SUCCESS !')
    logger.info(rsp_json)
    end = time.process_time()
    cost_time = (end - start) * 1000
    logger.debug('total cost time: ' + str(cost_time) + ' ms.')
    return rsp_json


if __name__ == '__main__':
    from src.tfidf_transformer import init_tfidf_transformer
    from src.annoy_search import init_annoy_search
    from src.config import init_faq_config

    faq_config = init_faq_config('faq.config')
    init_tfidf_transformer(faq_config.tfidf_transformer)

    init_annoy_search(faq_config.annoy_search)
    faq_config.check_annoy_search()

    q = 'who are you?'
    res = strategy(q, faq_config)
