# encoding: utf-8
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


from src.config import EsConfig
import time
import json
from src.utils import singleton
import logging.config
import logging
from elasticsearch import exceptions as es_exp
from elasticsearch import helpers
from elasticsearch import Elasticsearch

logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


@singleton
class ElasticSearch:

    def __init__(self, ip='127.0.0.1', port='9200'):
        self.logger = logging.getLogger('ElasticSearch')
        self.ip = ip
        self.port = port
        self.es = Elasticsearch(['%s:%s' % (self.ip, self.port)])

    # a sample of insert data from a json file
    def create_from_file(self, json_file):
        actions = []
        with open(json_file, 'r', encoding='utf8') as f:
            for line in f:
                json_line = json.loads(line)

                action = {
                    '_index': 'faq',
                    '_type': 'faq',
                    '_id': json_line['id'],
                    '_source': {
                        'question': json_line['question'],
                        'answer': json_line['answer'],
                        'question_tokens_zi': json_line['question_tokens_zi'],
                        'question_tokens_jieba': json_line['question_tokens_jieba'],
                        'question_vec': json_line['question_vec'],
                    }
                }
                actions.append(action)
                # insert per 500
                if len(actions) == 500:
                    helpers.bulk(self.es, actions)
                    del actions[0:len(actions)]
            if len(actions) > 0:
                helpers.bulk(self.es, actions)
        self.logger.info('create elastic search from json file SUCCESS !')

    def search_by_keywords(self, keywords, name='question', index='faq'):
        body = {
            "query": {
                "match": {
                    name: keywords
                }
            }
        }
        self.logger.debug(
            'search query is : {\'index\': \'' + index + '\', \'body\': ' + str(body) + '}')
        rsp = self.es.search(index=index, body=body)
        return rsp['hits']['hits']

    def search_by_ids(self, ids, index='faq'):
        body = {
            "query": {
                "ids": {
                    "values": ids
                }
            }
        }
        self.logger.debug(
            'search query is : {\'index\': \'' + index + '\', \'body\': ' + str(body) + '}')
        rsp = self.es.search(index=index, body=body)
        return rsp['hits']['hits']

    def delete_all(self):
        retry_count = 3
        while retry_count > 0:
            try:
                # delete all old index by query
                self.es.delete_by_query(
                    index='faq', body={
                        'query': {
                            'match_all': {}}})
                self.logger.info('delete old data SUCCESS !')
                break
            except es_exp.ConflictError:
                self.logger.error('delete ERROR ...', exc_info=True)
                time.sleep(3)
                retry_count -= 1
                continue
            except es_exp.NotFoundError:
                self.logger.error('No index in elastic search', exc_info=True)
                break


def init_elastic_search(es_config: EsConfig):
    logger = logging.getLogger('init_elastic_search')
    es = ElasticSearch(es_config.ip, es_config.port)
    es.delete_all()
    logger.info(es_config.es_file)
    es.create_from_file(es_config.es_file)

    time.sleep(3)

    logger.info('init elastic search SUCCESS !')


# 关键词召回，默认返回前20
def search_by_keywords(keywords, es_config: EsConfig, top_n=20):
    logger = logging.getLogger('search_by_keywords')
    try:
        es = ElasticSearch(es_config.ip, es_config.port)
        search_res = es.search_by_keywords(keywords)
    except es_exp.RequestError:
        search_res = []
        logger.error('search by keywords ERROR !', exc_info=True)

    if len(search_res) > top_n:
        search_res = search_res[:top_n]
    return search_res


def search_by_ids(ids, es_config: EsConfig):
    logger = logging.getLogger('search_by_ids')
    try:
        es = ElasticSearch(es_config.ip, es_config.port)
        search_res = es.search_by_ids(ids)
    except es_exp.RequestError:
        search_res = []
        logger.error('search by ids ERROR !', exc_info=True)

    return search_res


if __name__ == '__main__':

    from src.config import init_faq_config

    f_c = init_faq_config('faq.config')
    init_elastic_search(f_c.elastic_search)

    res = search_by_ids([1], f_c.elastic_search)

    # res = search_by_keywords('where are you')
    print('res : ', res)
