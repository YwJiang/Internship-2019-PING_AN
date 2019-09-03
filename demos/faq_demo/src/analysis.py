# encoding: utf-8
from src.config import FaqConfig
# from src.tfidf_transformer import generate_embedding
from src.skip_embedding import generate_embedding
from src.utils import query_item_to_dict
from src.utils import Cutter
from src.utils import QueryItem
import logging.config
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


def analysis(query, faq_config: FaqConfig):
    logger = logging.getLogger('analysis')
    query_item = QueryItem()
    query_item.query = query
    cutter = Cutter()
    # tokens = cutter.cut(query)
    tokens = cutter.cut_zi_and_remove_punc(query)
    query_item.query_tokens_zi = tokens

    tokens_jieba = cutter.cut_and_remove_punc(query)
    query_item.query_tokens_jieba = tokens_jieba

    text = ' '.join(query_item.query_tokens_jieba)
    text_new = text.strip()
    query_item.query_vec = generate_embedding(
        text_new, faq_config.skip_embedding)

    # query_item.query_vec = generate_embedding(
    #     query_item.query, faq_config.tfidf_transformer)
    logger.info('analysis SUCCESS !')
    logger.debug('query info : ' + str(query_item_to_dict(query_item)))
    return query_item


if __name__ == '__main__':
    pass
