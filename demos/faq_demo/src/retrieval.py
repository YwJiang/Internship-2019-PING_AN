# encoding: utf-8

from src.config import FaqConfig, TermRetrievalConfig, SemanticRetrievalConfig, AnnoyConfig, EsConfig
from src.utils import QueryItem
from src.utils import FAQItem
from src.annoy_search import search_by_vector
from src.elastic_search import search_by_ids
from src.elastic_search import search_by_keywords
import logging.config
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


def term_retrieval(query_item: QueryItem, term_config: TermRetrievalConfig, es_config: EsConfig):
    logger = logging.getLogger('term_retrieval')
    search_result = search_by_keywords(
        query_item.query,
        es_config,
        top_n=term_config.top_n)
    length = len(search_result)
    rsp_res = []
    for i in range(length):
        if search_result[i]['_score'] > term_config.threshold:
            faq_temp = FAQItem(query_item=query_item)
            faq_temp.is_term = True
            faq_temp.term_score = search_result[i]['_score']
            faq_temp.id = search_result[i]['_id']
            faq_temp.question = search_result[i]['_source']['question']
            faq_temp.answer = search_result[i]['_source']['answer']
            faq_temp.question_tokens_zi = search_result[i]['_source']['question_tokens_zi']
            faq_temp.question_tokens_jieba = search_result[i]['_source']['question_tokens_jieba']
            faq_temp.question_vec = search_result[i]['_source']['question_vec']
            rsp_res.append(faq_temp)
    logger.info('term retrieval SUCCESS !')
    logger.debug('term retrieval num is ' + str(len(search_result)) +
                 ', after filter, remain num is ' + str(len(rsp_res)))
    return rsp_res


def semantic_retrieval(
        query_item: QueryItem,
        semantic_config: SemanticRetrievalConfig,
        annoy_config: AnnoyConfig,
        es_config: EsConfig):
    logger = logging.getLogger('semantic_retrieval')
    rsp_res = []

    ann_result = search_by_vector(
        query_item.query_vec,
        vec_dim=annoy_config.vec_dim,
        top_n=semantic_config.top_n)

    if isinstance(ann_result, tuple):
        if ann_result[0]:
            score = dict(
                map(lambda x, y: [x, y], ann_result[0], ann_result[1]))
            # print(score)
            search_result = search_by_ids(ann_result[0], es_config)
            length = len(search_result)

            for i in range(length):
                faq_temp = FAQItem(query_item=query_item)
                faq_temp.is_semantic = True
                faq_temp.id = search_result[i]['_id']
                faq_temp.semantic_score = score[int(faq_temp.id)]
                faq_temp.question = search_result[i]['_source']['question']
                faq_temp.answer = search_result[i]['_source']['answer']
                faq_temp.question_tokens_zi = search_result[i]['_source']['question_tokens_zi']
                faq_temp.question_tokens_jieba = search_result[i]['_source']['question_tokens_jieba']
                faq_temp.question_vec = search_result[i]['_source']['question_vec']
                rsp_res.append(faq_temp)
    else:
        if ann_result:
            search_result = search_by_ids(ann_result, es_config)
            length = len(search_result)

            for i in range(length):
                faq_temp = FAQItem(query_item=query_item)
                faq_temp.is_semantic = True
                faq_temp.id = search_result[i]['_id']
                faq_temp.question = search_result[i]['_source']['question']
                faq_temp.answer = search_result[i]['_source']['answer']
                faq_temp.question_tokens_zi = search_result[i]['_source']['question_tokens_zi']
                faq_temp.question_tokens_jieba = search_result[i]['_source']['question_tokens_jieba']
                faq_temp.question_vec = search_result[i]['_source']['question_vec']
                rsp_res.append(faq_temp)

    logger.info('semantic retrieval SUCCESS !')
    logger.debug('semantic retrieval num is ' + str(len(rsp_res)))
    return rsp_res


# 去除召回的问答对中重复的部分， 通过id
def remove_duplication(faq_item_list):
    logger = logging.getLogger('remove_duplication')

    id_set = set()
    id_map = {}
    res_after_remove = []
    for i in range(len(faq_item_list)):
        if faq_item_list[i].id in id_set:
            index = id_map[faq_item_list[i].id]
            res_after_remove[index].is_term = True
            res_after_remove[index].is_semantic = True
            # term score 初始化为0.0, 越大表示越相似
            res_after_remove[index].term_score = max(
                res_after_remove[index].term_score, faq_item_list[i].term_score)
            # semantic_score 初始化为0.0, 越小表示越相似
            res_after_remove[index].semantic_score = max(
                res_after_remove[index].semantic_score,
                faq_item_list[i].semantic_score)
        else:
            res_after_remove.append(faq_item_list[i])
            id_set.add(faq_item_list[i].id)
            id_map[faq_item_list[i].id] = len(res_after_remove) - 1
    logger.info('remove duplication SUCCESS !')
    logger.debug('before remove duplication, num is ' +
                 str(len(faq_item_list)) +
                 ', after, num is ' +
                 str(len(res_after_remove)))
    return res_after_remove


def  retrieval(query_item: QueryItem, faq_config: FaqConfig):
    t_rsp = term_retrieval(
        query_item,
        faq_config.term_retrieval,
        faq_config.elastic_search)
    s_rsp = semantic_retrieval(
        query_item,
        faq_config.semantic_retrieval,
        faq_config.annoy_search,
        faq_config.elastic_search)
    rsp = remove_duplication(t_rsp + s_rsp)

    return rsp


if __name__ == '__main__':
    from src.tfidf_transformer import init_tfidf_transformer
    from src.tfidf_transformer import TfidfTransformer
    from src.annoy_search import init_annoy_search
    from src.analysis import analysis
    from src.config import init_faq_config
    f_config = init_faq_config('faq.config')
    init_tfidf_transformer(f_config.tfidf_transformer)
    tt = TfidfTransformer()
    init_annoy_search(f_config.annoy_search)
    q = analysis('who are you ？', f_config)

    res = retrieval(q, f_config)

    from src.utils import faq_items_to_list
    print(faq_items_to_list(res))
