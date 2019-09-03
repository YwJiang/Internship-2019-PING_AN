# encoding: utf-8

from src.light_gbm import light_gbm_predict
import pandas as pd
from src.config import FaqConfig
import logging.config
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


def calc_score(faq_item_list):
    jc_list = []
    ed_list = []
    bm_list = []
    for item in faq_item_list:
        jc_list.append(item.jaccard_similarity_score)
        ed_list.append(item.edit_similarity_score)
        bm_list.append(item.bm25_similarity_score)
    ma = [jc_list, ed_list, bm_list]
    predict_score = light_gbm_predict(pd.DataFrame(ma).T)
    for i in range(len(faq_item_list)):
        faq_item_list[i].score = predict_score[i]
    return faq_item_list


def calc_score_with_abcnn(faq_item_list):
    jc_list = []
    ed_list = []
    bm_list = []
    abcnn_list = []
    for item in faq_item_list:
        jc_list.append(item.jaccard_similarity_score)
        ed_list.append(item.edit_similarity_score)
        bm_list.append(item.bm25_similarity_score)
        abcnn_list.append(item.abcnn_similarity)
    ma = [jc_list, ed_list, bm_list, abcnn_list]
    predict_score = light_gbm_predict(pd.DataFrame(ma).T)
    for i in range(len(faq_item_list)):
        faq_item_list[i].score = predict_score[i]
    return faq_item_list


# 默认按照score从大到小排序
def sort_by_score(faq_item_list, top_n=5):
    logger = logging.getLogger('rank')
    length = len(faq_item_list)
    score_list = [0.0] * length
    for i in range(length):
        score_list[i] = faq_item_list[i].score
    sorted_score = sorted(
        enumerate(score_list),
        key=lambda x: x[1],
        reverse=True)
    idx = [x[0] for x in sorted_score]
    output_list = [{}] * length
    for i in range(length):
        output_list[i] = faq_item_list[idx[i]]
    logger.info('rank SUCCESS !')
    if length > top_n:
        output_list = output_list[:top_n]
    return output_list


def rank(input_list, faq_config: FaqConfig):
    output_list = calc_score(input_list)
    # output_list = calc_score_with_abcnn(input_list)
    output_list = sort_by_score(output_list, top_n=faq_config.rank.top_n)
    return output_list


if __name__ == '__main__':

    from src.utils import FAQItem, QueryItem, faq_items_to_list
    from src.config import init_faq_config

    f_c = init_faq_config('faq.config')
    q_i = QueryItem()
    q = []
    for ii in range(10):
        f_i = FAQItem(q_i)
        f_i.score = ii / 10.0
        q.append(f_i)

    r = rank(q, f_c)
    print(faq_items_to_list(r))
