# encoding: utf-8

import jieba
from src.utils import remove_punc
import math
import numpy as np
from src.utils import QueryItem, FAQItem
import logging.config
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.abcnn.abcnn_model import AbcnnModel

logging.config.fileConfig(fname='log.config')


def jaccard_similarity(list1, list2):
    logger = logging.getLogger('jaccard_similarity')
    intersection_res = list(set(list1).intersection(set(list2)))
    union_res = list(set(list1).union(set(list2)))
    sim = len(intersection_res) * 1.0 / (len(union_res) + 1e-9)

    logger.info('jaccard_similarity calculate SUCCESS !')
    return sim


def edit_similarity(v1, v2):
    # vec1, vec2: vector
    # 0-1 the bigger, the closer
    logger = logging.getLogger('edit_similarity')
    if len(v1) == 0:
        return 1 - len(v2) / (max(len(v1), len(v2)) + 1e-9)
    if len(v2) == 0:
        return 1 - len(v1) / (max(len(v1), len(v2)) + 1e-9)
    matrix = np.zeros((len(v1) + 1, len(v2) + 1))
    matrix[0, :] = range(0, len(v2) + 1)  # first row
    matrix[:, 0] = range(0, len(v1) + 1)  # first column

    for i in range(1, len(v1) + 1):
        for j in range(1, len(v2) + 1):
            temp = 0 if v1[i - 1] == v2[j - 1] else 1
            matrix[i, j] = min(matrix[i - 1, j] + 1,
                               matrix[i, j - 1] + 1, matrix[i - 1, j - 1] + temp)
    result = 1 - matrix[len(v1), len(v2)] / (max(len(v1), len(v2)) + 1e-9)

    logger.debug("edit_similarity()" + str(result))
    logger.info('edit_similarity calculate SUCCESS !')
    return result



def bm25_similarity(doc1_list, candits, k=1.5, b=0.75, avgl=12):
    # doc1: string sentence
    # candits: list of strings
    logger = logging.getLogger('bm25_similarity')
    doc1 = doc1_list

    # 统计candits中的单词：频数; 存单个句子的词频
    dic_candits = {}  # 每个词在几个句子里出现
    len_candits = len(candits)
    count_dic = []  # 存放每个句子长度
    list_dic_candits = []  # 单个句子的词频字典为元素
    for question in candits:  # question是list，元素是tokens
        count_dic.append(len(question))  # 每个单词的数量
        sentence_dic = {}
        for word in question:
            sentence_dic[word] = sentence_dic.get(word, 0) + 1
        for word in set(question):  # 为idf计算用，每个单词在几个句子里出现
            dic_candits[word] = dic_candits.get(word, 0) + 1
        list_dic_candits.append(sentence_dic)

    # 计算dic_candits中每个词的idf
    idf = {}
    for word, freq in dic_candits.items():
        idf[word] = math.log(len_candits - freq + 0.5) - math.log(freq + 0.5)

    # define the score
    score_result = []
    for i in range(len(list_dic_candits)):
        score = 0
        for word in doc1:
            idf_word = idf.get(word, 0)
            # print("idf", word, ":", idf_word)
            score += idf_word * (list_dic_candits[i].get(word, 0) / count_dic[i]) * (k + 1) / (
                (list_dic_candits[i].get(word, 0) / count_dic[i]) + k * (1 - b + b * len(doc1) / avgl) + 1)
        score = 1.0 / (1 + math.exp(-score))
        score_result.append(score)

    logger.info('bm25_similarity calculate SUCCESS !')
    return score_result


def cal_jaccard_similarity(query_item, retrieval_result):
    logger = logging.getLogger('cal_jaccard_similarity')
    v1 = query_item.query_tokens_zi
    for item in retrieval_result:
        question = item.question_tokens_zi
        logger.debug('cal_jaccard_similarity: query tokens of query item'+str(v1))
        logger.debug('cal_jaccard_similarity: item question_tokens in retrieval_result'+str(question))
        item.jaccard_similarity_score = jaccard_similarity(v1, question)
    logger.info('cal_jaccard_similarity finished SUCCESS !')


def cal_edit_similarity(query_item, retrieval_result):
    logger = logging.getLogger('cal_edit_similarity')
    v1 = query_item.query_tokens_zi
    for item in retrieval_result:
        question = item.question_tokens_zi
        logger.debug('cal_edit_similarity: query tokens of query item' + str(v1))
        logger.debug('cal_edit_similarity: item question_tokens in retrieval_result' + str(question))
        item.edit_similarity_score = edit_similarity(v1, question)
    logger.info('cal_edit_similarity finished SUCCESS !')


def cal_bm25_similarity(query_item, retrieval_result):
    logger = logging.getLogger('cal_bm25_similarity')
    doc1 = query_item.query_tokens_zi
    candits = [item.question_tokens_zi for item in retrieval_result]
    scores = bm25_similarity(doc1, candits)
    logger.debug('cal_bm25_similarity: query tokens of query item' + str(doc1))
    logger.debug('cal_bm25_similarity: item question_tokens in retrieval_result' + str(candits))

    for i in range(len(retrieval_result)):
        retrieval_result[i].bm25_similarity_score = scores[i]
    logger.info('cal_bm25_similarity finished SUCCESS !')


def cal_abcnn_similarity(query_item: QueryItem, retrieval_result):
    logger = logging.getLogger('abcnn')
    abcnn = AbcnnModel()
    sen1_list = [query_item.query for i in range(len(retrieval_result))]
    sen2_list = [item.question for item in retrieval_result]
    p_test, h_test = abcnn.transfer_char_data(sen1_list, sen2_list)
    prd = abcnn.predict(p_test, h_test).tolist()
    for i in range(len(retrieval_result)):
        retrieval_result[i].abcnn_similarity = prd[i]
        # print(item.abcnn_similarity)
        logger.debug("abcnn similarity"+str(prd[i]))
    logger.info('cal_abcnn_similarity finished SUCCESS !')

def match(query_item: QueryItem, retrieval_result):
    logger = logging.getLogger('match')
    cal_bm25_similarity(query_item, retrieval_result)
    cal_edit_similarity(query_item, retrieval_result)
    cal_jaccard_similarity(query_item, retrieval_result)
    cal_abcnn_similarity(query_item, retrieval_result)
    logger.info('match calculation finished SUCCESS !')

    return retrieval_result


if __name__ == '__main__':

    # s1 = '你 说 你 是 谁'
    # s2 = '我 不 知道 你 是 谁'
    # v1 = [12, 3, 4, 6]
    # v2 = [2, 4, 5, 6]
    # print(jaccard_similarity(s1.split(), s2.split()))
    # print(dice_similarity(s1.split(), s2.split()))
    # print(cos_similarity(v1, v2))

    q = QueryItem()
    q.query = "He did but the initiative did not get very far tomorrow however yesterday tomorrow however yesterday tomorrow however yesterday tomorrow however yesterday"
    q.query_tokens = remove_punc(jieba.cut(q.query))

    cand1 = FAQItem(q)
    cand1.question = "He did but the initiative did not get very far today however yesterday tomorrow however yesterday tomorrow however yesterday tomorrow however yesterday"
    cand1.question_tokens = remove_punc(jieba.cut(cand1.question))

    cand2 = FAQItem(q)
    cand2.question = "What happened the initiative does not go very far."
    cand2.question_tokens = remove_punc(jieba.cut(cand2.question))

    cand3 = FAQItem(q)
    cand3.question = "Those who stand apart from reinforced cooperation"
    cand3.question_tokens = remove_punc(jieba.cut(cand3.question))

    retrieval_result = [cand1, cand2, cand3]
    match(q, retrieval_result)

    for item in retrieval_result:
        print(item.question)
        print('bm25', item.bm25_similarity_score)
        print('edit', item.edit_similarity_score)
        print('jaccard', item.jaccard_similarity_score)
