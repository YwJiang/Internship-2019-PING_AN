# encoding: utf-8

from string import punctuation
from zhon import hanzi
import re
import jieba


# 单例
def singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


# 去除标点符号(中文和英文)
def remove_punctuation(input_string):
    punc = punctuation + hanzi.punctuation
    output = re.sub(r'[{}]+'.format(punc), '', input_string)
    return output


def remove_punc(tokens_list):
    result = []
    for item in tokens_list:
        after_remove = remove_punctuation(item)
        if after_remove.strip():
            result.append(after_remove)
    return result


# 分词器
@singleton
class Cutter:
    def __init__(self):
        self.cutter = jieba
        self.cutter.initialize()

    def cut(self, input_string):
        return self.cutter.lcut(input_string)

    def cut_all(self, input_string):
        return self.cutter.cut(input_string, cut_all=True)

    def cut_and_remove_punc(self, input_string):
        cut_res = self.cutter.lcut(input_string)
        remove_res = remove_punc(cut_res)
        return remove_res

    def cut_zi_and_remove_punc(self, input_string):
        cut_res = ' '.join(input_string).split()
        remove_res = remove_punc(cut_res)
        return remove_res

# query 数据结构
class QueryItem:
    def __init__(self):
        self.query = ''
        self.query_vec = ''
        self.query_tokens_jieba = []
        self.query_tokens_zi = []


# faq结果 数据结构
class FAQItem:
    def __init__(self, query_item):
        self.id = 0
        self.question = ''
        self.question_vec = ''
        self.question_tokens_zi = []
        self.question_tokens_jieba = []
        self.answer = ''
        self.query_item = query_item
        self.is_term = False
        self.is_semantic = False
        self.term_score = 0.0
        self.semantic_score = 0.0
        self.score = 0.0
        self.bm25_similarity_score = 0.0
        self.edit_similarity_score = 0.0
        self.jaccard_similarity_score = 0.0
        self.abcnn_similarity = 0.0


def query_item_to_dict(query_item):
    res = {}
    for name, value in vars(query_item).items():
        if name == 'query_vec':
            continue
        res[name] = value
    return res


def faq_item_to_dict(faq_item, has_query=True):
    res = {}
    for name, value in vars(faq_item).items():
        if name == 'query_item' or name == 'question_vec':
            # if has_query:
            #     value = query_item_to_dict(value)
            # else:
            #     value = ''
            continue
        res[name] = value
    return res


def faq_items_to_list(faq_item_list):
    res = [{}] * len(faq_item_list)
    for i in range(len(faq_item_list)):
        res[i] = faq_item_to_dict(faq_item_list[i])
    return res


if __name__ == '__main__':

    in_s = 'are you ok ?？'
    out_s = remove_punctuation(in_s)
    print(out_s)
