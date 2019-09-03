# -*- coding:utf-8 -*-

from src.sentence_embedding.sentence_emb import UsableEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jieba

model_path = './saved_models/skip-best'
dict_path = './data/wiki_clean_cn.txt.pkl'

usable_encoder = UsableEncoder()

stand_q_list = []
simi_q_list = []
standard_q = './data/standard_q.txt'
similar_q = './data/similar_q.txt'
f_in_stand = open(standard_q, 'r')
f_in_simi = open(similar_q,'r')

for line in f_in_stand:
    stand_q_list.append(line)

for line in f_in_simi:
    simi_q_list.append(line)

print(len(stand_q_list), len(simi_q_list))

stand_q_emb = []
simi_q_emb = []

for sentence in stand_q_list:
