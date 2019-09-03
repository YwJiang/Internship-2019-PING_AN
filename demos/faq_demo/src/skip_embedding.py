# encoding: utf-8

from src.config import FaqConfig, SkipConfig
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
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.sentence_embedding.sentence_emb import UsableEncoder
import jieba

def init_skip_embedding(skip_config: SkipConfig):
    usable_encoder = UsableEncoder(skip_config.model_file,
                                   skip_config.dict_file)

    logger = logging.getLogger('init_skip_embedding')
    logger.info('init skip embedding SUCCESS! ')
    return usable_encoder

# def generate_embedding(sent_strip, skip_config: SkipConfig):
#     usable_encoder = UsableEncoder(skip_config.model_file,
#                                    skip_config.dict_file)
#
#     logger = logging.getLogger('init_skip_embedding')
#     logger.info('init skip embedding SUCCESS!')
#
#     return usable_encoder

def generate_embedding(sent_strip, skip_config: SkipConfig):
    usable_encoder = UsableEncoder(skip_config.model_file,
                                   skip_config.dict_file)
    sentence_emb = usable_encoder.encode(sent_strip)
    return sentence_emb[0].tolist()

if __name__ == "__main__":

    dict_path = './sentence_embedding/data/faq.txt.pkl'
    model_path = './sentence_embedding/saved_models/skip_best'
    import os
    print(os.getcwd())
    usable_encoder = UsableEncoder( model_path, dict_path)

    sentence = u"you are the apple of my eye"
    sent_seg = jieba.cut(sentence)
    sent_new = ' '.join(sent_seg)
    sent_strip = sent_new.strip()
    print(sent_strip, type(sent_strip))
    sentence_emb = usable_encoder.encode(sent_strip)
    print(sentence_emb)
