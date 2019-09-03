# -*- coding: utf-8 -*-

from src.sentence_embedding.model import UniSkip, Encoder
from src.sentence_embedding.data_loader import DataLoader
from src.sentence_embedding.vocab import load_dictionary
from src.sentence_embedding.config import *
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch
import jieba
from src.utils import singleton


@singleton
class UsableEncoder:

    def __init__(self, model_path='./sentence_embedding/saved_models/skip_best', dict_path='./sentence_embedding/data/faq.txt.pkl'):

        print("Preparing the DataLoader. Loading the word dictionary")
        print("sentence_emb.py: load dict", load_dictionary(dict_path), "\n",
              load_dictionary(dict_path) )
        self.d = DataLoader(sentences=[''], word_dict=load_dictionary(dict_path))
        self.encoder = None

        print("Loading encoder from the saved model at {}". format(model_path))
        model = UniSkip()

        # print('sentence_emb: ', os.getcwd())
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.encoder = model.encoder
        if USE_CUDA:
            self.encoder.cuda(CUDA_DEVICE)


    def encode(self, sentence):
        sen_idx = [self.d.convert_sentence_to_indices(sentence)]
        sen_idx = torch.stack(sen_idx)

        sen_emb, _ = self.encoder(sen_idx)
        sen_emb = sen_emb.view(-1, self.encoder.thought_size)
        sen_emb = sen_emb.data.cpu().numpy()
        ret = np.array(sen_emb)

        return ret

if __name__ == "__main__":
    import os
    print(os.getcwd())
    model_path = './saved_models/skip_best'
    dict_path = './data/faq.txt.pkl'
    usable_encoder = UsableEncoder(model_path, dict_path)

    sentence = u'实现社会主义制度'
    sent_seg = jieba.cut(sentence)
    sent_new = ' '.join(sent_seg)
    sent_strip = sent_new.strip()
    sentence_emb = usable_encoder.encode(sent_strip)
    print(sentence_emb)
