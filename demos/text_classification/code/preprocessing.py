import numpy as np
import pandas as pd
import jieba
from string import punctuation
from zhon import hanzi
import re
import pickle as pkl
from collections import OrderedDict
import argparse
import os


# remove the punctuations (both English and Chinese)
def remove_punctuation(input_string):
    punc = punctuation + hanzi.punctuation
    output = re.sub(r'[{}]+'.format(punc), '', input_string)
    return output



def jieba_cut(sentence):
    sent_seg = jieba.cut(sentence)
    sent_new = ' '.join(sent_seg)
    sent_new = remove_punctuation(sent_new)
    sent_strip = sent_new.strip()
    return sent_strip



def build_dictionary(text):
    """
    Build a dictionary
    :param text: list of sentences (pre-tokens)
    :return:
    """
    wordcount = {}
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1

    sorted_words = sorted(list(wordcount.keys()), key=lambda x: wordcount[x], reverse=True)

    worddict = OrderedDict()
    worddict["<PAD>"] = 0
    worddict["<UNK>"] = 1
    for idx, word in enumerate(sorted_words):
        worddict[word] = idx+2

#     print("vocab.py: build_dictionary: wordict", worddict, type(worddict))

    return worddict, wordcount


def encode_sent(list_sent, dictionary):
    index = [np.nan]*len(list_sent)
    for i in range(len(list_sent)):
        sent = list_sent[i]
        list_words = jieba_cut(sent).split()
        list_index = [0]*len(list_words)
        for j in range(len(list_words)):
            list_index[j] = dictionary.get(list_words[j], 1)
        index[i] = np.array(list_index)
    return index


def text_df_to_array_pad(df, text_colname="fact_reason", pad_len=None):
    dataset = df
    clean_sent = ['']* len(dataset)
    for i in range(len(dataset[text_colname])):
        sent = dataset[text_colname].tolist()[i]
        try:
            clean_sent[i] = jieba_cut(sent)
        except:
            print(sent)
            print("WRONG: dataset Line ", i)

    # sentence_to_dict_index
    word_dictionary, _ = build_dictionary(clean_sent)
    all_sentence_index = encode_sent(dataset[text_colname].tolist(), word_dictionary)

    # calculate the max length of index
    if pad_len:
        max_len = pad_len
    else:
        len_list = [0]*len(all_sentence_index)
        for sentence in all_sentence_index:
            len_list.append(len(sentence))
        max_len = max(len_list)

    all_data = keras.preprocessing.sequence.pad_sequences(all_sentence_index,
                                                          value=word_dictionary['<PAD>'],
                                                          padding='post',
                                                          maxlen=max_len)
    return all_data, len(word_dictionary), max_len


if __name__ == "__main__":
    print(__name__)
    path = "../data/"
