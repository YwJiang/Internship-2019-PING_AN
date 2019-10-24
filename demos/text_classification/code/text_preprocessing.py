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
from tensorflow import keras

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


def select_more_than_ten(X, y):
    # X, y should be dataframe
    # filter to select y with more than ten labels'
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    y_colname = y.columns
    X_colname = X.columns
    con_df = pd.concat([X, y], axis=1)
    train_df = con_df.groupby([i for i in y_colname])
    train_df_new = train_df.filter(lambda x: x[X_colname].count()>10)
    return train_df_new[X_colname], train_df_new[y_colname]


def X_y_filter_to_array_pad(X, y, pad_len=None):
    X, y_label = select_more_than_ten(X, y)
    clean_sent = ['']*len(y)
    x_list = X.iloc[:, 0].tolist()
    for i in range(len(x_list)):
        sent = x_list[i]
        try:
            clean_sent[i] = jieba_cut(sent)
        except:
            print(sent)
            print("dataset line: ", i)

    # sentence_to_dict_index
    word_dictionary, _ = build_dictionary(clean_sent)
    all_sentence_index = encode_sent(x_list, word_dictionary)

    #
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
    return all_data, y_label, len(word_dictionary), max_len






if __name__ == "__main__":
    print(__name__)
    path = "../data/credit_card_labels_factreason.xlsx"
    pd_all = pd.read_excel(path)
    # df, dict_len, padding_len = text_df_to_array_pad(pd_all, 'fact_reason', pad_len=100)
    #     # df = pd.DataFrame(df)
    #     # print("df shape: ", df.shape)
    #     # print("dictionary: ", dict_len)
    #     # print("padding: ", padding_len)

    X = pd_all['fact_reason']
    y = pd_all.iloc[:, 0:3]
    df, y, dict_len, padding_len = X_y_filter_to_array_pad(X, y, pad_len=100)
    df = pd.DataFrame(df)
    print("df shape: ", df.shape)
    print("y shape: ", y.shape)
    print("dictionary len: ", dict_len)
    print("padding: ", padding_len)