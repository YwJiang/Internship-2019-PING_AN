"""
https://github.com/ryankiros/skip-thoughts

constructing and loading dictionaries
"""

import pickle as pkl
from collections import OrderedDict
import argparse
import os

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
    for idx, word in enumerate(sorted_words):
        worddict[word] = idx+2

    print("vocab.py: build_dictionary: wordict", worddict, type(worddict))

    return worddict, wordcount

def load_dictionary(loc='./data/faq.txt.pkl'):
    """
    load a dictionary
    :param loc:
    :return:
    """
    print(os.getcwd())
    print("vocab.py:", loc)
    with open(loc, 'rb') as f:
        worddict = pkl.load(f)
        wordcount = pkl.load(f)

    print("vocab.py: worddict", worddict, type(worddict))
    print("vocab.py: wordcount", wordcount)
    return worddict


def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location
    """
    with open(loc, 'wb') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)

def build_and_save_dictionary(text, source):
    save_loc = source+".pkl"
    try:
        cached = load_dictionary(save_loc)
        print("Using cached dictionary at {}".format(save_loc))
        return cached
    except:
        pass
    # build again and save
    print('unable to load from cached, building fresh')
    worddict, wordcount = build_dictionary(text)
    print('Got {} unique words'.format(len(worddict)))
    print('Saving dictionary at {}'.format(save_loc))
    save_dictionary(worddict, wordcount, save_loc)
    return worddict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", type=str)
    args = parser.parse_args()
    print('')


