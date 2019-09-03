import torch
from torch.autograd import Variable
from src.sentence_embedding.vocab import *
from src.sentence_embedding.config import *
import numpy as np
import random
import jieba
import tqdm

np.random.seed(0)

class DataLoader:
    EOS = 0
    UNK = 1

    maxlen = MAXLEN

    def __init__(self, text_file=None, sentences=None, word_dict=None):

        """
        若同时传入了text_file和sentences，读text_file然后加到sentences列表后，再初始化
        若只传入sentences列表，就直接初始化
        :param text_file:
        :param sentences:
        :param word_dict:
        :return:
        """


        if text_file:
            print('Loading text file at {}'.format(text_file))
            count = 0
            sentences = []
            for line in open(text_file, "rt"):
                line_seg = jieba.cut(line)
                line_new = ' '.join(line_seg)
                sentences.append(line_new)
                count += 1
                if count % 5000 == 0:
                    print("process %d sentences", count)
                if count % 100000 == 0:
                    break

            print("Making dictionary for these words")
            word_dict = build_and_save_dictionary(sentences, source=text_file)
            print("word_dcit length:", len(word_dict))

        assert sentences and word_dict, "Please provide the file to extract from or give sentences and word_dict "

        self.sentences = sentences
        self.word_dict = word_dict
        print("data_loader.py:", word_dict)
        print("word_dict_len:", len(self.word_dict))
        print("Making reverse dictionary")
        self.revmap = list(self.word_dict.items())
        self.lengths = [len(sent) for sent in self.sentences]

    def convert_sentence_to_indices(self, sentence):


        indices = []
        for w in sentence.split():
            if self.word_dict.get(w, VOCAB_SIZE + 1) < VOCAB_SIZE :
                item = self.word_dict.get(w)
            else:
                item = self.UNK

            # item = self.word_dict.get(w, self.UNK)
            indices.append(item)
        indices = indices[: self.maxlen - 1]

        indices += [self.EOS] * (self.maxlen - len(indices)) #补齐到相同长度

        indices = np.array(indices)
        indices = Variable(torch.from_numpy(indices))
        if USE_CUDA:
            indices = indices.cuda(CUDA_DEVICE)
        return indices

    def convert_indices_to_sentences(self, indices):
        print(indices, type(indices))
        def convert_index_to_word(idx):
            print("data_loader.py: convert_index_to_word", idx)
            idx = idx.data
            if idx == 0:
                return "EOS"
            elif idx == 1:
                return "UNK"

            search_idx = idx -2
            if search_idx >= len(self.revmap):
                return "NA"

            word, idx_ = self.revmap[search_idx]

            assert idx_ == idx
            return word

        words = [convert_index_to_word(idx) for idx in indices]
        return " ".join(words)

    def fetch_batch(self, batch_size):
        first_index = random.randint(0, len(self.sentences) - batch_size)
        batch = []
        lengths = []

        for i in range(first_index, first_index + batch_size):
            sent = self.sentences[i]
            ind = self.convert_sentence_to_indices(sent)
            batch.append(ind)
            lengths.append(min(len(sent.split()), MAXLEN))

        batch = torch.stack(batch)
        lengths = np.array(lengths)

        return batch, lengths


