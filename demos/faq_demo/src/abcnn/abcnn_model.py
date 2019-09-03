from src.abcnn.graph import Graph
from src.abcnn import args
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from src.utils import singleton
import logging
import logging.config
from src.config import AbcnnConfig

@singleton
class AbcnnModel:
    def __init__(self):
        self.model = Graph(True, True)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.word2idx = {}

    def load_model(self, model_file='../model/abcnn2.ckpt'):
        self.saver.restore(self.sess, model_file)
        print('load SUCCESS !')

    def train(self, p, h, y, p_eval, h_eval, y_eval,
              model_file='../model/abcnn.ckpt'):
        p, h, y = self.shuffle(p, h, y)

        p_holder = tf.placeholder(
            dtype=tf.int32, shape=(
                None, args.seq_length), name='p')
        h_holder = tf.placeholder(
            dtype=tf.int32, shape=(
                None, args.seq_length), name='h')
        y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

        dataset = tf.data.Dataset.from_tensor_slices(
            (p_holder, h_holder, y_holder))
        dataset = dataset.batch(args.batch_size).repeat(args.epochs)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1

        with tf.Session(config=config)as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(
                iterator.initializer,
                feed_dict={
                    p_holder: p,
                    h_holder: h,
                    y_holder: y})
            steps = int(len(y) / args.batch_size)
            last_loss = 1.0
            for epoch in range(args.epochs):
                for step in range(steps):
                    p_batch, h_batch, y_batch = sess.run(next_element)
                    _, loss, acc = sess.run([self.model.train_op, self.model.loss, self.model.acc],
                                            feed_dict={self.model.p: p_batch,
                                                       self.model.h: h_batch,
                                                       self.model.y: y_batch,
                                                       self.model.keep_prob: args.keep_prob})
                    print(
                        'epoch:',
                        epoch,
                        ' step:',
                        step,
                        ' loss: ',
                        loss,
                        ' acc:',
                        acc)

                loss_eval, acc_eval = sess.run([self.model.loss, self.model.acc],
                                               feed_dict={self.model.p: p_eval,
                                                          self.model.h: h_eval,
                                                          self.model.y: y_eval,
                                                          self.model.keep_prob: 1})
                print('loss_eval: ', loss_eval, ' acc_eval:', acc_eval)
                print('\n')

                if loss_eval < last_loss:
                    last_loss = loss_eval
                    self.saver.save(sess, model_file)

    def predict(self, p, h):
        # with self.sess:
        #     prediction = self.sess.run(self.model.prediction,
        #                                feed_dict={self.model.p: p,
        #                                           self.model.h: h,
        #                                           self.model.keep_prob: 1})
        prediction = self.sess.run(self.model.prediction,
                                   feed_dict={self.model.p: p,
                                              self.model.h: h,
                                              self.model.keep_prob: 1})
        return prediction

    def test(self, p, h, y):
        with self.sess:
            loss, acc = self.sess.run([self.model.loss, self.model.acc],
                                      feed_dict={self.model.p: p,
                                                 self.model.h: h,
                                                 self.model.y: y,
                                                 self.model.keep_prob: 1})

        return loss, acc


    # 加载字典
    def load_char_vocab(self):
        path = os.path.join(os.path.dirname(__file__), './input/vocab.txt')
        vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
        self.word2idx = {word: index for index, word in enumerate(vocab)}
        return self.word2idx


    def pad_sequences(self, sequences, maxlen=None, dtype='int32', padding='post',
                      truncating='post', value=0.):
        ''' pad_sequences

        把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
        最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
        开始处理，默认都是从尾部开始。

        Arguments:
            sequences: 序列
            maxlen: int 最大长度
            dtype: 转变后的数据类型
            padding: 填充方法'pre' or 'post'
            truncating: 截取方法'pre' or 'post'
            value: float 填充的值

        Returns:
            x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)

        '''
        lengths = [len(s) for s in sequences]

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue  # empty list was found
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % padding)

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
        return x

    def char_index(self, p_sentences, h_sentences):

        p_list, h_list = [], []
        for p_sentence, h_sentence in zip(p_sentences, h_sentences):
            p = [self.word2idx[word.lower()] for word in p_sentence if len(
                word.strip()) > 0 and word.lower() in self.word2idx.keys()]
            h = [self.word2idx[word.lower()] for word in h_sentence if len(
                word.strip()) > 0 and word.lower() in self.word2idx.keys()]

            p_list.append(p)
            h_list.append(h)

        p_list = self.pad_sequences(p_list, maxlen=args.seq_length)
        h_list = self.pad_sequences(h_list, maxlen=args.seq_length)


        return p_list, h_list

    def shuffle(self, *arrs):
        arrs = list(arrs)
        for i, arr in enumerate(arrs):
            assert len(arrs[0]) == len(arrs[i])
            arrs[i] = np.array(arr)
        p = np.random.permutation(len(arrs[0]))
        return tuple(arr[p] for arr in arrs)

    # 加载char_index训练数据
    def load_char_data(self, file, data_size=None):
        # path = os.path.join(os.path.dirname(__file__), '../' + file)
        # df = pd.read_csv(path)
        df = pd.read_csv(file)
        p = df['sentence1'].values[0:data_size]
        h = df['sentence2'].values[0:data_size]
        label = df['label'].values[0:data_size]

        p_c_index, h_c_index = self.char_index(p, h)

        return p_c_index, h_c_index, label

    # 针对传入两个列表（为match:相当于test数据）
    def transfer_char_data(self, p, h):
        p_c_index, h_c_index = self.char_index(p, h)
        return p_c_index, h_c_index


def init_abcnn(abcnn_config: AbcnnConfig):
    logger = logging.getLogger('init_abcnn')
    abcnn_model = AbcnnModel()
    abcnn_model.load_char_vocab()
    abcnn_model.load_model(abcnn_config.model_file)
    logger.info('init abcnn model SUCCESS !')


if __name__ == '__main__':

    abcnn = AbcnnModel()

    # predict
    abcnn.load_char_vocab()
    p_test, h_test, y_test = abcnn.load_char_data('./input/test.csv', data_size=None)
    abcnn.load_model('../model/abcnn2.ckpt')
    prd = abcnn.predict(p_test, h_test)





    # train
    # abcnn.load_char_vocab()
    # p, h, y = abcnn.load_char_data('input/train.csv', data_size=None)
    # p_eval, h_eval, y_eval = abcnn.load_char_data('input/dev.csv', data_size=1000)
    # abcnn.train(p, h, y, p_eval, h_eval, y_eval, '../model/abcnn2.ckpt')
