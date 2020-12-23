from keras.utils import Sequence
import numpy as np
import pandas as pd
import math
import os
import soundfile as sf
from scipy import sparse

from random import shuffle
from util.const import Const
from util.wav_util import compute_fbank_from_api, compute_fbank_from_file
from util.utils import sparse_tuple_from
from util.utils import build_LFR_features

home_dir = os.getcwd()
cur_path = os.path.dirname(__file__)


class DataLoader():
    def __init__(self, data_util, data_args, train_args):
        self.am_batch_size = train_args.am_batch_size
        self.lm_batch_size = train_args.lm_batch_size
        self.feature_dim = train_args.feature_dim
        self.feature_max_length = train_args.feature_max_length

        self.pinyin_dict = data_args.pinyin_dict
        self.hanzi_dict = data_args.hanzi_dict
        self.lfr_m = data_args.lfr_m
        self.lfr_n = data_args.lfr_n

        self.acoustic_vocab_size, self.pinyin2index, self.index2pinyin = self.get_acoustic_vocab_list()
        self.language_vocab_size, self.word2index, self.index2word = self.get_language_vocab_list()

        self.data = data_util

        self.path_lst = self.data.path_lst
        self.pny_lst = self.data.pny_lst
        self.han_lst = self.data.han_lst
        self.shuffle = data_util.shuffle
        # 生成batch_size个索引
        self.indexes = [i for i in range(len(self.path_lst))]

    def pny2id(self, line):
        """
        拼音转向量 one-hot embedding，没有成功在vocab中找到索引抛出异常，交给上层处理
        :param line:
        :param vocab:
        :return:
        """
        try:
            line = line.strip()
            line = line.split(' ')
            ret = []
            for pin in line:
                id = self.pinyin2index[pin]
                ret.append(id)
            return ret
        except Exception as e:
            raise ValueError

    def han2id(self, line):
        """
        文字转向量 one-hot embedding，没有成功在vocab中找到索引抛出异常，交给上层处理
        :param line:
        :param vocab:
        :return:
        """
        try:
            line = line.strip()
            res = []
            for han in line:
                if han == Const.PAD_FLAG:
                    res.append(Const.PAD)
                elif han == Const.SOS_FLAG:
                    res.append(Const.SOS)
                elif han == Const.EOS_FLAG:
                    res.append(Const.EOS)
                else:
                    res.append(self.word2index[han])
            return res
        except Exception as e:
            raise ValueError

    # 声学模型, 语料库大小
    def get_acoustic_vocab_list(self):
        text = pd.read_table(os.path.join(self.pinyin_dict), header=None)
        symbol_list = text.iloc[:, 0].tolist()
        symbol_list.append('_')
        symbol_num = len(symbol_list)
        pinyin2index = dict([pinyin, index] for index, pinyin in enumerate(symbol_list))
        index2pinyin = dict([index, pinyin] for index, pinyin in enumerate(symbol_list))
        return symbol_num, pinyin2index, index2pinyin

    # 语言模型, 语料库大小
    def get_language_vocab_list(self):
        pd_data = pd.read_csv(os.path.join(home_dir, self.hanzi_dict), header=None)
        hanzi_list = pd_data.T.values.tolist()[0]
        word_list = [Const.PAD_FLAG]
        word_list.extend(hanzi_list)
        word_num = len(word_list)
        word2index = dict([word, index] for index, word in enumerate(word_list))
        index2word = dict([index, word] for index, word in enumerate(word_list))
        return word_num, word2index, index2word

    def data_generation(self, batch_datas, py_label_datas, han_label_datas):
        # batch_wav_data.shape = (10 1600 200 1), inputs_length.shape = (10,)
        batch_wav_data = np.zeros((self.am_batch_size, self.feature_max_length, 200, 1), dtype=np.float)
        # batch_label_data.shape = (10 64) ,label_length.shape = (10,)
        batch_label_data = np.zeros((self.am_batch_size, 64), dtype=np.int32)
        batch_han_data = np.zeros((self.am_batch_size, 64), dtype=np.int32)
        # length
        input_length = []
        label_length = []
        word_length = []
        error_count = []
        # 随机选取batch_size个wav数据组成一个batch_wav_data
        for i, path in enumerate(batch_datas):
            # Fbank特征提取函数(从feature_python)
            try:
                file1 = os.path.join(Const.SpeechDataPath, path)
                file2 = os.path.join(Const.NoiseOutPath, path)
                if os.path.isfile(file1):
                    signal, sample_rate = sf.read(file1)
                elif os.path.isfile(file2):
                    signal, sample_rate = sf.read(file2)
                else:
                    print("file path Error")
                    return 0
                fbank = compute_fbank_from_api(signal, sample_rate)
                input_data = fbank.reshape([fbank.shape[0], fbank.shape[1], 1])
                wav_length = input_data.shape[0]
                data_length = min(200, math.ceil(wav_length//8+1))
                seq = han_label_datas[i]
                seq_ids = np.array(self.han2id(seq))
                pinyin = py_label_datas[i]
                py_label_ids = np.array(self.pny2id(pinyin))
                len_label = len(py_label_ids)
                # 将错误数据进行抛出异常,并处理
                if wav_length > self.feature_max_length:
                    raise ValueError
                if len_label > 64 or len_label >= data_length:
                    raise ValueError
                input_length.append(data_length)
                label_length.append(len_label)
                word_length.append(len_label)
                batch_wav_data[i, 0:len(input_data)] = input_data
                batch_label_data[i, 0:len(py_label_ids)] = py_label_ids
                batch_han_data[i, 0:len(seq_ids)] = seq_ids
            except ValueError:
                error_count.append(i)
                continue
        # 删除异常语音信息
        if error_count != []:
            batch_wav_data = np.delete(batch_wav_data, error_count, axis=0)
            batch_label_data = np.delete(batch_label_data, error_count, axis=0)
            batch_han_data = np.delete(batch_han_data, error_count, axis=0)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        word_length = np.array(word_length)
        # CTC 输入长度0-1600//8+1
        # label label真实长度
        return batch_wav_data, input_length, batch_label_data, label_length, batch_han_data, word_length

    def get_lm_batch(self):
        '''
        训练语言模型batch数据，拼音到汉字
        :return:
        '''
        shuffle_list = [i for i in range(len(self.pny_lst))]
        if self.shuffle == True:
            shuffle(shuffle_list)
        batch_num = len(self.pny_lst) // self.lm_batch_size
        for k in range(batch_num):
            begin = k * self.lm_batch_size
            end = begin + self.lm_batch_size
            index_list = shuffle_list[begin:end]
            max_len = max([len(self.pny_lst[index].strip().split(' ')) for index in index_list])
            input_data = []
            label_data = []
            input_length = []
            for i in index_list:
                try:
                    py_vec = self.pny2id(self.pny_lst[i]) + [0] * (max_len - len(self.pny_lst[i].strip().split(' ')))
                    han_vec = self.han2id(self.han_lst[i]) + [0] * (max_len - len(self.han_lst[i].strip()))
                    input_data.append(py_vec)
                    label_data.append(han_vec)
                    input_length.append(len(self.pny_lst[i]))
                except ValueError:
                    continue
            input_data = np.array(input_data)
            label_data = np.array(label_data)
            input_length = np.array(input_length)
            yield input_data, input_length, label_data

    def get_lm_batch_old(self):
        '''
        训练语言模型batch数据，拼音到汉字
        :return:
        '''
        batch_num = len(self.pny_lst) // self.lm_batch_size
        for k in range(batch_num):
            begin = k * self.lm_batch_size
            end = begin + self.lm_batch_size
            input_batch = self.pny_lst[begin:end]
            label_batch = self.han_lst[begin:end]
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array(
                [self.pny2id(line) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array(
                [self.han2id(line) + [0] * (max_len - len(line)) for line in label_batch])
            yield input_batch, label_batch

    def get_fbank_and_pinyin_data(self, index):
        """
        获取一条语音数据的Fbank与拼音信息
        :param index: 索引位置
        :return:
            input_data: 语音特征数据
            data_length: 语音特征数据长度
            label: 语音标签的向量
        """
        try:
            # Fbank特征提取函数(从feature_python)
            file = os.path.join(Const.SpeechDataPath, self.path_lst[index])
            noise_file = os.path.join(Const.NoiseOutPath, self.path_lst[index])
            fbank = compute_fbank_from_file(file) if os.path.exists(file) else\
                compute_fbank_from_file(noise_file)
            wav_data = np.zeros((1, self.feature_max_length, 200, 1), dtype=np.float)
            input_data = fbank.reshape([fbank.shape[0], fbank.shape[1], 1])
            wav_data[0, 0:len(input_data)] = input_data
            data_length = input_data.shape[0] // 8 + 1
            label = self.pny2id(self.pny_lst[index])
            label = np.array(label)
            len_label = len(label)
            # 将错误数据进行抛出异常,并处理
            if wav_data.shape[0] > self.feature_max_length:
                raise ValueError
            if len_label > 64 or len_label > data_length:
                raise ValueError
            wav_data[0, 0:len(input_data)] = input_data
            data_length = np.array([data_length])
            return wav_data, data_length, label, len_label
        except ValueError:
            raise ValueError

    def am_generator(self):
        """
        一个batch数据生成器，充当fit_general的参数
        :return:
            inputs: 输入数据
            outputs: 输出结果
        """
        for i in range(len(self)):
            feature_input, logits_length, py_target, py_target_length, hanzi_target, hanzi_target_length = self.__getitem__(i)
            yield feature_input, logits_length, py_target, py_target_length, hanzi_target, hanzi_target_length

    def end2end_generator(self):
        """
        一个batch数据生成器，充当fit_general的参数
        :return:
            inputs: 输入数据
            outputs: 输出结果
        """
        for i in range(len(self)):
            t = self.__getitem__(i)
            yield t

    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.am_batch_size:(index + 1) * self.am_batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.path_lst[k] for k in batch_indexs]
        py_label_datas = [self.pny_lst[k] for k in batch_indexs]
        han_label_datas = [self.han_lst[k] for k in batch_indexs]
        # 生成数据
        feature_input, logits_length, py_target, py_target_length, han_target, han_length\
            = self.data_generation(batch_datas, py_label_datas, han_label_datas)
        return feature_input, logits_length, py_target, py_target_length, han_target, han_length

    def __len__(self):
        return len(self.path_lst) // self.am_batch_size


if __name__ == '__main__':
    pass