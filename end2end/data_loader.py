import os, sys
home_dir = os.getcwd()
from random import shuffle
from pathlib import Path
import pandas as pd

from util.data_util import DataUtil
from util.utils import *
from util.wav_util import *
from util.const import Const


class dataloader():
    def __init__(self, train_args, data_args):
        self.batch_size = train_args.batch_size
        self.feature_dim = train_args.feature_dim
        self.feature_max_length = train_args.feature_max_length
        self.mode = train_args.mode
        self.data_length = train_args.data_length
        self.shuffle = train_args.shuffle

        self.data_path = Const.SpeechDataPath
        self.thchs30 = data_args.thchs30
        self.aishell = data_args.aishell
        self.stcmd = data_args.stcmd
        self.aidatatang = data_args.aidatatang
        self.aidatatang_1505 = data_args.aidatatang_1505
        self.prime = data_args.prime
        self.noise = data_args.noise
        self.pinyin_dict = data_args.pinyin_dict
        self.hanzi_dict = data_args.hanzi_dict

        self.lfr_m = data_args.lfr_m
        self.lfr_n = data_args.lfr_n

        self.acoustic_vocab_size, self.pinyin2index, self.inde2pinyin = self.get_acoustic_vocab_list()
        self.language_vocab_size, self.word2index, self.index2word = self.get_language_vocab_list()

        self.data = DataUtil(data_args, train_args.batch_size, train_args.mode, train_args.data_length, train_args.shuffle)

        self.path_lst = self.data.path_lst
        self.pny_lst = self.data.pny_lst
        self.han_lst = self.data.han_lst

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
            return [self.pinyin2index[pin] for pin in line]
        except ValueError:
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
        except ValueError:
            raise ValueError

    def wav_padding(self, wav_data_lst):
        '''
        wav的padding操作
        :param wav_data_lst:
        :return:
        '''
        feature_dim = wav_data_lst[0].shape[1]
        # len(data)实际上就是求语谱图的第一维的长度，也就是n_frames
        wav_lens = np.array([len(data) for data in wav_data_lst])
        # 取一个batch中的最长
        wav_max_len = max(wav_lens)
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, feature_dim), dtype=np.float32)
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst, pad_idx):
        '''
        label的padding操作
        :param label_data_lst:
        :param pad_idx:
        :return:
        '''
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len), dtype=np.int32)
        new_label_data_lst += pad_idx
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

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
            file = os.path.join(self.data_path, self.path_lst[index])
            noise_file = Const.NoiseOutPath + self.path_lst[index]
            fbank = compute_fbank_from_file(file) if os.path.isfile(file) else\
                compute_fbank_from_file(noise_file)
            input_data = fbank.reshape([fbank.shape[0], fbank.shape[1], 1])
            data_length = input_data.shape[0] // 8 + 1
            label = self.pny2id(self.pny_lst[index])
            label = np.array(label)
            len_label = len(label)
            # 将错误数据进行抛出异常,并处理
            if input_data.shape[0] > self.feature_max_length:
                raise ValueError
            if len_label > 64 or len_label > data_length:
                raise ValueError
            return input_data, data_length, label, len_label
        except ValueError:
            raise ValueError

    def get_fbank_and_hanzi_data(self, index):
        '''
        获取一条语音数据的Fbank与拼音信息
        :param index: 索引位置
        :return: 返回相应信息
        '''
        try:
            # Fbank特征提取函数(从feature_python)
            file = os.path.join(self.data_path, self.path_lst[index])
            noise_file = Const.NoiseOutPath + self.path_lst[index]
            input_data = compute_fbank_from_file(file, self.feature_dim, sf_flag=True) if os.path.isfile(file) else \
                compute_fbank_from_file(noise_file, self.feature_dim, sf_flag=True)
            label = self.han2id(self.han_lst[index])
            input_label = [Const.SOS] + label
            target_label = label + [Const.EOS]
            # 将错误数据进行抛出异常,并处理
            return input_data, np.array(target_label), np.array(input_label)
        except ValueError:
            raise ValueError

    def get_am_batch(self):
        """
        一个batch数据生成器，充当fit_general的参数
        :return:
            inputs: 输入数据
            outputs: 输出结果
        """
        # 数据列表长度
        shuffle_list = [i for i in range(len(self.path_lst))]
        while True:
            if self.shuffle == True:
                shuffle(shuffle_list)
            # batch_wav_data.shape = (10 1600 200 1), inputs_length.shape = (10,)
            batch_wav_data = np.zeros((self.batch_size, self.feature_max_length, 200, 1), dtype=np.float)
            # batch_label_data.shape = (10 64) ,label_length.shape = (10,)
            batch_label_data = np.zeros((self.batch_size, 64), dtype=np.int64)
            # length
            input_length = []
            label_length = []
            error_count = []
            for i in range(len(self.path_lst) // self.batch_size):
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                for index in sub_list:
                    try:
                        # 随机选取一个batch
                        input_data, data_length, label, len_label, = self.get_fbank_and_pinyin_data(index)
                        input_length.append([data_length])
                        label_length.append([len_label])
                        batch_wav_data[i, 0:len(input_data)] = input_data
                        batch_label_data[i, 0:len_label] = label
                    except ValueError:
                        error_count.append(i)
                        continue

                # 删除异常语音信息
                if error_count != []:
                    batch_wav_data = np.delete(batch_wav_data, error_count, axis=0)
                    batch_label_data = np.delete(batch_label_data, error_count, axis=0)

                label_length = np.mat(label_length)
                input_length = np.mat(input_length)
                # CTC 输入长度0-1600//8+1
                # label label真实长度
                inputs = {'the_inputs': batch_wav_data,
                          'the_labels': batch_label_data,
                          'input_length': input_length,
                          'label_length': label_length,
                          }
                outputs = {'ctc': np.zeros((self.batch_size - len(error_count), 1), dtype=np.float32)}
                yield inputs, outputs

    def get_lm_batch(self):
        '''
        训练语言模型batch数据，拼音到汉字
        :return:
        '''
        shuffle_list = [i for i in range(len(self.pny_lst))]
        if self.shuffle == True:
            shuffle(shuffle_list)
        batch_num = len(self.pny_lst) // self.batch_size
        for k in range(batch_num):
            begin = k * self.batch_size
            end = begin + self.batch_size
            index_list = shuffle_list[begin:end]
            max_len = max([len(self.pny_lst[index].strip().split(' ')) for index in index_list])
            input_data = []
            label_data = []
            for i in index_list:
                try:
                    py_vec = self.pny2id(self.pny_lst[i])\
                             + [0] * (max_len - len(self.pny_lst[i].strip().split(' ')))
                    han_vec = self.han2id(self.han_lst[i]) + [0] * (max_len - len(self.han_lst[i].strip()))
                    input_data.append(py_vec)
                    label_data.append(han_vec)
                except ValueError:
                    continue
            input_data = np.array(input_data)
            label_data = np.array(label_data)
            yield input_data, label_data
        pass

    def get_lm_batch_old(self):
        '''
        训练语言模型batch数据，拼音到汉字
        :return:
        '''
        batch_num = len(self.pny_lst) // self.batch_size
        for k in range(batch_num):
            begin = k * self.batch_size
            end = begin + self.batch_size
            input_batch = self.pny_lst[begin:end]
            label_batch = self.han_lst[begin:end]
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array(
                [self.pny2id(line) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array(
                [self.han2id(line) + [0] * (max_len - len(line)) for line in label_batch])
            yield input_batch, label_batch

    def get_transformer_batch(self):
        '''
        # transformer训练batch数据
        :return:
        '''
        wav_length = len(self.path_lst)
        shuffle_list = [i for i in range(wav_length)]
        if self.shuffle == True:
            shuffle(shuffle_list)
        # 随机选取batch_size个wav数据组成一个batch_wav_data
        for i in range(wav_length // self.batch_size):
            # length
            wav_data_lst = []
            target_label_lst = []
            input_label_lst = []
            error_count = []
            begin = i * self.batch_size
            end = begin + self.batch_size
            sub_list = shuffle_list[begin:end]
            # 随机选取一个batch
            for index in sub_list:
                try:
                    input_data, target_label, input_label = self.get_fbank_and_hanzi_data(index)
                    input_data = build_LFR_features(input_data, self.lfr_m, self.lfr_n)
                    wav_data_lst.append(input_data)
                    target_label_lst.append(target_label)
                    input_label_lst.append(input_label)
                except ValueError:
                    error_count.append(i)
                    continue
            # label为decoder的输入，ground_truth为decoder的输出, label_data为decoder的输入
            pad_wav_data, input_length = self.wav_padding(wav_data_lst)
            pad_label_data, _ = self.label_padding(input_label_lst, Const.EOS)
            pad_target_data, _ = self.label_padding(target_label_lst, Const.IGNORE)
            # 删除异常语音信息
            if error_count != []:
                pad_wav_data = np.delete(pad_wav_data, error_count, axis=0)
                pad_label_data = np.delete(pad_label_data, error_count, axis=0)
                pad_target_data = np.delete(pad_target_data, error_count, axis=0)
            yield pad_wav_data, pad_label_data, pad_target_data

    def get_transformer_data_from_file(self, file):
        try:
            fbank = compute_fbank_from_file(file, feature_dim=self.feature_dim)
            input_data = build_LFR_features(fbank, self.lfr_m, self.lfr_n)
            input_data = np.expand_dims(input_data, axis=0)
            return input_data
        except ValueError:
            raise ValueError

    # 声学模型, 语料库大小
    def get_acoustic_vocab_list(self):
        text = pd.read_table(Path(home_dir).joinpath(self.pinyin_dict), header=None)
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
        word_list = Const.PAD_FLAG + ' ' + Const.SOS_FLAG + ' ' + Const.EOS_FLAG
        word_list = word_list.split(' ')
        word_list.extend(hanzi_list)
        word_num = len(word_list)
        word2index = dict([word, index] for index, word in enumerate(word_list))
        index2word = dict([index, word] for index, word in enumerate(word_list))
        return word_num, word2index, index2word

    def __len__(self):
        return len(self.path_lst) // self.batch_size


if __name__ == "__main__":

    pass