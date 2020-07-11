import os
home_dir = os.getcwd()

print(home_dir)
from pathlib import Path
import difflib
import numpy as np
from keras import backend as K
import pandas as pd
import tensorflow as tf
from util.hparams import DataHparams
from util.const import Const


hparams = DataHparams()
parser = hparams.parser
hp = parser.parse_args()


def wav_padding(wav_data_lst):
    feature_dim = wav_data_lst[0].shape[1]
    # len(data)实际上就是求语谱图的第一维的长度，也就是n_frames
    wav_lens = np.array([len(data) for data in wav_data_lst])
    # 取一个batch中的最长
    wav_max_len = max(wav_lens)
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, feature_dim), dtype=np.float32)
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens


def label_padding(label_data_lst, pad_idx):
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = max(label_lens)
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len), dtype=np.int32)
    new_label_data_lst += pad_idx
    for i in range(len(label_data_lst)):
        new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
    return new_label_data_lst, label_lens


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else:
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


def downsample(feature, contact):
    add_len = (contact - feature.shape[0] % contact) % contact
    pad_zero = np.zeros((add_len, feature.shape[1]), dtype=np.float)
    feature = np.append(feature, pad_zero, axis=0)
    feature = np.reshape(feature, (feature.shape[0] / 4, feature.shape[1] * 4))
    return feature


# 声学模型, 语料库大小
def get_acoustic_vocab_list():
    text = pd.read_table(Path(home_dir).joinpath(hp.pinyin_dict), header=None)
    symbol_list = text.iloc[:, 0].tolist()
    symbol_list.append('_')
    symbol_num = len(symbol_list)
    pinyin2index = dict([pinyin, index] for index, pinyin in enumerate(symbol_list))
    index2pinyin = dict([index, pinyin] for index, pinyin in enumerate(symbol_list))
    return symbol_num, pinyin2index, index2pinyin


# 语言模型, 语料库大小
def get_language_vocab_list():
    pd_data = pd.read_csv(Path(home_dir).joinpath(hp.hanzi_dict), header=None)
    hanzi_list = pd_data.T.values.tolist()[0]
    word_list = Const.PAD_FLAG + ' ' + Const.SOS_FLAG + ' ' + Const.EOS_FLAG
    word_list = word_list.split(' ')
    word_list.extend(hanzi_list)
    word_num = len(word_list)
    word2index = dict([word, index] for index, word in enumerate(word_list))
    index2word = dict([index, word] for index, word in enumerate(word_list))
    return word_num, word2index, index2word


# word error rate------------------------------------
def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost


# 定义解码器------------------------------------
def decode_ctc(num_result, input_length):
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype=np.int32)
    in_len[0] = input_length
    r = K.ctc_decode(result, in_len, greedy=True, beam_width=100, top_paths=1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    r1 = r[0][0].eval(session=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    tf.reset_default_graph()  # 然后重置tf图，这句很关键
    r1 = r1[0]
    return r1