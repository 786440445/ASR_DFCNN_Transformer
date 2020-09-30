# coding=utf-8
import random
import sys
import os
import tensorflow as tf
import warnings
import numpy as np
import datetime
home_dir = os.getcwd()
sys.path.append(home_dir)
from src.lm_and_am.model.acoustic_model3 import CNNCTCModel
from src.lm_and_am.model.language_model import Language_Model
from src.lm_and_am.data_loader import DataLoader

# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
from util.const import Const
from util.hparams import AmDataHparams, LmDataHparams, AmLmHparams
from util.utils import GetEditDistance
from util.data_util import DataUtil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


def speech_test(am_model, lm_model, am_sess, lm_sess, num):
    # 3. 进行测试-------------------------------------------
    num_data = len(dataloader.pny_lst)
    length = test_data_util.data_length
    if length == None:
        length = num_data
    ran_num = random.randint(0, length - 1)
    words_num = 0
    word_error_num = 0
    han_num = 0
    han_error_num = 0
    data = ''
    for i in range(num):
        print('\nthe ', i+1, 'th example.')
        # 载入训练好的模型，并进行识别
        index = (ran_num + i) % num_data
        # try:
        # 声学模型预测
        try:
            with am_sess.as_default():
                hanzi = dataloader.han_lst[index]
                hanzi_vec = [dataloader.word2index.get(word, Const.PAD) for word in hanzi]
                inputs, input_length, label, _ = dataloader.get_fbank_and_pinyin_data(index)
                decoded = am_sess.run([am_model.decoded[0]], feed_dict={
                    am_model.wav_input: inputs,
                    am_model.logits_length: input_length})
                pinyin_ids = tf.sparse_tensor_to_dense(decoded[0], default_value=0).eval(session=lm_sess)
                pinyin = ' '.join([dataloader.index2pinyin[k] for k in pinyin_ids[0]])
                y = dataloader.pny_lst[index]

            # 语言模型预测
            with lm_sess.as_default():
                py_in = np.array(pinyin_ids)
                lm_model.is_training = False
                lm_model.dropout_rate = 0
                han_pred = lm_sess.run(lm_model.preds, {lm_model.x: py_in})
                han = ''.join(dataloader.index2word.get(idx) for idx in han_pred[0])
        except ValueError:
            continue

        print('原文汉字结果:', ''.join(hanzi))
        print('原文拼音结果:', ''.join(y))
        print('预测拼音结果:', pinyin)
        print('预测汉字结果:', han)
        data += '原文汉字结果:' + ''.join(hanzi) + '\n'
        data += '原文拼音结果:' + ''.join(y) + '\n'
        data += '预测拼音结果:' + pinyin + '\n'
        data += '预测汉字结果:' + han + '\n'

        words_n = label.shape[0]
        words_num += words_n  # 把句子的总字数加上
        py_edit_distance = GetEditDistance(label, pinyin_ids[0])
        # 拼音距离
        if (py_edit_distance <= words_n):  # 当编辑距离小于等于句子字数时
            word_error_num += py_edit_distance  # 使用编辑距离作为错误字数
        else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
            word_error_num += words_n  # 就直接加句子本来的总字数就好了

        # 汉字距离
        words_n = np.array(hanzi_vec).shape[0]
        han_num += words_n  # 把句子的总字数加上
        han_edit_distance = GetEditDistance(np.array(hanzi_vec), han_pred[0])
        if (han_edit_distance <= words_n):  # 当编辑距离小于等于句子字数时
            han_error_num += han_edit_distance  # 使用编辑距离作为错误字数
        else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
            han_error_num += words_n  # 就直接加句子本来的总字数就好了

    data += '*[Test Result] Speech Recognition ' + 'test' + ' set 拼音 word accuracy ratio: ' + str(
        (1 - word_error_num / words_num) * 100) + '%'
    data += '*[Test Result] Speech Recognition ' + 'test' + ' set 汉字 word accuracy ratio: ' + str(
        (1 - han_error_num / han_num) * 100) + '%'
    with open(os.path.join(home_dir, Const.PredResultFolder, 'pred_log'), 'w', encoding='utf-8') as f:
        f.writelines(data)
    print('*[Test Result] Speech Recognition ' + 'test' + ' set 拼音 word accuracy ratio: ',
          (1 - word_error_num / words_num) * 100, '%')
    print('*[Test Result] Speech Recognition ' + 'test' + ' set 汉字 word accuracy ratio: ',
          (1 - han_error_num / han_num) * 100, '%')


if __name__ == '__main__':
    # 测试长度
    # 1. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试
    am_data_params = AmDataHparams().args
    lm_data_params = LmDataHparams().args

    # 2.声学模型-----------------------------------
    hparams = AmLmHparams()
    parser = hparams.parser
    am_hp = parser.parse_args()
    am_hp.dropout_rate = 0
    am_hp.is_training = False
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

    test_data_util = DataUtil(am_data_params, am_hp.am_batch_size, mode='test', data_length=None, shuffle=True)
    dataloader = DataLoader(test_data_util, am_data_params, am_hp)
    print('loading acoustic model...')
    am_graph = tf.Graph()
    with am_graph.as_default():
        am_model = CNNCTCModel(am_hp, dataloader.acoustic_vocab_size, dataloader.language_vocab_size)
        lm_saver = tf.train.Saver()
        am_sess = tf.Session(graph=am_graph, config=tf.ConfigProto(gpu_options=gpu_options))
        latest = tf.train.latest_checkpoint(Const.AmModelFolder)
        lm_saver.restore(am_sess, latest)

    # 3.语言模型-----------------------------------
    print('loading language model...')
    lm_model = Language_Model(am_hp, dataloader.acoustic_vocab_size, dataloader.language_vocab_size)
    lm_sess = tf.Session(graph=lm_model.graph, config=tf.ConfigProto(gpu_options=gpu_options))
    with lm_model.graph.as_default():
        saver = tf.train.Saver()
        latest = tf.train.latest_checkpoint(Const.LmModelFolder)
        saver.restore(lm_sess, latest)
        test_count = 5000
        speech_test(am_model, lm_model, am_sess, lm_sess, test_count)