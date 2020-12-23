# coding=utf-8
import random
import sys
import os
import tensorflow as tf
import warnings
import numpy as np

home_dir = os.getcwd()
sys.path.append(home_dir)
from lm_and_am.model.language_model import Language_Model
from lm_and_am.data_loader import DataLoader

# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
from util.const import Const
from util.hparams import LmDataHparams, AmLmHparams
from util.utils import GetEditDistance
from util.data_util import DataUtil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


def speech_test(lm_model, num, sess):
    # 3. 进行测试-------------------------------------------
    num_data = len(dataloader.pny_lst)
    length = test_data_util.data_length
    if length == None:
        length = num_data
    ran_num = random.randint(0, length - 1)
    han_num = 0
    han_error_num = 0
    data = ''
    for i in range(num):
        print('\nthe ', i+1, 'th example.')
        # 载入训练好的模型，并进行识别
        index = (ran_num + i) % num_data
        try:
            hanzi = dataloader.han_lst[index]
            hanzi_vec = [dataloader.word2index.get(word, Const.PAD) for word in hanzi]
            pinyin = dataloader.pny_lst[index].split(' ')
            pinyin_vec = [dataloader.pinyin2index.get(word, Const.PAD) for word in pinyin]
            # 语言模型预测
            with sess.as_default():
                lm_model.is_training = False
                lm_model.dropout_rate = 0
                input_x = np.array([pinyin_vec])
                han_pred = sess.run(lm_model.preds, {lm_model.x: input_x})
                han = ''.join(dataloader.index2word.get(idx) for idx in han_pred[0])
        except ValueError:
            continue

        print('原文汉字结果:', ''.join(hanzi))
        print('预测汉子结果:', ''.join(han))
        data += '原文汉字结果:' + ''.join(hanzi) + '\n'
        data += '预测汉字结果:' + han + '\n'
        # 汉字距离
        words_n = np.array(hanzi_vec).shape[0]
        han_num += words_n  # 把句子的总字数加上
        han_edit_distance = GetEditDistance(np.array(hanzi_vec), han_pred[0])
        if (han_edit_distance <= words_n):  # 当编辑距离小于等于句子字数时
            han_error_num += han_edit_distance  # 使用编辑距离作为错误字数
        else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
            han_error_num += words_n  # 就直接加句子本来的总字数就好了

    data += '*[Test Result] Speech Recognition ' + 'test' + ' set 汉字 word accuracy ratio: ' + str(
        (1 - han_error_num / han_num) * 100) + '%'
    with open(os.path.join(home_dir, Const.PredResultFolder, 'pred_log'), 'w', encoding='utf-8') as f:
        f.writelines(data)
    print('*[Test Result] Speech Recognition ' + 'test' + ' set 汉字 word accuracy ratio: ',
          (1 - han_error_num / han_num) * 100, '%')


if __name__ == '__main__':
    # 测试长度
    # 1. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试
    lm_data_params = LmDataHparams().args

    # 2.声学模型-----------------------------------
    hparams = AmLmHparams()
    parser = hparams.parser
    am_hp = parser.parse_args()
    test_data_util = DataUtil(lm_data_params, am_hp.am_batch_size, mode='test', data_length=None, shuffle=False)
    dataloader = DataLoader(test_data_util, lm_data_params, am_hp)
    lm_model = Language_Model(am_hp, dataloader.acoustic_vocab_size, dataloader.language_vocab_size)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(graph=lm_model.graph, config=tf.ConfigProto(gpu_options=gpu_options))
    with lm_model.graph.as_default():
        print('loading language model...')
        saver = tf.train.Saver()
        latest = tf.train.latest_checkpoint(Const.LmModelFolder)
        saver.restore(sess, latest)
    test_count = 500
    speech_test(lm_model, test_count, sess)