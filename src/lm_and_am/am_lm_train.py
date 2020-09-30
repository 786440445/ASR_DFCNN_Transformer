#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：SpeechDemo -> am_lm_train
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/8/4 11:04 AM
@Desc   ：
=================================================='''

import tensorflow as tf
import keras
import os
import warnings
import sys
home_dir = os.getcwd()
sys.path.append(home_dir)
cur_path = os.path.dirname(__file__)
import numpy as np
from src.lm_and_am.model.am_lm_model import CNNCTCModel
from src.lm_and_am.data_loader import DataLoader
from util.hparams import AmLmHparams, AmDataHparams
from util.data_util import DataUtil
from util.const import Const

warnings.filterwarnings('ignore')


def train_model(data_args, am_hp):
    """
    声学模型
    :param train_data: 训练数据集合
    :param dev_data: 验证数据集合
    :return:
    """
    epochs = am_hp.epochs
    batch_size = am_hp.am_batch_size
    data_util_train = DataUtil(data_args, batch_size=batch_size, mode='train', data_length=None, shuffle=True)
    data_util_dev = DataUtil(data_args, batch_size=batch_size, mode='dev', data_length=None, shuffle=True)

    train_dataloader = DataLoader(data_util_train, data_args, am_hp)
    dev_dataloader = DataLoader(data_util_dev, data_args, am_hp)
    print(len(train_dataloader.path_lst))

    with tf.Graph().as_default():
        acoustic_model = CNNCTCModel(am_hp, train_dataloader.acoustic_vocab_size, train_dataloader.language_vocab_size)
        saver = tf.train.Saver(max_to_keep=5)
        # 数据读取处理部分
        dataset = tf.data.Dataset.from_generator(train_dataloader.end2end_generator,
                                                 output_types=(tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32))
        dataset = dataset.map(lambda x, y, z, w, m, n: (x, y, z, w, m, n), num_parallel_calls=64).prefetch(buffer_size=10000)

        with tf.Session() as sess:
            latest = tf.train.latest_checkpoint(Const.AmModelFolder)
            latest = None
            if latest != None:
                print('load acoustic model...')
                sess.load_model(latest)
            else:
                sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(Const.End2EndTensorboard, tf.get_default_graph())
            batch_nums = len(train_dataloader)
            old_wer = 1
            for epoch in range(epochs):
                total_loss = 0
                iterator_train = dataset.make_one_shot_iterator().get_next()
                for train_step in range(batch_nums):
                    input_x_batch, input_length_batch, pinyin_target, pinyin_length, word_target, word_length = \
                        sess.run(iterator_train)
                    feed = {acoustic_model.wav_input: input_x_batch,
                            acoustic_model.wav_length: input_length_batch,
                            acoustic_model.target_py: pinyin_target,
                            acoustic_model.target_py_length: pinyin_length,
                            acoustic_model.target_hanzi: word_target,
                            acoustic_model.target_hanzi_length: word_length}
                    mean_loss, label_err, han_wer, summary, _ = sess.run([
                        acoustic_model.lm_mean_loss,
                        acoustic_model.label_err,
                        acoustic_model.han_wer,
                        acoustic_model.summary,
                        acoustic_model.train_op], feed_dict=feed)
                    total_loss += mean_loss
                    if (train_step + 1) % 2 == 0:
                        print('epoch: {0:d}   step:{1:d}/{2:d}   average loss:{3:.4f}   label_err:{4:.4f}   acc:{5:.4f}'.format
                              (epoch+1, train_step+1, batch_nums, total_loss/(train_step+1), label_err, han_wer))
                writer.add_summary(summary)

                # 验证集测试
                total_wer = 0
                total_acc = 0
                total_loss = 0
                total_am_loss = 0
                eval_steps = len(dev_dataloader)
                for feature_input, logits_length, target_y, target_length in dev_dataloader:
                    feed = {acoustic_model.wav_input: feature_input,
                            acoustic_model.wav_length: logits_length,
                            acoustic_model.target_py: target_y,
                            acoustic_model.target_py_length: target_length,
                            acoustic_model.target_hanzi: word_target,
                            acoustic_model.target_hanzi_length: word_length}
                    mean_loss, label_err, acc = sess.run([
                        acoustic_model.lm_mean_loss,
                        acoustic_model.label_err,
                        acoustic_model.han_wer], feed_dict=feed)
                    total_wer += label_err
                    total_loss += mean_loss
                    total_acc += acc
                    total_am_loss += am_loss
                wer = total_wer/eval_steps
                acc = total_acc/eval_steps
                mean_loss = total_loss/eval_steps
                am_loss = total_am_loss/eval_steps
                print('epoch:%d   loss:%.4f   wer:%.4f   acc:%.4f' % (epoch+1, mean_loss, wer, acc))
                save_ckpt = "model_{epoch_d}-{val_loss_.2f}-{acc_.2f}.ckpt"
                saver.save(sess, os.path.join(home_dir, Const.End2EndModelFolder, save_ckpt % (epoch, mean_loss, acc)))
                if wer < old_wer:
                    saver.save(sess, os.path.join(home_dir, Const.End2EndModelFolder, 'final_model.ckpt'))
                    old_wer = wer


def main():
    print('//-----------------------start am_lm model-----------------------//')
    params = AmLmHparams().args
    data_args = AmDataHparams.args
    train_model(data_args, params)


if __name__ == '__main__':
    main()