#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：SpeechDemo -> am_lm_model
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/8/4 11:07 AM
@Desc   ：
=================================================='''
import os
home_dir = os.getcwd()

from tensorflow.python.ops import math_ops as tf_math_ops
from keras import backend as K
from end2end.transformer import *


class CNNCTCModel():
    def __init__(self, args, acoustic_vocab_size, language_vocab_size):
        # 神经网络最终输出的每一个字符向量维度的大小
        # py_vocab
        self.acoustic_vocab_size = acoustic_vocab_size
        # hanzi_vocab
        self.language_vocab_size = language_vocab_size
        self.gpu_nums = args.gpu_nums
        self.lr = args.am_lr
        self.feature_dim = args.feature_dim
        self.feature_max_length = args.feature_max_length
        self.is_training = args.is_training
        self.hidden_units = args.hidden_units
        self.num_heads = args.num_heads
        self.num_blocks = args.num_blocks
        self.position_max_length = args.position_max_length
        self.dropout_rate = args.dropout_rate

        self.init_placeholder()
        self.build_model()
        if self.is_training:
            self.calc_loss()
            self.opt_init()

    def init_placeholder(self):
        self.wav_input = tf.placeholder(tf.float32, shape=[None, self.feature_max_length, self.feature_dim, 1], name='wav_input')
        self.wav_length = tf.placeholder(tf.int32, shape=[None], name='logits_length')
        self.target_py = tf.placeholder(tf.int32, shape=[None, None], name='target_py')
        self.target_py_length = tf.placeholder(tf.int32, shape=[None], name='target_py_length')
        self.target_hanzi = tf.placeholder(tf.int32, shape=[None, None], name='target_hanzi')
        self.target_hanzi_length = tf.placeholder(tf.int32, shape=[None], name='target_hanzi_length')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def build_model(self):
        # [None, L=200, 512]
        self.am_model()
        self.language_model()

    def am_model(self):
        self.h1 = cnn_cell(32, self.wav_input)
        self.h2 = cnn_cell(64, self.h1)
        self.h3 = cnn_cell(128, self.h2, nin_flag=True, nin_size=32)
        self.h4 = cnn_cell(128, self.h3, nin_flag=True, nin_size=32, pool=False)
        self.h5 = cnn_cell(128, self.h4, nin_flag=True, nin_size=32, pool=False)
        # [10, 200, 25, 128]
        shape_size = [-1, self.h5.get_shape().as_list()[1], self.h5.get_shape().as_list()[2] * self.h5.get_shape().as_list()[3]]
        self.h6 = tf.reshape(self.h5, shape=shape_size)
        self.h6 = dropout(self.h6, 0.3)
        self.h7 = dense(self.h6, 128)
        self.h7 = dropout(self.h7, 0.3)
        # [B, 200, 128]
        self.am_logits = dense(self.h7, self.acoustic_vocab_size, activation='softmax')

        with tf.variable_scope('am_loss'):
            # 这里input_length指的是网络softmax输出后的结果长度，也就是经过ctc计算的loss的输入长度。
            # 由于网络的时域维度由1600经过三个池化变成200，因此output的长度为200，因此input_length<=200
            self.sparse_labels = tf.cast(tf.contrib.layers.dense_to_sparse(self.target_py), tf.int32)  # 没有标记0
            self.am_logits = tf_math_ops.log(tf.transpose(self.am_logits, perm=[1, 0, 2]) + K.epsilon())
            self.am_loss = tf.expand_dims(tf.nn.ctc_loss_v2(self.target_py, self.am_logits, self.target_py_length,
                                                            self.wav_length, blank_index=self.acoustic_vocab_size - 1), 1)
            self.am_mean_loss = tf.reduce_mean(self.am_loss)
            self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.am_logits, self.target_py_length)
            self.lm_out = tf.sparse_to_dense(self.decoded[0], default_value=0)
            self.distance = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.sparse_labels)
            self.label_err = tf.reduce_mean(self.distance, name='label_error_rate')

    def language_model(self):
        # embedding
        # X的embedding + position的embedding
        lm_in = self.am_out
        position_emb = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(lm_in)[1]), 0), [tf.shape(lm_in)[0], 1]),
                                 vocab_size=self.position_max_length, num_units=self.hidden_units, zero_pad=False,
                                 scale=False, scope="enc_pe")
        self.enc = lm_in + position_emb

        # Dropout
        self.enc = tf.layers.dropout(self.enc,
                                     rate=self.dropout_rate,
                                     training=tf.convert_to_tensor(self.is_training))

        # Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                # Multihead Attention
                self.enc = multihead_attention(queries=self.enc,
                                               keys=self.enc,
                                               d_model=self.hidden_units,
                                               num_heads=self.num_heads,
                                               dropout_rate=self.dropout_rate,
                                               is_training=self.is_training,
                                               causality=False)

                # Feed Forward
                self.outputs = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units])

        # Final linear projection
        # self.outputs.shape=[B, 200, 6524]
        self.lm_logits = tf.layers.dense(self.outputs, self.language_vocab_size, activation='softmax')
        self.sparse_labels = tf.cast(tf.contrib.layers.dense_to_sparse(self.target_hanzi), tf.int32)  # 没有标记0
        self.lm_logits = tf_math_ops.log(tf.transpose(self.lm_logits, perm=[1, 0, 2]) + K.epsilon())
        self.lm_loss = tf.expand_dims(tf.nn.ctc_loss_v2(self.target_py, self.lm_logits, self.target_py_length,
                                                        self.wav_length, blank_index=self.acoustic_vocab_size - 1), 1)
        self.lm_mean_loss = tf.reduce_mean(self.lm_loss)
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.lm_logits, self.target_py_length)
        self.distance = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.sparse_labels)
        self.han_wer = tf.reduce_mean(self.distance, name='han_wer')
        tf.summary.scalar('acc', self.han_wer)

        # self.preds = tf.to_int32(tf.argmax(self.lm_logits, axis=-1))
        # self.istarget = tf.to_float(tf.not_equal(self.target_hanzi, Const.PAD))
        # self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.target_hanzi)) * self.istarget) / (
        #     tf.reduce_sum(self.istarget))
        # tf.summary.scalar('acc', self.acc)

    def opt_init(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
        # Summary
        tf.summary.scalar('mean_loss', self.mean_loss)
        self.summary = tf.summary.merge_all()

    def calc_loss(self):
        # Loss label平滑

        with tf.variable_scope('lm_loss'):
            # self.y_smoothed = label_smoothing(tf.one_hot(self.target_hanzi, depth=self.language_vocab_size))
            # loss计算，预测结果与平滑值的交叉熵
            # self.lm_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.lm_logits, labels=self.y_smoothed)
            # 平均loss
            # self.lm_mean_loss = tf.reduce_sum(self.lm_loss * self.istarget) / (tf.reduce_sum(self.istarget))
            # 合并loss
            self.mean_loss = self.am_mean_loss + self.lm_mean_loss


# ============================模型组件=================================
def conv1x1(input, filters):
    return tf.layers.conv2d(input, filters=filters, kernel_size=(1, 1), use_bias=True, activation='relu', padding='same')


def conv2d(input, filters):
    return tf.layers.conv2d(input, filters=filters, kernel_size=(3, 3), use_bias=True, activation='relu', padding='same')


def batch_norm(input):
    return tf.layers.batch_normalization(input)


def maxpool(input):
    return tf.layers.max_pooling2d(input, pool_size=(2, 2), strides=(2, 2), padding='valid')


def dense(input, units, activation='relu'):
    return tf.layers.dense(input, units, activation=activation, use_bias=True)


def dropout(inpux, rate):
    if rate == None:
        return inpux
    else:
        return tf.layers.dropout(inpux, rate)


def cnn_cell(size, x, nin_flag=False, nin_size=32, pool=True):
    x = batch_norm(conv2d(x, size))
    if nin_flag:
        x = nin_network(x, nin_size)
        x = batch_norm(conv2d(x, size))
    if pool:
        x = maxpool(x)
    return x


def nin_network(x, size):
    return batch_norm(conv1x1(x, size))


def global_avg_pool(x):
    return tf.layers.AveragePooling2D(x)