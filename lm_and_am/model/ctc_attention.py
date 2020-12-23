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
        self.dacay_step = args.dacay_step
        self.min_learning_rate = args.min_learning_rate
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
            self.opt_init()

    def init_placeholder(self):
        self.wav_input = tf.placeholder(tf.float32, shape=[None, self.feature_max_length, self.feature_dim, 1], name='wav_input')
        self.wav_length = tf.placeholder(tf.int32, shape=[None], name='logits_length')
        self.target_hanzi = tf.placeholder(tf.int32, shape=[None, None], name='target_hanzi')
        self.target_hanzi_length = tf.placeholder(tf.int32, shape=[None], name='target_hanzi_length')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.initializer = None

    def build_model(self):
        # [None, L=200, 512]
        self.am_model()
        self.language_model()
        self.ctc_loss()

    def am_model(self):
        # # [B,500,320,1]
        # self.h1 = self.cnn_cell(32, self.wav_input, pool=True)
        # # [B,250,160,32]
        # self.h2 = self.cnn_cell(64, self.h1)
        # self.h3 = self.cnn_cell(64, self.h2)
        # self.h3 = self.maxpool(self.h2 + self.h3)
        # # [B,125,80,64]
        # self.h4 = self.cnn_cell(128, self.h3, nin_flag=True)
        # self.h5 = self.cnn_cell(128, self.h4, nin_flag=True) + self.h4
        # # [10, 125, 80, 128]
        # self.h5 = self.cnn_cell(256, self.h5, nin_flag=True)
        # # [10, 125, 80, 256]
        # self.h6 = self.dense(self.h5, 64)
        # # [B, 125, 80, 64]
        # shape_size = [-1, self.h6.get_shape().as_list()[1], self.h6.get_shape().as_list()[2] * self.h6.get_shape().as_list()[3]]
        # self.h6 = tf.reshape(self.h6, shape=shape_size)
        # [B,1600,200,1]
        self.h1 = self.cnn_cell(32, self.wav_input, pool=True)
        # [B,800,100,32]
        self.h2 = self.cnn_cell(64, self.h1, pool=True)
        # [B,400,50,64]
        self.h3 = self.cnn_cell(128, self.h2, pool=True)
        # [B,200,25,128]
        self.h4 = self.cnn_cell(128, self.h3)
        # [B,200,25,128]
        self.h5 = self.cnn_cell(128, self.h4)
        # [10, 200, 25, 128]
        shape_size = [-1, self.h5.get_shape().as_list()[1], self.h5.get_shape().as_list()[2] * self.h5.get_shape().as_list()[3]]
        self.h6 = tf.reshape(self.h5, shape=shape_size)
        # [B, 200, 3200]
        self.atten_in = self.dense(self.h6, 32)

    def language_model(self):
        # embedding
        # X的embedding + position的embedding
        lm_in = self.dense(self.atten_in, 512)
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
                                               causality=False,
                                               reuse=tf.AUTO_REUSE)

                # Feed Forward
                self.outputs = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units])

        # Final linear projection
        # self.outputs.shape=[B, 200, 6524]
        self.logits = tf.layers.dense(self.outputs, self.language_vocab_size, activation='softmax')

    def ctc_loss(self):
        # 这里input_length指的是网络softmax输出后的结果长度，也就是经过ctc计算的loss的输入长度。
        # 由于网络的时域维度由1600经过三个池化变成200，因此output的长度为200，因此input_length<=200
        self.sparse_labels = tf.cast(tf.contrib.layers.dense_to_sparse(self.target_hanzi), tf.int32)  # 没有标记0
        self.logits = tf_math_ops.log(tf.transpose(self.logits, perm=[1, 0, 2]) + K.epsilon())
        self.loss = tf.expand_dims(tf.nn.ctc_loss_v2(self.target_hanzi, self.logits, self.target_hanzi_length,
                                                     self.wav_length, blank_index=self.language_vocab_size - 1),
                                   1)
        self.mean_loss = tf.reduce_mean(self.loss)
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.target_hanzi_length)
        self.distance = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.sparse_labels)
        self.label_err = tf.reduce_mean(self.distance, name='label_error_rate')

    def opt_init(self):
        self.current_learning = tf.train.polynomial_decay(self.lr, self.global_step,
                                                          self.dacay_step, self.min_learning_rate,
                                                          cycle=True, power=0.5)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.current_learning, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
        # Summary
        tf.summary.scalar('mean_loss', self.mean_loss)
        self.summary = tf.summary.merge_all()

    # ============================模型组件=================================
    def conv1x1(self, input, filters):
        return tf.layers.conv2d(input, filters=filters, kernel_size=(1, 1),
                                use_bias=True, activation='relu',
                                padding='same', kernel_initializer=self.initializer)

    def conv2d(self, input, filters):
        return tf.layers.conv2d(input, filters=filters, kernel_size=(3, 3),
                                use_bias=True, activation='relu',
                                padding='same', kernel_initializer=self.initializer)

    def batch_norm(self, input):
        return tf.layers.batch_normalization(input)

    def maxpool(self, input):
        return tf.layers.max_pooling2d(input, pool_size=(2, 2), strides=(2, 2), padding='valid')

    def dense(self, input, units, activation='relu'):
        return tf.layers.dense(input, units, activation=activation, use_bias=True, kernel_initializer=self.initializer)

    def dropout(self, inpux):
        if self.dropout_rate == 0:
            return inpux
        else:
            return tf.layers.dropout(inpux, self.dropout_rate)

    def cnn_cell(self, size, x, nin_flag=False, nin_size=32, pool=False):
        x = self.batch_norm(self.conv2d(x, size))
        if nin_flag:
            x = self.nin_network(x, nin_size)
            x = self.batch_norm(self.conv2d(x, size))
        if pool:
            x = self.maxpool(x)
        return x

    def nin_network(self, x, size):
        return self.batch_norm(self.conv1x1(x, size))

    def global_avg_pool(self, x):
        return tf.layers.AveragePooling2D(x)