import os
home_dir = os.getcwd()

from src.end2end.transformer import *
from tensorflow.python.ops import math_ops as tf_math_ops
from keras import backend as K
from keras.layers import Lambda
from tensorflow.python.ops import ctc_ops as ctc
import warnings
warnings.filterwarnings('ignore')

class CNNCTCModel():
    def __init__(self, args, acoustic_vocab_size, label_vocab_size):
        # 神经网络最终输出的每一个字符向量维度的大小
        self.acoustic_vocab_size = acoustic_vocab_size
        self.label_vocab_size = label_vocab_size
        self.gpu_nums = args.gpu_nums
        self.lm_lr = args.am_lr
        self.dacay_step = args.dacay_step
        self.min_learning_rate = args.min_learning_rate
        self.feature_dim = args.feature_dim
        self.feature_max_length = args.feature_max_length
        self.is_training = args.is_training
        self.init_placeholder()
        self.build_model()
        if self.is_training:
            self.calc_loss()
            self.opt_init()

    def init_placeholder(self):
        self.wav_input = tf.placeholder(tf.float32, shape=[None, self.feature_max_length, self.feature_dim, 1], name='wav_input')
        self.logits_length = tf.placeholder(tf.int32, shape=[None], name='logits_length')
        self.target_py = tf.placeholder(tf.int32, shape=[None, None], name='target_py')
        self.target_length = tf.placeholder(tf.int32, shape=[None], name='target_length')
        self.drop_rate = tf.placeholder(tf.float32, name='dropout_rate')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.initializer = None

    def build_model(self):
        # [B,1600,200,1]
        self.h1 = self.cnn_cell(32, self.wav_input, pool=True)
        self.h1 += self.squeeze_excitation_layer(self.h1, 32, 1)
        self.h1 = self.cnn_cell(32, self.h1, pool=False)

        # [B,800,100,32]
        self.h2 = self.cnn_cell(64, self.h1, pool=True)
        self.h2 += self.squeeze_excitation_layer(self.h2, 64, 2)
        self.h2 = self.cnn_cell(64, self.h2, pool=False)

        # [B,400,50,64]
        self.h3 = self.cnn_cell(128, self.h2, pool=True)
        self.h3 += self.squeeze_excitation_layer(self.h3, 128, 2)
        self.h3 = self.cnn_cell(128, self.h3, pool=False)

        # [B,200,25,128]
        self.h6 = self.cnn_cell(128, self.h3)
        self.h6 = self.cnn_cell(256, self.h6)

        shape_size = [-1, self.h6.get_shape().as_list()[1], self.h6.get_shape().as_list()[2] * self.h6.get_shape().as_list()[3]]
        self.h6 = tf.reshape(self.h6, shape=shape_size)
        self.h6 = self.dropout(self.h6)
        self.logits = self.dense(self.h6, self.acoustic_vocab_size, activation='softmax')
        self.logits = tf_math_ops.log(tf.transpose(self.logits, perm=[1, 0, 2]) + K.epsilon())
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.logits_length)

        self.sparse_labels = tf.cast(tf.contrib.layers.dense_to_sparse(self.target_py), tf.int32)#没有标记0
        self.distance = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.sparse_labels)
        self.label_err = tf.reduce_mean(self.distance, name='label_error_rate')
        tf.summary.scalar('accuracy', self.label_err)

    def calc_loss(self):
        # 这里input_length指的是网络softmax输出后的结果长度，也就是经过ctc计算的loss的输入长度。
        # 由于网络的时域维度由1600经过三个池化变成200，因此output的长度为200，因此input_length<=200\
        self.loss = tf.expand_dims(tf.nn.ctc_loss_v2(self.sparse_labels, self.logits, self.target_length, self.logits_length,
                                                     blank_index=self.acoustic_vocab_size-1), 1)
        # self.loss = Lambda(self.ctc_lambda, output_shape=(1,), name='ctc')([self.target_py, self.logits, self.logits_length, self.target_length])
        # self.loss = tf.expand_dims(ctc.ctc_loss(inputs=self.logits, labels=self.sparse_labels, sequence_length=self.logits_length), 1)
        self.mean_loss = tf.reduce_mean(self.loss)

    def opt_init(self):
        self.current_learning = tf.train.polynomial_decay(self.lm_lr, self.global_step,
                                                          self.dacay_step, self.min_learning_rate,
                                                          cycle=True, power=0.5)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.current_learning)
        self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
        tf.summary.scalar('mean_loss', self.mean_loss)
        self.summary = tf.summary.merge_all()


    def ctc_lambda(self, args):
        labels, y_pred, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

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
        return tf.layers.average_pooling2d(input, pool_size=(2, 2), strides=(2, 2), padding='valid')

    def dense(self, input, units, activation='relu'):
        return tf.layers.dense(input, units, activation=activation, use_bias=True, kernel_initializer=self.initializer)

    def dropout(self, inpux):
        return tf.layers.dropout(inpux, self.drop_rate)

    def cnn_cell(self, size, x, nin_flag=False, nin_size=32, pool=False):
        x = self.batch_norm(self.conv2d(x, size))
        if nin_flag:
            x = self.nin_network(x, nin_size)
            x = self.batch_norm(self.conv2d(x, size))
        if pool:
            x = self.maxpool(x)
        return x

    def Global_Average_Pooling(self, x):
        # w = x.get_shape().as_list()[1]
        # h = x.get_shape().as_list()[2]
        # return tf.layers.average_pooling2d(x, pool_size=(w, h), strides=(1, 1), padding='valid', name='Global_avg_pooling')
        return tf.reduce_mean(x, [1, 2], keep_dims=True)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio):
        squeeze = self.Global_Average_Pooling(input_x)
        excitation = self.dense(squeeze, units=out_dim / ratio, activation='relu')
        excitation = self.dense(excitation, units=out_dim, activation='sigmoid')
        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = tf.multiply(input_x, excitation)
        return scale

    def nin_network(self, x, size):
        return self.batch_norm(self.conv1x1(x, size))