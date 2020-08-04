import os
home_dir = os.getcwd()

from src.end2end.transformer import *


class CNNCTCModel():
    def __init__(self, args, acoustic_vocab_size, label_vocab_size):
        # 神经网络最终输出的每一个字符向量维度的大小
        self.acoustic_vocab_size = acoustic_vocab_size
        self.label_vocab_size = label_vocab_size
        self.gpu_nums = args.gpu_nums
        self.lr = args.am_lr
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
        self.target_py = tf.sparse_placeholder(tf.int32)
        self.target_length = tf.placeholder(tf.int32, shape=[None], name='target_length')

    def build_model(self):
        self.h1 = cnn_cell(32, self.wav_input)
        self.h2 = cnn_cell(64, self.h1)
        self.h3 = cnn_cell(128, self.h2)
        self.h4 = cnn_cell(128, self.h3, pool=False)
        self.h5 = cnn_cell(128, self.h4, pool=False)
        # [10, 200, 25, 128]
        shape_size = [-1, self.h5.get_shape().as_list()[1], self.h5.get_shape().as_list()[2] * self.h5.get_shape().as_list()[3]]
        self.h6 = tf.reshape(self.h5, shape=shape_size)
        self.h6 = dropout(self.h6, 0.3)
        self.h7 = dense(self.h6, 128)
        self.h6 = dropout(self.h7, 0.3)
        self.logits = dense(self.h7, self.acoustic_vocab_size, activation='softmax')
        # # 采用全局平均池化代替Dense
        # self.h6 = nin(self.h5, self.vocab_size)
        # # [10, 200, 25, 1424]
        # self.h7 = global_avg_pool(self.h6)
        # self.model.outputs = tf.nn.softmax(self.h7)

    def calc_loss(self):
        # 这里input_length指的是网络softmax输出后的结果长度，也就是经过ctc计算的loss的输入长度。
        # 由于网络的时域维度由1600经过三个池化变成200，因此output的长度为200，因此input_length<=200
        logits = tf.reshape(self.logits, shape=[1, 0, 2])
        self.loss = tf.nn.ctc_loss(self.target_py, logits, self.target_length, self.logits_length,
                                      blank_index=self.acoustic_vocab_size-1)
        self.mean_loss = tf.reduce_mean(self.loss)
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.target_length)
        self.wrong_nums = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.target_py)

    def opt_init(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
        # Summary
        tf.summary.scalar('mean_loss', self.mean_loss)
        self.summary = tf.summary.merge_all()

    def load_model(self, model, sess):
        sess.load_model(os.path.join(home_dir, 'model_and_log\\logs_am\\checkpoint', model + '.ckpt'))
        return sess

# ============================模型组件=================================
def conv1x1(input, filters):
    return tf.layers.conv2d(input, filters=filters, kernel_size=(1, 1),
                            use_bias=True, activation='relu',
                            padding='same', kernel_initializer='he_normal')


def conv2d(input, filters):
    return tf.layers.conv2d(input, filters=filters, kernel_size=(3, 3),
                            use_bias=True, activation='relu',
                            padding='same', kernel_initializer='he_normal')


def batch_norm(input):
    return tf.layers.batch_normalization(input)


def maxpool(input):
    return tf.layers.max_pooling2d(input, pool_size=(2, 2), strides=(2, 2), padding='valid')


def dense(input, units, activation='relu'):
    return tf.layers.dense(input, units, activation=activation, use_bias=True, kernel_initializer='he_normal')


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