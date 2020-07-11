import os, sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import argparse
import logging
import warnings
import numpy as np
from datetime import datetime

from src.end2end.transformer import *
from src.end2end.data_loader import dataloader
from util.const import Const
from util.hparams import DataHparams

parser = argparse.ArgumentParser()
# 初始学习率为0.001,10epochs后设置为0.0001
parser.add_argument('--gpu_nums', default=1, type=int)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--lr', default=0.0001, type=int)
parser.add_argument('--is_training', default=True, type=bool)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--feature_max_length', default=1600, type=int)
parser.add_argument('--dimension', default=80, type=int)
parser.add_argument('--shuffle', default='shuffle', type=str)
parser.add_argument('--data_length', default=None, type=int)
parser.add_argument('--save_nums', default=3, type=int)
parser.add_argument('--save_path', default='./model', type=str)
parser.add_argument('--log_dir', default='./log', type=str)

# 声学模型参数
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--num_blocks', default=6, type=int)
parser.add_argument('--position_max_length', default=500, type=int)
parser.add_argument('--hidden_units', default=512, type=int)
parser.add_argument('--seq_length', default=16, type=int)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument('--feature_dim', default=80, type=int)
parser.add_argument('--beam_size', default=3, type=int)
parser.add_argument('--lp_alpha', default=0.6, type=int)
parser.add_argument('--max_target_length', default=50, type=int)
parser.add_argument('--summary_step', default=200, type=int)
parser.add_argument('--save_every_n', default=1000, type=int)
parser.add_argument('--log_every_n', default=5, type=int)
parser.add_argument('--learning_rate', default=0.01, type=int)
parser.add_argument('--min_learning_rate', default=1e-6, type=float)
parser.add_argument('--dacay_step', default=3000, type=float)

# 预测长度
parser.add_argument('--count', default=500, type=int)
parser.add_argument('--concat', default=4, type=int)
parser.add_argument('--death_rate', default=0.5, type=int)
parser.add_argument('--test', default=False, type=bool)
args = parser.parse_args()
data_args = DataHparams()

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

cur_path = os.path.dirname(__file__)


class transformerTrain():
    def __init__(self):
        self.data_loader = dataloader(args, data_args.args)
        self.model = Transformer_Model(args, self.data_loader)
        self.model_path = os.path.join(cur_path, args.save_path)
        self.log_dir = os.path.join(cur_path, args.log_dir)
        self.summary_step = args.summary_step
        self.save_every_n = args.save_every_n
        self.log_every_n = args.log_every_n
        self.learning_rate = args.learning_rate

    def train(self):
        logging.info("# Session")
        with tf.Session() as sess:
            self.model.build_transformer()
            saver = tf.train.Saver(max_to_keep=args.save_nums)
            ckpt = tf.train.latest_checkpoint(self.model_path)
            if ckpt is None:
                logging.info("Initializing from scratch")
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, ckpt)

            summary_writer = tf.compat.v1.summary.FileWriter(self.log_dir, sess.graph)
            train_steps = 0
            for epoch in range(args.epochs):
                batch_data = self.data_loader.get_transformer_batch()
                for inputs_data in batch_data:
                    train_steps += 1
                    feed = {self.model.x_input: inputs_data['the_inputs'],
                            self.model.y_input: inputs_data['the_labels'],
                            self.model.y_target: inputs_data['ground_truth'],
                            self.model.learning_rate: self.learning_rate}
                    train_loss, summary, lr, _ = sess.run([self.model.mean_loss, self.model.merged,
                                                           self.model.current_learning, self.model.train_op], feed_dict=feed)
                    # 每50000step计算一个验证集效果
                    if train_steps % self.summary_step == 0:
                        summary_writer.add_summary(summary, global_step=train_steps // self.summary_step)
                    if train_steps % self.save_every_n == 0:
                        dirpath = self.model_path
                        if not os.path.exists(dirpath):
                            os.mkdir(dirpath)
                        model_file = 'model_{}'.format(train_steps)
                        saver.save(sess, os.path.join(dirpath, model_file))
                        saver.save(sess, os.path.join(dirpath, 'final_model'))
                    if train_steps % self.log_every_n == 0:
                        now_time = datetime.now()
                        msg = 'Epoch: {0:>6}, Iter: {1:>6}, LR:{2:>10.6f} Train Loss: {3:>6.2}, Time: {4}'
                        print(msg.format(epoch, train_steps, lr, train_loss, now_time))


class Transformer_Model():
    def __init__(self, arg, data_loader):
        self.graph = tf.Graph()
        self.is_training = arg.is_training
        self.hidden_units = arg.hidden_units
        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.position_max_length = arg.position_max_length
        self.lr = arg.lr
        self.seq_length = arg.seq_length
        self.dropout_rate = arg.dropout_rate
        self.batch_size = arg.batch_size
        self.dimension = arg.dimension
        self.label_vocab_size = data_loader.language_vocab_size
        self.dacay_step = args.dacay_step
        self.min_learning_rate = args.min_learning_rate

    def build_transformer(self):
        """
        构建transformer结构，分为训练和预测两个逻辑
        :return:
        """
        self.x_input = tf.placeholder(tf.float32, shape=(self.batch_size, None, 4 * self.dimension))
        self.y_input = tf.placeholder(tf.int32, shape=(self.batch_size, None))
        self.y_target = tf.placeholder(tf.int32, shape=(self.batch_size, None))
        self.learning_rate = tf.placeholder(tf.float64)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if self.is_training:
            # X的embedding + position的embedding
            self.embedding_input()
            self.encoder()
            self.decoder()
            self.loss()
        else:
            self.embedding_input()
            self.encoder()
            self.predict_decoder()

    def embedding_input(self):
        x_position_emb = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x_input)[1]), 0), [tf.shape(self.x_input)[0], 1]),
                                   vocab_size=self.position_max_length, num_units=self.hidden_units, zero_pad=False,
                                   scale=False, scope="enc_pe")
        self.input_vec = tf.layers.dense(self.x_input, self.hidden_units)
        self.embedding_input = self.input_vec + x_position_emb

        y_position_emb = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.y_input)[1]), 0), [tf.shape(self.y_input)[0], 1]),
                                   vocab_size=self.position_max_length, num_units=self.hidden_units, zero_pad=False,
                                   scale=False, scope="dec_pe")
        self.y_input_emb = embedding(self.y_input, vocab_size=self.label_vocab_size, num_units=self.hidden_units, zero_pad=False,
                                     scale=False, scope="dec_input")
        self.y_input_emb = self.y_input_emb + y_position_emb


    def encoder(self):
        self.enc = tf.layers.dropout(self.embedding_input,
                                     rate=self.dropout_rate,
                                     training=tf.convert_to_tensor(self.is_training))

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
                self.memory = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units], reuse=tf.AUTO_REUSE)

    def decoder(self):
        self.memory = tf.layers.dropout(self.memory,
                                        rate=self.dropout_rate,
                                        training=tf.convert_to_tensor(self.is_training))
        self.dec = self.y_input_emb
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                # Multihead Attention
                self.dec = multihead_attention(queries=self.dec,
                                               keys=self.memory,
                                               d_model=self.hidden_units,
                                               num_heads=self.num_heads,
                                               dropout_rate=self.dropout_rate,
                                               is_training=self.is_training,
                                               causality=True,
                                               reuse=tf.AUTO_REUSE)

                # Feed Forward
                self.outputs = feedforward(self.dec, num_units=[4 * self.hidden_units, self.hidden_units], reuse=tf.AUTO_REUSE)


    def loss(self):
        # Final linear projection
        # [B, L, 6347]
        self.logits = tf.layers.dense(self.outputs, self.label_vocab_size)
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        # [B, L]
        self.istarget = tf.to_float(tf.not_equal(self.y_target, Const.PAD))
        # [B, L]
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y_target)) * self.istarget) / (tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)

        if self.is_training:
            # Loss label平滑
            self.y_smoothed = label_smoothing(tf.one_hot(self.y_target, depth=self.label_vocab_size))
            # loss计算，预测结果与平滑值的交叉熵
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
            # 平均loss
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
            # Training Scheme
            self.current_learning = tf.train.polynomial_decay(self.learning_rate, self.global_step,
                                                              self.dacay_step, self.min_learning_rate, power=0.5)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.current_learning)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()


    def predict_decoder(self):

        pass


if __name__ == '__main__':
    train = transformerTrain()
    train.train()
