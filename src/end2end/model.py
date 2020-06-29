import argparse

from src.end2end.transformer import *
from util.const import Const
from util.hparams import TransformerHparams
from util.hparams import DataHparams

from src.end2end.data_loader import dataloader
from util.data_util import language_vocab_size


parser = argparse.ArgumentParser()
# 初始学习率为0.001,10epochs后设置为0.0001
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--gpu_nums', default=1, type=int)
parser.add_argument('--is_training', default=True, type=bool)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--feature_dim', default=200, type=int)
parser.add_argument('--feature_max_length', default=1600, type=int)
parser.add_argument('--dimension', default=800, type=int)
parser.add_argument('--shuffle', default='shuffle', type=str)
parser.add_argument('--train_data_length', default=None, type=int)


train_args = parser.parse_args()
data_args = DataHparams()
transformer_args = TransformerHparams()

class transformerTrain():
    def __init__(self):
        self.data_loader = dataloader(train_args, data_args)
        self.model = Transformer_Model(train_args. transformer_args)

    def train(self):
        with tf.Session() as sess:
            total_loss = 0
            for epoch in range(train_args.epochs):
                batch_data = self.data_loader.get_transformer_batch()
                for x_input, y_input, y_target in batch_data:
                    feed = {self.model.x_input: x_input,
                            self.model.y_input: y_input,
                            self.model.y_target: y_target}
                    cost, _ = sess.run([self.model.mean_loss, self.model.train_op], feed_dict=feed)
                    total_loss += cost


class Transformer_Model():
    def __init__(self, arg):
        self.graph = tf.Graph()
        self.is_training = arg.is_training
        self.hidden_units = arg.hidden_units
        self.label_vocab_size = language_vocab_size
        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.position_max_length = arg.position_max_length
        self.lr = arg.lr
        self.dropout_rate = arg.dropout_rate
        self.batch_size = arg.batch_size
        self.dimension = arg.dimension

    def inference(self):
        # input
        self.x_input = tf.placeholder(tf.float32, shape=(self.batch_size, None, None))
        self.y_input = tf.placeholder(tf.float32, shape=(None, None))
        self.y_target = tf.placeholder(tf.float32, shape=(None, None))
        # X的embedding + position的embedding
        self.embedding_input()
        self.encoder()
        self.decoder()
        self.loss()

    def embedding_input(self):
        position_emb = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x_input)[1]), 0), [tf.shape(self.x_input)[0], 1]),
                                 vocab_size=self.position_max_length, num_units=self.hidden_units, zero_pad=False,
                                 scale=False, scope="enc_pe")
        self.embedding_input = self.x_input + position_emb

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
                                               causality=False)

                # Feed Forward
                self.memory = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units])

    def decoder(self):
        self.memory = tf.layers.dropout(self.memory,
                                        rate=self.dropout_rate,
                                        training=tf.convert_to_tensor(self.is_training))
        self.dec = self.y_input
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                # Multihead Attention
                self.dec = multihead_attention(queries=self.memory,
                                               keys=self.dec,
                                               d_model=self.hidden_units,
                                               num_heads=self.num_heads,
                                               dropout_rate=self.dropout_rate,
                                               is_training=self.is_training,
                                               causality=True)

                # Feed Forward
                self.outputs = feedforward(self.dec, num_units=[4 * self.hidden_units, self.hidden_units])


    def loss(self):
        # Final linear projection
        self.logits = tf.layers.dense(self.outputs, self.label_vocab_size)
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y_target, Const.PAD))
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
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    train = transformerTrain()
    train.train()
