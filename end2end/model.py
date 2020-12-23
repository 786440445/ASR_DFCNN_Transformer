import os, sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import argparse
import logging
import warnings
from datetime import datetime

from end2end.transformer import *
from end2end.data_loader import dataloader
from util.const import Const
from util.hparams import TransDataHparams

parser = argparse.ArgumentParser()
# 初始学习率为0.001,10epochs后设置为0.0001
parser.add_argument('--gpu_nums', default=1, type=int)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--is_training', default=True, type=bool)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--feature_max_length', default=1600, type=int)
parser.add_argument('--dimension', default=80, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--data_length', default=None, type=int)
parser.add_argument('--save_nums', default=3, type=int)
parser.add_argument('--save_path', default='./model', type=str)
parser.add_argument('--log_dir', default='./log', type=str)

# 声学模型参数
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--num_blocks', default=6, type=int)
parser.add_argument('--position_max_length', default=600, type=int)
parser.add_argument('--hidden_units', default=512, type=int)
parser.add_argument('--seq_length', default=16, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--feature_dim', default=80, type=int)
parser.add_argument('--beam_size', default=3, type=int)
parser.add_argument('--lp_alpha', default=0.6, type=int)
parser.add_argument('--max_target_length', default=50, type=int)

parser.add_argument('--summary_step', default=200, type=int)
parser.add_argument('--save_every_n', default=1000, type=int)
parser.add_argument('--log_every_n', default=2, type=int)
parser.add_argument('--learning_rate', default=0.0005, type=int)
parser.add_argument('--min_learning_rate', default=1e-6, type=float)
parser.add_argument('--dacay_step', default=5000, type=float)

# 预测长度
parser.add_argument('--count', default=500, type=int)
parser.add_argument('--concat', default=4, type=int)
parser.add_argument('--death_rate', default=0.5, type=int)
parser.add_argument('--test', default=False, type=bool)
args = parser.parse_args()
data_args = TransDataHparams()

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
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        # config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
        with tf.compat.v1.Session(config=config) as sess:
            self.model.build_transformer()
            saver = tf.train.Saver(max_to_keep=args.save_nums)
            ckpt = tf.train.latest_checkpoint(self.model_path)
            if ckpt is None:
                logging.info("Initializing from scratch")
                sess.run(tf.global_variables_initializer())
                # sess.run(tf.local_variables_initializer())
            else:
                saver.restore(sess, ckpt)

            summary_writer = tf.compat.v1.summary.FileWriter(self.log_dir, sess.graph)

            # 数据读取处理部分
            dataset = tf.data.Dataset.from_generator(self.data_loader.get_transformer_batch,
                                                     output_types=(tf.float32, tf.int32, tf.int32))
            dataset = dataset.map(lambda x, y, z: (x, y, z), num_parallel_calls=32).prefetch(buffer_size=10000)
            batch_nums = len(self.data_loader)
            train_steps = 0
            for epoch in range(args.epochs):
                iterator_train = dataset.make_one_shot_iterator().get_next()
                total_loss = 0
                for train_step in range(batch_nums):
                    train_steps += 1
                    the_inputs, the_labels, ground_truth = sess.run(iterator_train)
                    feed = {self.model.x_input: the_inputs,
                            self.model.y_input: the_labels,
                            self.model.y_target: ground_truth,
                            self.model.learning_rate: self.learning_rate}
                    train_loss, summary, lr, _ = sess.run([self.model.mean_loss, self.model.merged,
                                                           self.model.current_learning, self.model.train_op], feed_dict=feed)
                    total_loss += train_loss
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
                        msg = 'Epoch: {0:>3}, Iter: {1:>6}, LR:{2:>10.6f} Average Loss: {3:>6.6f}, Time: {4}'
                        average_loss = total_loss / (train_step + 1)
                        print(msg.format(epoch+1, train_steps, lr, average_loss, now_time))



    def eval(self):

        pass

def dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth) or (N, num_heads, seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth) or (N, num_heads, seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth) or (N, num_heads, seq_len_v, depth)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
            shape == (N, 1, 1, seq_len_k) [用在encoder和docoder block2，后者seq_len_q和seq_len_k不同] or (N, 1, seq_len_q, seq_len_k) [用在decoder block1，三个len相同]
    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (N, heads, seq_q, d_k)*(N, heads, d_k, seq_k)=(N, heads, seq_q, seq_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.cast(tf.math.sqrt(dk),matmul_qk.dtype)
    # print('shape'+str(scaled_attention_logits.shape.as_list()))

    # add the mask to the scaled tensor.
    # -inf经过softmax后会接近于0，一方面保证句子里pad的部分几乎不会分到注意力，另一方面也保证解码的时候当前时间步后面的部分句子不会分配到注意力
    if mask is not None:
        scaled_attention_logits += tf.cast((mask * -1e9),scaled_attention_logits.dtype) # mask中padding部分是1,使得logits变成-inf

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # FIXME: 可能不需要dropout https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/attention.py#L83
    # attention_weights = tf.keras.layers.Dropout(rate=0.1)(attention_weights)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth) 实际上是(..., seq_len_q, depth)，只是三种len都一样

    return output, attention_weights

class Transformer_Model():
    def __init__(self, arg, data_loader):
        self.graph = tf.Graph()
        self.is_training = arg.is_training
        self.hidden_units = arg.hidden_units
        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.position_max_length = arg.position_max_length
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
        self.learning_rate = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if self.is_training:
            # X的embedding + position的embedding
            self.pre_net()
            self.embedding_input()
            self.encoder()
            self.decoder()
            self.loss()
        else:
            self.pre_net()
            self.embedding_input()
            self.encoder()
            self.predict_decoder()

    def pre_net(self):
        # downsample
        # print(self.x_input)
        input = tf.expand_dims(self.x_input, -1)
        print('before ds.shape', input.shape)
        input_x1 = tf.layers.conv2d(input, 64, 3, 2, 'same', activation='tanh', kernel_initializer='glorot_normal')
        input_x1 = tf.layers.batch_normalization(input_x1, training=self.is_training)
        input_x2 = tf.layers.conv2d(input_x1, 64, 3, 2, 'same', activation='tanh', kernel_initializer='glorot_normal')
        input_x2 = tf.layers.batch_normalization(input_x2, training=self.is_training)
        print('downsample.shape:', input_x2.shape)

        for i in range(2):
            residual = input_x2
            q = tf.layers.conv2d(input_x2, 64, 3, 1, 'same', kernel_initializer='glorot_normal')
            q = tf.layers.batch_normalization(q, training=self.is_training)
            k = tf.layers.conv2d(input_x2, 64, 3, 1, 'same', kernel_initializer='glorot_normal')
            k = tf.layers.batch_normalization(k, training=self.is_training)
            v = tf.layers.conv2d(input_x2, 64, 3, 1, 'same', kernel_initializer='glorot_normal')
            v = tf.layers.batch_normalization(v, training=self.is_training)
            # shape = [B, L, features, channels]
            q_time = tf.transpose(q, [0, 3, 1, 2])
            k_time = tf.transpose(k, [0, 3, 1, 2])
            v_time = tf.transpose(v, [0, 3, 1, 2])

            # print(q_time)
            # print(k_time)
            # print(v_time)

            q_fre = tf.transpose(q, [0, 3, 2, 1])
            k_fre = tf.transpose(k, [0, 3, 2, 1])
            v_fre = tf.transpose(v, [0, 3, 2, 1])

            # print(q_fre)
            # print(k_fre)
            # print(v_fre)

            scaled_attention_time, time_attention_weights = dot_product_attention(q_time, k_time, v_time, mask=False)  # B*c*T*D
            scaled_attention_fre, fre_attention_weights = dot_product_attention(q_fre, k_fre, v_fre, mask=False)

            scaled_attention_time = tf.transpose(scaled_attention_time, [0, 2, 3, 1])
            scaled_attention_fre = tf.transpose(scaled_attention_fre, [0, 3, 2, 1])
            out = tf.concat([scaled_attention_time, scaled_attention_fre], -1)  # B*T*D*2c

            out = layer_norm(tf.layers.conv2d(out, 64, 3, 1, 'same', kernel_initializer='glorot_normal') + residual) # B*T*D*n

            final_out1 = tf.layers.conv2d(out, 64, 3, 1, 'same', activation='relu', kernel_initializer='glorot_normal')
            final_out1 = tf.layers.batch_normalization(final_out1, training=self.is_training)

            final_out2 = tf.layers.conv2d(final_out1, 64, 3, 1, 'same', kernel_initializer='glorot_normal')
            final_out2 = tf.layers.batch_normalization(final_out2, training=self.is_training)
            self.pre_out = tf.keras.layers.Activation('relu')(final_out2 + out)


    def embedding_input(self):
        print('pre_out', self.pre_out.shape)
        pre_out = tf.transpose(self.pre_out, [1, 0, 2, 3])
        shape = pre_out.get_shape().as_list()
        pre_out = tf.reshape(pre_out, [-1, shape[1], shape[2]*shape[3]])
        input_x = tf.transpose(pre_out, [1, 0, 2])
        print('input_x', input_x.shape)
        x_position_emb = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(input_x)[1]), 0), [tf.shape(input_x)[0], 1]),
                                   vocab_size=self.position_max_length, num_units=self.hidden_units, zero_pad=False,
                                   scale=False, scope="enc_pe")
        self.input_vec = tf.layers.dense(input_x, self.hidden_units, activation='relu')
        self.input_vec = layer_norm(self.input_vec)
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
        print(self.enc)
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
                self.memory = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units],
                                          dropout_rate=self.dropout_rate,
                                          is_training=self.is_training,
                                          reuse=tf.AUTO_REUSE)

    def decoder(self):
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
                self.outputs = feedforward(self.dec, num_units=[4 * self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate,
                                           is_training=self.is_training,
                                           reuse=tf.AUTO_REUSE)

    def predict_decoder(self):
        outputs = self.memory
        dec_slf_attn_list, dec_enc_attn_list = [], []
        # Get Deocder Input and Output
        ys_in_pad, ys_out_pad = self.preprocess(padded_input)

        pass

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
            #
            self.current_learning = tf.train.polynomial_decay(self.learning_rate, self.global_step,
                                                              self.dacay_step, self.min_learning_rate,
                                                              cycle=True, power=0.5)
            #
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.current_learning)
            # self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
            # self.current_learning = self.learning_rate
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.current_learning, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()



if __name__ == '__main__':
    train = transformerTrain()
    train.train()
