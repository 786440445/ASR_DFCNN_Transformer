import tensorflow as tf
from util.const import Const
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood


class CRF_Language_Model():
    def __init__(self, arg, acoustic_vocab_size, language_vocab_size):
        self.is_training = arg.is_training
        self.hidden_units = arg.hidden_units
        self.input_vocab_size = acoustic_vocab_size
        self.label_vocab_size = language_vocab_size
        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.position_max_length = arg.position_max_length
        self.lr = arg.lm_lr
        self.dropout_rate = arg.dropout_rate

        self.embedding_size = arg.embedding_size
        self.hidden_dim = arg.hidden_dim
        self.clip_grad = 3

    def build_model(self):
        self.init_placeholder()
        self.inference()
        self.calc_loss()

    def init_placeholder(self):
        # 第一维大小为batch_size,第二维是句子的长度是动态获得的没法设置
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        self.target_y = tf.placeholder(tf.int32, [None, None], name='target_y')
        self.seq_lengths = tf.placeholder(tf.int32, shape=[None], name="seq_lengths")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    def inference(self):
        with tf.name_scope("embedding"):
            embedding_mat = tf.get_variable('lookup_table', dtype=tf.float32,
                                            shape=[self.input_vocab_size, self.embedding_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
            self.embedding_x = tf.nn.embedding_lookup(embedding_mat, self.input_x)

        with tf.variable_scope('lstm'):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embedding_x,
                                                                        self.seq_lengths, dtype=tf.float32)

            out_put = tf.concat([output_fw, output_bw], axis=-1)  # 对正反向的输出进行合并
            out_put = tf.nn.dropout(out_put, self.keep_prob)  # 防止过拟合

            self.logits = tf.layers.dense(out_put, self.label_vocab_size,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='logits')

        with tf.variable_scope('crf'):
            self.log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                             tag_indices=self.target_y,
                                                                             sequence_lengths=self.seq_lengths)

    def calc_loss(self):
        self.loss = -tf.reduce_mean(self.log_likelihood, name='loss')
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optim = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = optim.compute_gradients(self.loss)
        grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
        self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)



    def __init__(self, arg, acoustic_vocab_size, language_vocab_size):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.is_training = arg.is_training
            self.hidden_units = arg.hidden_units
            self.input_vocab_size = acoustic_vocab_size
            self.label_vocab_size = language_vocab_size
            self.num_heads = arg.num_heads
            self.num_blocks = arg.num_blocks
            self.position_max_length = arg.position_max_length
            self.lr = arg.lm_lr
            self.dropout_rate = arg.dropout_rate

            # input
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.int32, shape=(None, None))
            # embedding
            # X的embedding + position的embedding
            self.emb = embedding(self.x, vocab_size=self.input_vocab_size, num_units=self.hidden_units, scale=True,
                                 scope="enc_embed")
            position_emb = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                vocab_size=self.position_max_length, num_units=self.hidden_units, zero_pad=False, scale=False, scope="enc_pe")
            self.enc = self.emb + position_emb

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
            self.logits = tf.layers.dense(self.outputs, self.label_vocab_size)
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, Const.PAD))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)

            if self.is_training:
                # Loss label平滑
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.label_vocab_size))
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