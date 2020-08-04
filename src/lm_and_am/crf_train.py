import tensorflow as tf
import os
import warnings
import sys
import numpy as np
home_dir = os.getcwd()
sys.path.append(home_dir)
cur_path = os.path.dirname(__file__)

from src.lm_and_am.model.lstm_crf import CRF_Language_Model
from src.lm_and_am.data_loader import DataLoader
from util.hparams import AmLmHparams, DataHparams
from util.data_util import DataUtil
from util.const import Const

warnings.filterwarnings('ignore')

def train_language_model(data_args, am_hp):
    """
    语言模型
    :param train_data: 训练数据
    :return:
    """
    epochs = am_hp.epochs
    batch_size = am_hp.crf_batch_size

    data_util_train = DataUtil(data_args, batch_size=batch_size, mode='train', data_length=None, shuffle=True)
    dataloader = DataLoader(data_util_train, data_args, am_hp)
    batch_num = len(data_util_train.path_lst) // batch_size

    with tf.Graph().as_default():
        lm_model = CRF_Language_Model(am_hp, dataloader.acoustic_vocab_size, dataloader.language_vocab_size)
        lm_model.build_model()
        saver = tf.train.Saver(max_to_keep=5)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.85  # 占用GPU90%的显存

        with tf.Session(config=config) as sess:
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            add_num = 0
            if os.path.exists(Const.CRFLmModelFolder):
                latest = tf.train.latest_checkpoint(Const.CRFLmModelFolder)
                if latest != None:
                    print('loading language model...')
                    add_num = int(latest.split('_')[-2])
                    saver.restore(sess, latest)
            writer = tf.summary.FileWriter(Const.LmModelTensorboard, tf.get_default_graph())
            for k in range(epochs):
                total_loss = 0
                batch = dataloader.get_lm_batch()
                for i in range(batch_num):
                    input_batch, input_length, label_batch = next(batch)
                    # print(np.shape(input_batch))
                    # print(np.shape(input_length))
                    # print(np.shape(label_batch))
                    feed = {lm_model.input_x: input_batch,
                            lm_model.target_y: label_batch,
                            lm_model.seq_lengths: input_length,
                            lm_model.keep_prob: am_hp.keep_prob}
                    cost, _ = sess.run([lm_model.loss, lm_model.train_op], feed_dict=feed)
                    total_loss += cost
                    if i % 10 == 0:
                        print("epoch: %d step: %d/%d  train loss=6%f" % (k+1, i, batch_num, cost))
                rs = sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
                print('epochs', k + 1, ': average loss = ', total_loss / batch_num)
                saver.save(sess, Const.CRFLmModelFolder + 'model_%d_%.3f.ckpt' % (k + 1 + add_num, total_loss / batch_num))
            writer.close()


if __name__ == '__main__':
    params = AmLmHparams().args
    data_args = DataHparams.args
    print('//-----------------------start crf train language model-----------------------//')
    train_language_model(data_args, params)