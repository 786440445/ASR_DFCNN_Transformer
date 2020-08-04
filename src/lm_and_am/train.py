import tensorflow as tf
import keras
import os
import warnings
import sys
home_dir = os.getcwd()
sys.path.append(home_dir)
cur_path = os.path.dirname(__file__)
import numpy as np
from src.lm_and_am.model.acoustic_model import CNNCTCModel
from src.lm_and_am.model.language_model import Language_Model
from src.lm_and_am.data_loader import DataLoader
from util.hparams import AmLmHparams, DataHparams
from util.data_util import DataUtil
from util.const import Const

warnings.filterwarnings('ignore')


def train_acoustic_model(data_args, am_hp):
    """
    声学模型
    :param train_data: 训练数据集合
    :param dev_data: 验证数据集合
    :return:
    """
    epochs = am_hp.epochs
    batch_size = am_hp.batch_size
    data_util_train = DataUtil(data_args, batch_size=batch_size, mode='train', data_length=None, shuffle=True)
    data_util_dev = DataUtil(data_args, batch_size=batch_size, mode='dev', data_length=None, shuffle=True)

    train_dataloader = DataLoader(data_util_train, data_args, am_hp)
    dev_dataloader = DataLoader(data_util_dev, data_args, am_hp)
    print(len(train_dataloader.path_lst))

    with tf.Graph().as_default():
        acoustic_model = CNNCTCModel(am_hp, train_dataloader.acoustic_vocab_size, train_dataloader.language_vocab_size)
        saver = tf.train.Saver(max_to_keep=5)
        with tf.Session() as sess:
            latest = tf.train.latest_checkpoint(Const.AmModelFolder)
            if latest != None:
                print('load acoustic model...')
                sess.load_model(latest)
                sess.run(tf.global_variables_initializer())
            train_steps = len(train_dataloader)
            old_wer = 0
            train_step = 0
            for epoch in range(epochs):
                print('训练长度', train_steps)
                print(train_dataloader[0])
                for (feature_input, logits_length, sparse_target_y, target_length) in train_dataloader:
                    print(1)
                    print(np.shape(feature_input))
                    print(np.shape(logits_length))
                    print(np.shape(target_y))
                    print(np.shape(target_length))
                    feed = {acoustic_model.wav_input: feature_input,
                            acoustic_model.logits_length: logits_length,
                            acoustic_model.target_py: target_y,
                            acoustic_model.target_length: target_length}
                    mean_loss, summary, _ = sess.run([acoustic_model.mean_loss,
                                                      acoustic_model.summary,
                                                      acoustic_model.train_op], feed_dict=feed)

                    if (train_step + 1) % 10 == 0:
                        print('epoch: ', epoch + 1, ': average loss = ', mean_loss)
                # 测试集测试
                total_wrong_words = 0
                total_words = 0
                total_loss = 0
                eval_steps = len(dev_dataloader) // batch_size
                for feature_input, logits_length, target_y, target_length in dev_dataloader:
                    feed = {acoustic_model.wav_input: feature_input,
                            acoustic_model.logits_length: logits_length,
                            acoustic_model.target_py: target_y,
                            acoustic_model.target_length: target_length}
                    mean_loss, wrong_nums = sess.run([acoustic_model.mean_loss, acoustic_model.wrong_nums], feed_dict=feed)
                    for index, error_words in enumerate(wrong_nums):
                        total_wrong_words += error_words
                        total_words += target_length[index]
                        total_loss += mean_loss
                wer = total_wrong_words / total_words
                mean_loss = total_loss/eval_steps
                save_ckpt = "model_{epoch:02d}-{val_loss:.2f}-{acc:.2f}.ckpt"
                saver.save(sess, os.path.join(home_dir, Const.LmModelFolder, save_ckpt % (epoch, mean_loss, wer)))
                print('epoch: ', epoch + 1, ': average loss = ', mean_loss)
                if wer < old_wer:
                    saver.save(sess, os.path.join(home_dir, Const.LmModelFolder, 'final_model.ckpt'))
                    old_wer = wer


def train_language_model(data_args, am_hp):
    """
    语言模型
    :param train_data: 训练数据
    :return:
    """
    epochs = am_hp.epochs
    batch_size = am_hp.batch_size

    data_util_train = DataUtil(data_args, batch_size=batch_size, mode='train', data_length=None, shuffle=True)
    data_util_eval = DataUtil(data_args, batch_size=batch_size, mode='dev', data_length=None, shuffle=True)
    dataloader = DataLoader(data_util_train, data_args, am_hp)
    dataloader_eval = DataLoader(data_util_eval, data_args, am_hp)
    lm_model = Language_Model(am_hp, dataloader.acoustic_vocab_size, dataloader.language_vocab_size)
    batch_num = len(data_util_train.path_lst) // batch_size
    eval_batch_num = len(data_util_eval.path_lst) // batch_size

    with lm_model.graph.as_default():
        saver = tf.train.Saver(max_to_keep=5)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.85  # 占用GPU90%的显存

    with tf.Session(graph=lm_model.graph, config=config) as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        add_num = 0
        if os.path.exists(Const.LmModelFolder):
            latest = tf.train.latest_checkpoint(Const.LmModelFolder)
            if latest != None:
                print('loading language model...')
                saver.restore(sess, latest)
                add_num = int(latest.split('_')[-1])
        writer = tf.summary.FileWriter(Const.LmModelTensorboard, tf.get_default_graph())
        old_acc = 0
        for epoch in range(epochs):
            total_loss = 0
            batch = dataloader.get_lm_batch()
            for i in range(batch_num):
                input_batch, _, label_batch = next(batch)
                feed = {lm_model.x: input_batch, lm_model.y: label_batch}
                cost, _ = sess.run([lm_model.mean_loss, lm_model.train_op], feed_dict=feed)
                total_loss += cost
                if i % 10 == 0:
                    print("epoch: %d step: %d/%d  train loss=%.6f" % (epoch + 1, i, batch_num, cost))
            summary = sess.run(merged, feed_dict=feed)
            writer.add_summary(summary, epoch * batch_num + i)
            print('epochs', epoch+1, ': average loss = ', total_loss / batch_num)
            saver.save(sess, Const.LmModelFolder + 'model_%d_%.3f.ckpt' % (epoch + 1 + add_num, total_loss / batch_num))
            ### test acc
            total_acc = 0
            total_loss = 0
            batch = dataloader_eval.get_lm_batch()
            for j in range(eval_batch_num):
                input_batch, _, label_batch = next(batch)
                feed = {lm_model.x: input_batch, lm_model.y: label_batch}
                loss, acc = sess.run([lm_model.mean_loss, lm_model.acc], feed_dict=feed)
                total_loss += cost
                total_acc += acc
            acc = total_acc / eval_batch_num
            loss = total_loss / eval_batch_num
            print("epoch: %d test acc:%.4f  test loss=%.6f" % (epoch+1, acc, loss))
            if acc > old_acc:
                saver.save(sess, os.path.join(Const.LmModelFolder, 'final_model_%d.ckpt' % (epoch + 1)))
                old_acc = acc
        writer.close()


def main():
    print('//-----------------------start acoustic model-----------------------//')
    params = AmLmHparams().args
    data_args = DataHparams.args

    # train_acoustic_model(data_args, params)

    print('//-----------------------start language model-----------------------//')
    train_language_model(data_args, params)


if __name__ == '__main__':
    main()