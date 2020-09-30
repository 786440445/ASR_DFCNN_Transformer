import tensorflow as tf
import keras
import os
import warnings
import sys
home_dir = os.getcwd()
sys.path.append(home_dir)
cur_path = os.path.dirname(__file__)
import numpy as np
from src.lm_and_am.model.ctc_attention import CNNCTCModel
from src.lm_and_am.model.language_model import Language_Model
from src.lm_and_am.data_loader2 import DataLoader
from util.hparams import AmLmHparams, AmDataHparams, LmDataHparams
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
    batch_size = am_hp.am_batch_size
    data_util_train = DataUtil(data_args, batch_size=batch_size, mode='train', data_length=None, shuffle=True)
    data_util_dev = DataUtil(data_args, batch_size=batch_size, mode='dev', data_length=None, shuffle=True)

    train_dataloader = DataLoader(data_util_train, data_args, am_hp)
    dev_dataloader = DataLoader(data_util_dev, data_args, am_hp)

    with tf.Graph().as_default():
        acoustic_model = CNNCTCModel(am_hp, train_dataloader.acoustic_vocab_size, train_dataloader.language_vocab_size)
        saver = tf.train.Saver(max_to_keep=5)
        # 数据读取处理部分
        dataset = tf.data.Dataset.from_generator(train_dataloader.am_generator,
                                                 output_types=(tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32))
        dataset = dataset.map(lambda x, y, z, w, m, n: (x, y, z, w, m, n), num_parallel_calls=64).prefetch(buffer_size=10000)
        with tf.Session() as sess:
            print('Start training')
            latest = tf.train.latest_checkpoint(Const.AmModelFolder)
            if latest != None:
                print('load acoustic model...')
                saver.restore(sess, latest)
            else:
                sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(Const.AmModelTensorboard, tf.get_default_graph())
            old_wer = 1
            batch_nums = len(train_dataloader)
            for epoch in range(epochs):
                total_loss = 0
                iterator_train = dataset.make_one_shot_iterator().get_next()
                for train_step in range(batch_nums):
                    input_x_batch, input_length_batch, _, _, target_y_batch, seq_length_batch = sess.run(iterator_train)
                    feed = {acoustic_model.wav_input: input_x_batch,
                            acoustic_model.wav_length: input_length_batch,
                            acoustic_model.target_hanzi: target_y_batch,
                            acoustic_model.target_hanzi_length: seq_length_batch}
                    loss, mean_loss, lr, summary, label_err, _ = sess.run([acoustic_model.loss,
                                                                     acoustic_model.mean_loss,
                                                                     acoustic_model.current_learning,
                                                                     acoustic_model.summary,
                                                                     acoustic_model.label_err,
                                                                     acoustic_model.train_op], feed_dict=feed)
                    total_loss += mean_loss
                    if (train_step + 1) % 2 == 0:
                        print('epoch: %d    step: %d/%d  mean_loss: %.4f    total_loss: %.4f  lr: %.6f   label_err: %.4f'
                              % (epoch+1, train_step+1, batch_nums, mean_loss, total_loss/(train_step+1), lr, label_err))
                        print(loss)
                writer.add_summary(summary, epoch)

                # 测试集测试
                total_err = 0
                total_loss = 0
                eval_steps = len(dev_dataloader)
                for feature_input, logits_length, _, _, target_y, target_length in dev_dataloader.am_generator():
                    feed = {acoustic_model.wav_input: feature_input,
                            acoustic_model.wav_length: logits_length,
                            acoustic_model.target_hanzi: target_y,
                            acoustic_model.target_hanzi_length: target_length}
                    mean_loss, label_err = sess.run([acoustic_model.mean_loss, acoustic_model.label_err], feed_dict=feed)
                    total_loss += mean_loss
                    total_err += label_err
                wer = total_err / eval_steps
                mean_loss = total_loss / eval_steps
                save_ckpt = 'epoch_%d_loss_%.2f_wer_%.2f.ckpt'
                saver.save(sess,  os.path.join(Const.AmModelFolder, save_ckpt % (epoch, mean_loss, wer)))
                print('epoch: ', epoch + 1, ': average loss = ', mean_loss)
                if wer < old_wer:
                    saver.save(sess, os.path.join(Const.AmModelFolder, 'final_model.ckpt'))
                    old_wer = wer
            pass


def train_language_model(data_args, am_hp):
    """
    语言模型
    :param train_data: 训练数据
    :return:
    """
    epochs = am_hp.epochs
    batch_size = am_hp.lm_batch_size
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
                # add_num = int(latest.split('_')[-1])
        writer = tf.summary.FileWriter(Const.LmModelTensorboard, tf.get_default_graph())
        old_acc = 0
        for epoch in range(epochs):
            total_loss = 0
            batch = dataloader.get_lm_batch()
            for i in range(batch_num):
                input_batch, _, label_batch = next(batch)
                feed = {lm_model.x: input_batch, lm_model.y: label_batch}
                cost, cur_lr, _ = sess.run([lm_model.mean_loss,
                                            lm_model.current_learning,
                                            lm_model.train_op], feed_dict=feed)
                total_loss += cost
                if i % 10 == 0:
                    print("epoch: %d    step: %d/%d lr:%.6f train loss=%.6f" % (epoch + 1, i, batch_num, cur_lr, cost))
            summary = sess.run(merged, feed_dict=feed)
            writer.add_summary(summary, epoch)
            print('epochs', epoch+1, ': average loss = ', total_loss / batch_num)
            saver.save(sess, Const.LmModelFolder + 'model_%d_%.3f.ckpt' % (epoch + 1, total_loss / batch_num))
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
    # print('//-----------------------start acoustic model-----------------------//')
    params = AmLmHparams().args
    am_data_args = AmDataHparams.args
    lm_data_args = LmDataHparams.args

    train_acoustic_model(am_data_args, params)
    #
    # print('//-----------------------start language model-----------------------//')
    # train_language_model(lm_data_args, params)


if __name__ == '__main__':
    main()