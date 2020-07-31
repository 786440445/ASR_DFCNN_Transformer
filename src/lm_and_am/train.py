import tensorflow as tf
import keras
import os
import warnings
import sys
from keras.callbacks import ModelCheckpoint
home_dir = os.getcwd()
sys.path.append(home_dir)
cur_path = os.path.dirname(__file__)

from src.lm_and_am.model.cnn_ctc import CNNCTCModel
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
    data_util_train = DataUtil(data_args, batch_size=am_hp.batch_size, mode='train', data_length=None, shuffle=True)
    data_util_dev = DataUtil(data_args, batch_size=am_hp.batch_size, mode='dev', data_length=None, shuffle=True)

    dataloader = DataLoader(data_util_train, data_args, am_hp)
    dev_generator = DataLoader(data_util_dev, data_args, am_hp)

    model = CNNCTCModel(am_hp, dataloader.acoustic_vocab_size)

    save_step = len(data_util_train.path_lst) // am_hp.batch_size
    latest = tf.train.latest_checkpoint(Const.AmModelFolder)
    select_model = 'model_22-7.31'
    if os.path.exists(os.path.join(home_dir, Const.AmModelFolder, select_model + '.hdf5')):
        print('load acoustic model...')
        model.load_model(select_model)

    ckpt = "model_{epoch:02d}-{val_loss:.2f}.hdf5"
    cpCallBack = ModelCheckpoint(os.path.join(Const.AmModelFolder, ckpt), verbose=1, save_best_only=True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=Const.AmModelTensorBoard, histogram_freq=0, write_graph=True,
                                             write_images=True, update_freq='epoch')
    model.ctc_model.fit_generator(dataloader,
                                  steps_per_epoch=save_step,
                                  validation_data=dev_generator,
                                  validation_steps=20,
                                  epochs=epochs,
                                  workers=2,
                                  use_multiprocessing=False,
                                  callbacks=[cpCallBack,
                                             tbCallBack])
    pass


def train_language_model(data_args, am_hp):
    """
    语言模型
    :param train_data: 训练数据
    :return:
    """
    epochs = am_hp.epochs
    batch_size = am_hp.batch_size

    data_util_train = DataUtil(data_args, batch_size=batch_size, mode='train', data_length=None, shuffle=True)
    dataloader = DataLoader(data_util_train, data_args, am_hp)
    lm_model = Language_Model(am_hp, dataloader.acoustic_vocab_size, dataloader.language_vocab_size)
    batch_num = len(data_util_train.path_lst) // batch_size

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
                add_num = int(latest.split('_')[-2])
                saver.restore(sess, latest)
        writer = tf.summary.FileWriter(Const.LmModelTensorboard, tf.get_default_graph())
        for k in range(epochs):
            total_loss = 0
            batch = dataloader.get_lm_batch()
            for i in range(batch_num):
                input_batch, label_batch = next(batch)
                print(input_batch[0])
                print(label_batch[0])
                feed = {lm_model.x: input_batch, lm_model.y: label_batch}
                cost, _ = sess.run([lm_model.mean_loss, lm_model.train_op], feed_dict=feed)
                total_loss += cost
                if i % 10 == 0:
                    print("epoch: %d step: %d/%d  train loss=6%f" % (k+1, i, batch_num, cost))
            rs = sess.run(merged, feed_dict=feed)
            writer.add_summary(rs, k * batch_num + i)
            print('epochs', k + 1, ': average loss = ', total_loss / batch_num)
            saver.save(sess, Const.LmModelFolder + 'model_%d_%.3f.ckpt' % (k + 1 + add_num, total_loss / batch_num))
        writer.close()
    pass


def main():
    print('//-----------------------start acoustic model-----------------------//')
    params = AmLmHparams().args
    data_args = DataHparams.args

    # train_acoustic_model(data_args, params)

    print('//-----------------------start language model-----------------------//')
    train_language_model(data_args, params)


if __name__ == '__main__':
    main()