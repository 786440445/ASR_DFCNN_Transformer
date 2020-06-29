import pyaudio
import tensorflow as tf
from util.wav_util import *
from lm_and_am.model.cnn_ctc import CNNCTCModel
from lm_and_am.test import pred_pinyin
from lm_and_am.hparams import AmHparams, LmHparams, TransformerHparams
from lm_and_am.model.language_model import Language_Model
from lm_and_am.model import Transformer
from util.data_util import language_vocab, hanzi_vocab, han2id, GetEditDistance
from lm_and_am.const import Const
from lm_and_am.data_loader import prepare_data


def receive_wav(file):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 16

    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

    print("* recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")

    stream.stop_stream()
    stream.close()
    pa.terminate()
    wf = wave.open(file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def dfcnn_speech(sess, am_model, lm_model, file):
    fbank = compute_fbank_from_file(file, feature_dim=200)
    inputs = fbank.reshape(fbank.shape[0], fbank.shape[1], 1)
    input_length = inputs.shape[0] // 8 + 1
    pred, pinyin = pred_pinyin(am_model, inputs, input_length)
    # 语言模型预测
    with sess.as_default():
        pred = np.array(pred)
        han_in = pred.reshape(1, -1)
        han_vec = sess.run(lm_model.preds, {lm_model.x: han_in})
        han_pred = ''.join(language_vocab[idx] for idx in han_vec[0])
        return pinyin, han_pred


def transformer_speech(sess, model, train_data, file):
    with sess.as_default():
        X, Y = train_data.get_transformer_data_from_file(file)
        preds = sess.run(model.preds, feed_dict={model.x: X, model.y: Y})
        han_pred =  ''.join(hanzi_vocab[idx] for idx in preds[0][:-1])
        print('中文预测结果：', han_pred)


def recognition(type='dfcnn'):
    S_index = 0
    path0 = 'aidatatang_200zh/corpus/test/G1260/T0055G1260S0020.wav'
    paht1 = 'aidatatang_200zh/corpus/test/G2863/T0055G2863S0015.wav'
    path2 = 'aidatatang_200zh/corpus/test/G2863/T0055G2863S0245.wav'
    path3 = 'aidatatang_200zh/corpus/test/G2863/T0055G2863S0247.wav'

    path4 = 'aidatatang_200zh/corpus/test/G2914/T0055G2914S0394.wav'
    path5 = 'aidatatang_200zh/corpus/test/G2914/T0055G2914S0086.wav'
    path6 = 'aidatatang_200zh/corpus/test/G2914/T0055G2914S0138.wav'
    path7 = 'aidatatang_200zh/corpus/test/G2914/T0055G2914S0363.wav'
    path = [path0, paht1, path2, path3, path4, path5, path6, path7]
    file = '../wav_file/input'
    S_hanzi0 = '你看过天天向上吗'
    S_hanzi1 = '你在我手机里'
    S_hanzi2 = '说句英文给我听'
    S_hanzi3 = '打电话给我干嘛'
    S_hanzi4 = '今天到哪里去吃饭'
    S_hanzi5 = '那你告诉我你会什么呢'
    S_hanzi6 = '能不能给我唱首歌'
    S_hanzi7 = '你能把我怎么样'
    hanzi = [S_hanzi0, S_hanzi1, S_hanzi2, S_hanzi3, S_hanzi4, S_hanzi5, S_hanzi6, S_hanzi7]


    # 现场输入识别
    if type == 'dfcnn':
        # 1.声学模型-----------------------------------
        hparams = AmHparams()
        parser = hparams.parser
        hp = parser.parse_args()
        am_model = CNNCTCModel(hp)
        print('loading acoustic model...')
        select_model_step = 'model_04-14.91'
        am_model.load_model(select_model_step)

        # 2.语言模型-----------------------------------
        hparams = LmHparams()
        parser = hparams.parser
        hp = parser.parse_args()
        hp.is_training = False
        print('loading language model...')
        lm_model = Language_Model(hp)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        sess = tf.Session(graph=lm_model.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        with lm_model.graph.as_default():
            saver = tf.train.Saver()
        with sess.as_default():
            latest = tf.train.latest_checkpoint(Const.LmModelFolder)
            saver.restore(sess, latest)
        words_num1 = 0
        word_error_num1 = 0
        words_num2 = 0
        word_error_num2 = 0

        for i in range(8):
            index = i + S_index
            filename = file + str(index) + '.wav'
            # receive_wav(filename)
            pinyin1, hanzi1 = dfcnn_speech(sess, am_model, lm_model, filename)
            print('人说话拼音预测结果：', pinyin1)
            print('人说话汉字预测结果：', hanzi1)
            pinyin2, hanzi2 = dfcnn_speech(sess, am_model, lm_model, Const.SpeechDataPath + path[index])
            print('数据集拼音预测结果：', pinyin2)
            print('数据集汉字预测结果：', hanzi2)
            print('标准汉字结果：', hanzi[i])
            print()
            hanzi_standard = han2id(hanzi[i], hanzi_vocab)
            words_n = len(hanzi_standard)
            words_num1 += words_n  # 把句子的总字数加上
            py_edit_distance = GetEditDistance(hanzi_standard, han2id(hanzi1, hanzi_vocab))
            # 拼音距离
            if (py_edit_distance <= words_n):
                word_error_num1 += py_edit_distance
            else:
                word_error_num1 += words_n

            words_num2 += words_n  # 把句子的总字数加上
            py_edit_distance = GetEditDistance(hanzi_standard, han2id(hanzi2, hanzi_vocab))
            # 拼音距离
            if (py_edit_distance <= words_n):
                word_error_num2 += py_edit_distance
            else:
                word_error_num2 += words_n
        print('人说话字错误率:', word_error_num1 / words_num1)
        print('数据集字错误率:', word_error_num2 / words_num2)

    if type == 'transformer':
        hparams = TransformerHparams()
        parser = hparams.parser
        hp = parser.parse_args()
        hp.is_training = False
        train_data = prepare_data('train', hp, shuffle=True, length=None)

        model = Transformer(hp)
        with model.graph.as_default():
            saver = tf.train.Saver()
        with tf.Session(graph=model.graph) as sess:
            latest = tf.train.latest_checkpoint(Const.TransformerFolder)
            saver.restore(sess, latest)
        while(True):
            receive_wav(file)
            transformer_speech(sess, model, train_data, file)


if __name__ == '__main__':
    recognition('dfcnn')
