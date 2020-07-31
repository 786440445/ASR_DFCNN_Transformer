import difflib
import numpy as np
from keras import backend as K
import tensorflow as tf


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else:
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


def downsample(feature, contact):
    add_len = (contact - feature.shape[0] % contact) % contact
    pad_zero = np.zeros((add_len, feature.shape[1]), dtype=np.float)
    feature = np.append(feature, pad_zero, axis=0)
    feature = np.reshape(feature, (feature.shape[0] / 4, feature.shape[1] * 4))
    return feature


# word error rate------------------------------------
def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost


# 定义解码器------------------------------------
def decode_ctc(num_result, input_length):
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype=np.int32)
    in_len[0] = input_length
    r = K.ctc_decode(result, in_len, greedy=True, beam_width=100, top_paths=1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    r1 = r[0][0].eval(session=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    tf.reset_default_graph()  # 然后重置tf图，这句很关键
    r1 = r1[0]
    return r1