import shutil, os, sys
import random
home_dir = os.getcwd()
sys.path.append(home_dir)

from util.data_util import DataUtil
from util.noise import add_noise
from util.const import Const
from util.hparams import AmLmHparams, AmDataHparams


def delete_files(pathDir):
    fileList = list(os.listdir(pathDir))
    for file in fileList:
        file = os.path.join(pathDir, file)
        if os.path.isfile(file):
            os.remove(file)
        else:
            shutil.rmtree(file)
    print("delete noise data successfully")


def main():
    data_args = AmDataHparams().args

    rate = 1
    out_path = Const.NoiseOutPath
    delete_files(out_path)
    train_data = DataUtil(data_args, batch_size=8, mode='train', data_length=None, shuffle=False)
    pathlist = train_data.path_lst
    pylist = train_data.pny_lst
    hzlist = train_data.han_lst
    length = len(pathlist)
    rand_list = random.sample(range(length), int(rate * length))

    pre_list = []
    for i in rand_list:
        path = pathlist[i]
        pre_list.append(os.path.join(Const.SpeechDataPath, path))
    _, filename_list = add_noise(pre_list, out_path=Const.NoiseOutPath, keep_bits=False)

    data = ''
    with open(os.path.join(home_dir, Const.NoiseDataTxT), 'w', encoding='utf-8') as f:
        for i in range(len(rand_list)):
            pinyin = pylist[rand_list[i]]
            hanzi = hzlist[rand_list[i]]
            data += filename_list[i] + '\t' + pinyin + '\t' + hanzi + '\n'
        f.writelines(data[:-1])
    print('---------------噪声数据生成完毕------------')


if __name__ == '__main__':
    main()