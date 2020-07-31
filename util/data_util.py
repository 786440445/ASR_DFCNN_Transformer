import os
home_dir = os.getcwd()
import pandas as pd
from collections import Counter

class DataUtil():
    def __init__(self, data_args, batch_size, mode='train', data_length=None, shuffle=False):
        self.batch_size = batch_size
        self.mode = mode
        self.data_length = data_length
        self.shuffle = shuffle

        self.thchs30 = data_args.thchs30
        self.aishell = data_args.aishell
        self.stcmd = data_args.stcmd
        self.aidatatang = data_args.aidatatang
        self.aidatatang_1505 = data_args.aidatatang_1505
        self.prime = data_args.prime
        self.noise = data_args.noise

        self.path_lst = []
        self.pny_lst = []
        self.han_lst = []

        self.source_init()

    def source_init(self):
        """
        txt文件初始化，加载
        :return:
        """
        print('get source list...')
        read_files = []
        if self.mode == 'train':
            if self.thchs30 == True:
                read_files.append('thchs_train.txt')
            if self.aishell == True:
                read_files.append('aishell_train.txt')
            if self.stcmd == True:
                read_files.append('stcmd_train.txt')
            if self.aidatatang == True:
                read_files.append('aidatatang_train.txt')
            if self.aidatatang_1505 == True:
                read_files.append('aidatatang_1505_train.txt')
            if self.prime == True:
                read_files.append('prime.txt')
            if self.noise == True:
                read_files.append('noise_data.txt')

        elif self.mode == 'dev':
            if self.thchs30 == True:
                read_files.append('thchs_dev.txt')
            if self.aishell == True:
                read_files.append('aishell_dev.txt')
            if self.stcmd == True:
                read_files.append('stcmd_dev.txt')
            if self.aidatatang == True:
                read_files.append('aidatatang_dev.txt')
            if self.aidatatang_1505 == True:
                read_files.append('aidatatang_1505_dev.txt')

        elif self.mode == 'test':
            if self.thchs30 == True:
                read_files.append('thchs_test.txt')
            if self.aishell == True:
                read_files.append('aishell_test.txt')
            if self.stcmd == True:
                read_files.append('stcmd_test.txt')
            if self.aidatatang == True:
                read_files.append('aidatatang_test.txt')
            if self.aidatatang_1505 == True:
                read_files.append('aidatatang_1505_test.txt')

        for file in read_files:
            print('load ', file, ' data...')
            sub_file = os.path.join(home_dir, 'data', file)
            data = pd.read_table(sub_file, header=None)
            paths = data.iloc[:, 0].tolist()
            pny = data.iloc[:, 1].tolist()
            hanzi = data.iloc[:, 2].tolist()
            self.path_lst.extend(paths)
            self.pny_lst.extend(pny)
            self.han_lst.extend(hanzi)

        if self.data_length:
            self.path_lst = self.path_lst[:self.data_length]
            self.pny_lst = self.pny_lst[:self.data_length]
            self.han_lst = self.han_lst[:self.data_length]
        # 保留batch size的倍数
        stay_index = len(self.path_lst) // self.batch_size * self.batch_size
        self.path_lst = self.path_lst[:stay_index]
        self.pny_lst = self.pny_lst[:stay_index]
        self.han_lst = self.han_lst[:stay_index]

    def generate_dict(self):
        han_counter = Counter(*self.han_lst)
        print(han_counter.items()[:10])