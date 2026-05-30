import os
import numpy as np
import pandas as pd
import random
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')


# abilene
class Dataset_net_abilene(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='abilene_tm.csv', label_path='abilene_label.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=100, sample_num=1000, seasonal_patterns=None,
                 use_full_data=False):

        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.label_path = label_path
        self.use_full_data = use_full_data

        self.data_scale = 0
        self.tot_len = 0
        self.sample_num = sample_num
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), header=None)
        if df_raw.iloc[:, -1].isnull().all():
            df_raw = df_raw.iloc[:, :-1]
        df_raw = df_raw.iloc[:5100]
        deal_x = df_raw.values

        label_file = os.path.join(self.root_path, self.label_path)
        if os.path.exists(label_file):
            df_label = pd.read_csv(label_file, header=None)
            if df_label.iloc[:, -1].isnull().all():
                df_label = df_label.iloc[:, :-1]
            df_label = df_label.iloc[:5100]
            deal_y = df_label.values

        all_data = 5100
        num_train = int(all_data * 0.8)
        num_test = int(all_data * 0.1)
        num_vali = int(all_data - num_train - num_test)

        if self.set_type == 0:
            self.data_scale = 1
            # tot_len is used to determine the position of the first data point when randomly selecting samples.
            self.tot_len = num_train - self.seq_len
        elif self.set_type == 1:
            self.data_scale = 1 / 8
            self.tot_len = num_vali - self.seq_len
        else:
            self.data_scale = 1 / 8
            self.tot_len = num_test - self.seq_len

        # Divide in an 8:1:1 ratio
        self.sample_num = int(self.sample_num * self.data_scale)

        border1s = [0, num_train - self.seq_len, all_data - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, all_data]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            self.tot_len = border2 - border1 - self.seq_len

        # standardize
        if self.scale:
            deal_x = deal_x / 1e9

        self.data_x = deal_x[border1:border2]
        self.data_y = deal_y[border1:border2]

    def __getitem__(self, index):
        if self.use_full_data:
            s_begin = min(index, self.tot_len)
        else:
            s_begin = random.randint(0, self.tot_len)

        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[s_begin:s_end, :]

        return seq_x, seq_y

    def __len__(self):
        if self.use_full_data:
            return self.tot_len
        else:
            return self.sample_num

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# geant
class Dataset_net_geant(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='geant_tm.csv', label_path='test_label.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=100, sample_num=1000, seasonal_patterns=None,
                 use_full_data=False):

        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.label_path = label_path
        self.use_full_data = use_full_data

        self.data_scale = 0
        self.tot_len = 0
        self.sample_num = sample_num
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), header=None)
        if df_raw.iloc[:, -1].isnull().all():
            df_raw = df_raw.iloc[:, :-1]

        cols = list(df_raw.columns)
        if self.target in cols and 'data' in cols:
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
        else:
            df_data = df_raw

        label_file = os.path.join(self.root_path, self.label_path)
        if os.path.exists(label_file):
            df_label = pd.read_csv(label_file, header=None)
            if df_label.iloc[:, -1].isnull().all():
                df_label = df_label.iloc[:, :-1]
            if self.features == 'S' and self.target in cols:
                deal_y = df_label[[self.target]].values
            else:
                deal_y = df_label.values

        if self.scale:
            df = df_data / 1e7

        data = df.values

        # all_data = 3000
        all_data = len(df_data)
        num_train = int(all_data * 0.8)
        num_test = int(all_data * 0.1)
        num_vali = int(all_data - num_train - num_test)

        if self.set_type == 0:
            self.data_scale = 1
            # tot_len is used to determine the position of the first data point when randomly selecting samples.
            self.tot_len = num_train - self.seq_len
        elif self.set_type == 1:
            self.data_scale = 1 / 8
            self.tot_len = num_vali - self.seq_len
        else:
            self.data_scale = 1 / 8
            self.tot_len = num_test - self.seq_len

        # Divide in an 8:1:1 ratio
        self.sample_num = int(self.sample_num * self.data_scale)

        border1s = [0, num_train - self.seq_len, all_data - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, all_data]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            self.tot_len = border2 - border1 - self.seq_len

        self.data_x = data[border1:border2]
        self.data_y = deal_y[border1:border2]

    def __getitem__(self, index):
        if self.use_full_data:
            s_begin = min(index, self.tot_len)
        else:
            s_begin = random.randint(0, self.tot_len)

        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[s_begin:s_end, :]

        return seq_x, seq_y

    def __len__(self):
        if self.use_full_data:
            return self.tot_len
        else:
            return self.sample_num

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_net_wsdream(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='wsdream_tm.csv', label_path='test_label.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=100, sample_num=1000, seasonal_patterns=None,
                 use_full_data=False):

        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.label_path = label_path
        self.use_full_data = use_full_data  # <--- 保存参数状态

        self.data_scale = 0
        self.tot_len = 0
        self.sample_num = sample_num
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), header=None)
        if df_raw.iloc[:, -1].isnull().all():
            df_raw = df_raw.iloc[:, :-1]
        df_raw = df_raw.iloc[:18000]
        deal_x = df_raw.values

        label_file = os.path.join(self.root_path, self.label_path)
        if os.path.exists(label_file):
            df_label = pd.read_csv(label_file, header=None)
            if df_label.iloc[:, -1].isnull().all():
                df_label = df_label.iloc[:, :-1]
            df_label = df_label.iloc[:18000]
            deal_y = df_label.values

        all_data = 18000
        num_train = int(all_data * 0.8)
        num_test = int(all_data * 0.1)
        num_vali = int(all_data - num_train - num_test)

        if self.set_type == 0:
            self.data_scale = 1
            # tot_len is used to determine the position of the first data point when randomly selecting samples.
            self.tot_len = num_train - self.seq_len
        elif self.set_type == 1:
            self.data_scale = 1 / 8
            self.tot_len = num_vali - self.seq_len
        else:
            self.data_scale = 1 / 8
            self.tot_len = num_test - self.seq_len

        # Divide in an 8:1:1 ratio
        self.sample_num = int(self.sample_num * self.data_scale)

        border1s = [0, num_train - self.seq_len, all_data - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, all_data]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            self.tot_len = border2 - border1 - self.seq_len

        # standardize
        if self.scale:
            deal_x = deal_x / 1e1

        self.data_x = deal_x[border1:border2]
        self.data_y = deal_y[border1:border2]

    def __getitem__(self, index):
        if self.use_full_data:
            s_begin = min(index, self.tot_len)
        else:
            s_begin = random.randint(0, self.tot_len)

        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[s_begin:s_end, :]

        return seq_x, seq_y

    def __len__(self):
        if self.use_full_data:
            return self.tot_len
        else:
            return self.sample_num

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)