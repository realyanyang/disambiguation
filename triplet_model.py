#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   triplet_model.py
@Time    :   2019/11/27 14:38:41
@Author  :   Yan Yang
@Contact :   yanyangbupt@gmail.com
@Desc    :   None
'''
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import load_pickle, TextToVec
import numpy as np


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, 3)
        # self.maxpool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(1, 1, 2)
        # self.maxpool2 = nn.MaxPool1d(2)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(297, 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        # x = self.maxpool2(x)
        x = self.drop(x)
        x = self.fc(x)
        return torch.squeeze(x)


class TripletModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddingnet = EmbeddingNet()

    def forward(self, anchor, posi, neg):
        anchor_emb = self.embeddingnet(anchor)
        posi_emb = self.embeddingnet(posi)
        neg_emb = self.embeddingnet(neg)
        return anchor_emb, posi_emb, neg_emb

    def get_emb(self, x):
        return self.embeddingnet(x)


class ReadData(Dataset):
    def __init__(self, posi_pair_path, neg_pair_path, whole_profile_pub, aid2cate, cate='title'):
        super().__init__()
        self.posi_pair = load_pickle(posi_pair_path)
        self.neg_pair = load_pickle(neg_pair_path)
        self.whole_profile_pub = whole_profile_pub
        self.posi_pid2aid = {}
        for pair in self.posi_pair:
            self.posi_pid2aid[pair[1]] = pair[0]
        self.neg_pid2aid = {}
        for pair in self.neg_pair:
            self.neg_pid2aid[pair[1]] = pair[0]
        self.cate = cate
        self.texttovec = TextToVec()
        self.aid2cate = aid2cate

        posi_pids = set(self.posi_pid2aid.keys())
        neg_pids = set(self.neg_pid2aid.keys())
        self.innter_pid_set = list(posi_pids & neg_pids)

    def __len__(self):
        return len(self.innter_pid_set)

    def __getitem__(self, index):
        pid_with_index = self.innter_pid_set[index]
        pid, _ = pid_with_index.split('-')
        info = self.whole_profile_pub[pid].get(self.cate)
        if info is None:
            anchor_data = np.zeros(300)
        else:
            anchor_data = self.texttovec.get_vec(info)
        posi_data = self.aid2cate[self.posi_pid2aid[pid_with_index]]
        neg_data = self.aid2cate[self.neg_pid2aid[pid_with_index]]

        anchor_data = torch.from_numpy(np.expand_dims(anchor_data, axis=0)).to(torch.float)
        posi_data = torch.from_numpy(np.expand_dims(posi_data, axis=0)).to(torch.float)
        neg_data = torch.from_numpy(np.expand_dims(neg_data, axis=0)).to(torch.float)
        return anchor_data, posi_data, neg_data
