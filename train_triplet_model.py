#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   train_triplet_model.py
@Time    :   2019/11/27 19:51:15
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
from triplet_model import ReadData, TripletModel
from utils import load_json, load_pickle
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np


TRAIN_AUTHOR_PATH = './data2/train/train_author.json'
TRAIN_PUB_PATH = './data2/train/train_pub.json'
WHOLE_AUTHOR_PROFILE_PATH = './data2/cna_data/whole_author_profile.json'
WHOLE_AUTHOR_PROFILE_PUB_PATH = './data2/cna_data/whole_author_profile_pub.json'
VALID_PUB_PATH = './data2/cna_data/cna_valid_pub.json'
VALID_UNASS_PATH = './data2/cna_data/cna_valid_unass_competition.json'
NEW_DATA_DIR = './new-data'          # original info, for test
NEW_DATA_V2_DIR = './new-data-v2'    # last 1 year info
NEW_DATA_V3_DIR = './new-data-v3'    # last 2 year info
OUT_DIR_v2 = './out-v2'
SPLIT_DIR = './split-data'
TEST_FEATURE_DIR_V2 = './test-feature-v2'
STACK_MODEL_DIR_v2 = './stack_model_aid2abstractvecv2'
RANDOM_SEED = 1129

BATCH_SIZE = 512
LR = 0.01
EPOCHS = 20


def AccuracyDis(anchor_emb, posi_emb, neg_emb):
    pos_distance = torch.sqrt(torch.sum(torch.pow((anchor_emb - posi_emb), 2), dim=1))
    neg_distance = torch.sqrt(torch.sum(torch.pow((anchor_emb - neg_emb), 2), dim=1))
    acc = torch.mean((pos_distance < neg_distance).to(torch.float))
    return acc


if __name__ == "__main__":
    whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    train_posi_pair_path = os.path.join(NEW_DATA_V3_DIR, 'train-posi-pair-list.pkl')
    train_neg_pair_path = os.path.join(NEW_DATA_V3_DIR, 'train-neg-pair-list.pkl')
    test_posi_pair_path = os.path.join(NEW_DATA_V3_DIR, 'test-posi-pair-list.pkl')
    test_neg_pair_path = os.path.join(NEW_DATA_V3_DIR, 'test-neg-pair-list.pkl')

    # all_posi_pair_path = os.path.join(NEW_DATA_V3_DIR, 'posi-pair-list-extend1.pkl')
    # all_neg_pair_path = os.path.join(NEW_DATA_V3_DIR, 'neg-pair-list-extend1.pkl')

    aid2abstractvec = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2abstractvec.pkl'))
    aid2titlevec = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2titlevec.pkl'))

    keyarg = {
        'aid2cate': aid2titlevec,
        'cate': 'title',
        # 'aid2cate': aid2abstractvec,
        # 'cate': 'abstract'
    }
    print(keyarg['cate'])
    train_dataset = ReadData(train_posi_pair_path, train_neg_pair_path, whole_author_profile_pub, **keyarg)
    test_dataset = ReadData(test_posi_pair_path, test_neg_pair_path, whole_author_profile_pub, **keyarg)

    # all_dataset = ReadData(all_posi_pair_path, all_neg_pair_path, whole_author_profile_pub, **keyarg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    # loader = DataLoader(train_loader, batch_size=BATCH_SIZE, num_workers=2)
    triplet_model = TripletModel().to(device)
    criterion = nn.TripletMarginLoss()
    optimizer = torch.optim.Adam(triplet_model.parameters(), lr=LR)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    for epoch in range(EPOCHS):
        triplet_model.train()
        train_loss = []
        for anchor, posi, neg in train_loader:
            anchor, posi, neg = anchor.to(device), posi.to(device), neg.to(device)
            optimizer.zero_grad()
            embs = triplet_model(anchor, posi, neg)
            loss = criterion(*embs)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        triplet_model.eval()
        test_loss = []
        accuracy = []
        with torch.no_grad():
            for test_anchor, test_posi, test_neg in test_loader:
                test_anchor, test_posi, test_neg = test_anchor.to(device), test_posi.to(device), test_neg.to(device)
                test_embs = triplet_model(test_anchor, test_posi, test_neg)
                loss = criterion(*test_embs)
                acc = AccuracyDis(*test_embs)
                accuracy.append(acc.item())
                test_loss.append(loss.item())
        lr_schedule.step()
        print('Epoch: [%d/%d], train loss: %f, test loss %f, acc: %f' % (epoch + 1, EPOCHS, np.mean(train_loss), np.mean(test_loss), np.mean(accuracy)))
        # print('Epoch: [%d/%d], train loss: %f\n' % (epoch + 1, EPOCHS, np.mean(train_loss)))
    os.makedirs('./text-model', exist_ok=True)
    torch.save(triplet_model.state_dict(), './text-model/tm.%s.1.checkpoint.pth' % keyarg['cate'])
