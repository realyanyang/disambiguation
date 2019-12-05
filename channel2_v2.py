#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   channel2_v2.py
@Time    :   2019/11/15 22:39:45
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
import os
from pyjarowinkler import distance
from collections import defaultdict
import numpy as np
import tqdm
import torch
from triplet_model import TripletModel
# import random
import pandas as pd
import itertools
from utils import load_json, load_pickle, save_json, save_pickle, clean_name, get_name_index, SK_MLP
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from stack_model import StackModel
import datetime
import time
from utils import get_coauthor_v2, get_year_diff_v2, get_venue_score_v2, get_org_score_v2, get_key_word_num_v2, get_relative_year_feature
from utils import get_org_with_set_score, get_venue_with_set_score, get_keywords_with_set_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import TextToVec
from sklearn.model_selection import train_test_split
sns.set()


TRAIN_AUTHOR_PATH = './data2/train/train_author.json'
TRAIN_PUB_PATH = './data2/train/train_pub.json'
WHOLE_AUTHOR_PROFILE_PATH = './data2/cna_data/whole_author_profile.json'
WHOLE_AUTHOR_PROFILE_PUB_PATH = './data2/cna_data/whole_author_profile_pub.json'
VALID_PUB_PATH = './data2/cna_data/cna_valid_pub.json'
VALID_UNASS_PATH = './data2/cna_data/cna_valid_unass_competition.json'
NEW_DATA_DIR = './new-data'          # original info, for test
NEW_DATA_V2_DIR = './new-data-v2'    # last 1 year info
NEW_DATA_V3_DIR = './new-data-v3'    # last 2 year info
NEW_DATA_V4_DIR = './new-data-v4'    # add paper count info
OUT_DIR_v2 = './out-v2'
SPLIT_DIR = './split-data'
TEST_FEATURE_DIR_V2 = './test-feature-v2'
STACK_MODEL_DIR_v2 = './stack_model_v2'
RANDOM_SEED = 1129

# random.seed()
np.random.seed(RANDOM_SEED)


BASE_COLS = [
    'coauthors_count', 'coauthors_count_by_all_count', 'coauthors_count_by_this_coauthor_count',
    'this_paper_coauthor_count', 'this_paper_coathor_count_by_all_coauthor', 'this_paper_coauthor_count_by_this_paper_coauthor_count',
    'min_diff', 'max_diff', 'mean_diff', 'meadian_diff', 'min_max_avg_diff', 'is_in_range',
    'this_year_count', 'this_year_count_by_all_year', 'is_in_cate_range', 'before_one', 'before_two',
    'later_one', 'later_two', 'venue_max_score', 'venue_mean_score', 'venue_max_add_score',
    'venue_mean_add_score', 'venue_is_match', 'venue_score_add_score', 'org_max_score', 'org_mean_score',
    'org_max_add_score', 'org_mean_add_score', 'org_is_match', 'org_score_add_score', 'org_year_abs_diff',
    'keywords_max_score', 'keywords_mean_score', 'rela_year_diff', 'rela_coauthor_count',
    'rela_coauthor_count_by1', 'rela_coauthor_count_by2', 'rela_org_max_score',
    'rela_org_mean_score', 'rela_org_max_add_score', 'rela_org_mean_add_score',
    'rela_org_score_add_score', 'rela_venue_max_score', 'rela_venue_mean_score',
    'rela_venue_max_add_score', 'rela_venue_mean_add_score', 'rela_venue_score_add_score',
    'rela_keyword_max_score', 'rela_keyword_mean_score'
]
# length: 50 !
SET_INFO_COLS = [
    'org_set_count', 'org_set_count_by_all_count',
    'org_set_count_by_this_count', 'venue_word_count', 'venue_word_count_by_all_count',
    'venue_word_count_by_this_venue_count', 'keyword_count', 'keyword_count_by_all_count',
    'keyword_count_by_this_keyword_count'
]
# length: 9 !
TITLE_COLS = [
    'title'
]


def split_data(last_n_year):
    """
    split original data to train and test for paper-author wise feature.
    select the lastest paper as the test for every author.
    """
    def get_last_n_year_paper(n, paper_ids, whole_author_profile_pub):
        years = []
        for pid in paper_ids:
            year = whole_author_profile_pub[pid].get('year', '0')
            if year == '':
                year = 0
            else:
                year = int(year)
            if year < 1500 or year > 2100:
                year = 0
            years.append(year)
        # big to small
        years_sort_index = np.argsort(years)[::-1]
        target_index = years_sort_index[:n]
        paper_ids_array = np.array(paper_ids)
        return paper_ids_array[target_index]

    # assert last_n_year >= 2
    whole_author_profile = load_json(WHOLE_AUTHOR_PROFILE_PATH)
    whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    test_profile = {}
    for aid in tqdm.tqdm(whole_author_profile):
        inner_dict = {}
        papers = whole_author_profile[aid]['papers']
        inner_dict['name'] = whole_author_profile[aid]['name']
        if len(papers) <= 1:
            inner_dict['papers'] = []
        elif len(papers) <= last_n_year:
            inner_dict['papers'] = get_last_n_year_paper(1, papers, whole_author_profile_pub).tolist()
        else:
            inner_dict['papers'] = get_last_n_year_paper(last_n_year, papers, whole_author_profile_pub).tolist()
        test_profile[aid] = inner_dict
        for pid in inner_dict['papers']:
            whole_author_profile[aid]['papers'].remove(pid)
        # years = []
        # for pid in papers:
        #     year = whole_author_profile_pub[pid].get('year', '0')
        #     if year == '':
        #         year = 0
        #     else:
        #         year = int(year)
        #     if year < 1500 or year > 2100:
        #         year = 0
        #     years.append(year)
        # last_year_index = np.argmax(years)
        # last_year_paper = papers[last_year_index]
        # whole_author_profile[aid]['papers'].remove(last_year_paper)
        # inner_dict['name'] = whole_author_profile[aid]['name']
        # inner_dict['papers'] = [last_year_paper]
        # test_profile[aid] = inner_dict
    os.makedirs(SPLIT_DIR, exist_ok=True)
    train_profile_path = os.path.join(SPLIT_DIR, 'train_profile-last%dyear.json' % last_n_year)
    test_profile_path = os.path.join(SPLIT_DIR, 'test_profile-last%dyear.json' % last_n_year)
    save_json(whole_author_profile, train_profile_path)
    save_json(test_profile, test_profile_path)


def preprocessing(mission='train'):
    # os.makedirs(NEW_DATA_DIR, exist_ok=True)

    # ------------------------------------------
    # process whole_author_profile.json, add index, and save to pickle
    # save format: name2aids --> {name: [aids, ...]}, aid2pids --> {aid: [pid-index, ...]}
    os.makedirs(NEW_DATA_DIR, exist_ok=True)
    os.makedirs(NEW_DATA_V2_DIR, exist_ok=True)
    if mission == 'train':
        whole_author_profile = load_json(os.path.join(SPLIT_DIR, 'train_profile-last1year.json'))
    elif mission == 'test':
        whole_author_profile = load_json(WHOLE_AUTHOR_PROFILE_PATH)
    else:
        raise ValueError("check mission value")
    whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    name2aids = {}
    aid2pids = {}
    aids = []
    names = []
    pids_with_index = []
    for aid in tqdm.tqdm(whole_author_profile):
        aids.append(aid)
        names.append(whole_author_profile[aid]['name'])
        pids = whole_author_profile[aid]['papers']
        tmp = []
        for paper in pids:
            paper_authors = whole_author_profile_pub[paper]['authors']
            author_names = [clean_name(item['name']) for item in paper_authors]
            # print(author_names)
            index = get_name_index(names[-1], author_names)
            tmp.append('%s-%d' % (paper, index))
        pids_with_index.append(tmp)
    assert len(aids) == len(names)
    assert len(names) == len(pids_with_index)
    print('all aids num: ', len(aids))
    name_set = set(names)
    names_array = np.array(names)
    aids_array = np.array(aids)
    for name in name_set:
        target_aid = aids_array[names_array == name]
        name2aids[name] = target_aid
    for aid, pid in zip(aids, pids_with_index):
        aid2pids[aid] = pid
    if mission == 'train':
        save_pickle(name2aids, os.path.join(NEW_DATA_V2_DIR, 'name2aids.pkl'))
        save_pickle(aid2pids, os.path.join(NEW_DATA_V2_DIR, 'aid2pids.pkl'))
    elif mission == 'test':
        save_pickle(name2aids, os.path.join(NEW_DATA_DIR, 'name2aids.pkl'))
        save_pickle(aid2pids, os.path.join(NEW_DATA_DIR, 'aid2pids.pkl'))

    # ------------------------------------------
    # save format: aid2year --> {aid: {min: xxx, max: xxx, mean: xxx, median: xxx, min_max_avg: xxx, year_list: [year, ...]}}
    if mission == 'train':
        aid2pids = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2pids.pkl'))
    elif mission == 'test':
        aid2pids = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2pids.pkl'))
    aid2year = {}
    print('Process year info ...')
    for aid in tqdm.tqdm(aid2pids.keys()):
        pids = aid2pids[aid]
        all_years = []
        for pid_with_index in pids:
            pid = pid_with_index.split('-')[0]
            year = whole_author_profile_pub[pid].get('year', '0')
            if year == '':
                year = 0
            else:
                year = int(year)
            if any([year < 1500, year > 2100]):
                year = 0
            all_years.append(year)
        all_years = np.array(all_years)
        all_years = all_years[all_years != 0]
        if len(all_years) == 0:
            year_info = None
        else:
            year_info = {
                'min': np.min(all_years),
                'max': np.max(all_years),
                'mean': np.mean(all_years),
                'min_max_avg': (np.min(all_years) + np.max(all_years)) / 2,
                'median': np.median(all_years),
                'year_list': all_years,
            }
        aid2year[aid] = year_info
    if mission == 'train':
        save_pickle(aid2year, os.path.join(NEW_DATA_V2_DIR, 'aid2year.pkl'))
    elif mission == 'test':
        save_pickle(aid2year, os.path.join(NEW_DATA_DIR, 'aid2year.pkl'))

    # ------------------------------------------
    # save format: aid2coauthor --> {aid: {anuthor-name: count, ...}}
    aid2coauthor = {}
    print('aid2coauthor processing ...')
    for aid in tqdm.tqdm(aid2pids.keys()):
        pids = aid2pids[aid]
        inner_dict = defaultdict(int)
        for pid_with_index in pids:
            pid, index = pid_with_index.split('-')
            authors = whole_author_profile_pub[pid]['authors']
            authors_name = [clean_name(item['name']) for item in authors]
            authors_name.pop(int(index))
            for name in authors_name:
                inner_dict[name] += 1
        aid2coauthor[aid] = inner_dict
    if mission == 'train':
        save_pickle(aid2coauthor, os.path.join(NEW_DATA_V2_DIR, 'aid2coauthor.pkl'))
    elif mission == 'test':
        save_pickle(aid2coauthor, os.path.join(NEW_DATA_DIR, 'aid2coauthor.pkl'))

    # ------------------------------------------
    # save format: aid2venue --> {aid: {venue-name: count ...}}
    aid2venue = {}
    print('aid2venue processing ...')
    for aid in tqdm.tqdm(aid2pids.keys()):
        pids = aid2pids[aid]
        inner_dict = defaultdict(int)
        for pid_with_index in pids:
            pid, index = pid_with_index.split('-')
            venue = whole_author_profile_pub[pid]['venue'].lower()
            if venue != '':
                # aid2venue[aid].add(venue)
                inner_dict[venue] += 1
        aid2venue[aid] = inner_dict
    if mission == 'train':
        save_pickle(aid2venue, os.path.join(NEW_DATA_V2_DIR, 'aid2venue.pkl'))
    elif mission == 'test':
        save_pickle(aid2venue, os.path.join(NEW_DATA_DIR, 'aid2venue.pkl'))

    # ------------------------------------------
    # save format: aid2keywords --> {aid: {keyword: count, ...}}
    aid2keywords = {}
    print('aid2keywords processing ...')
    for aid in tqdm.tqdm(aid2pids.keys()):
        pids = aid2pids[aid]
        inner_dict = defaultdict(int)
        for pid_with_index in pids:
            pid, index = pid_with_index.split('-')
            keywords = whole_author_profile_pub[pid].get('keywords', '')
            if len(keywords) == 0:
                continue
            for keyword in keywords:
                if keyword != '':
                    # aid2keywords[aid].add(keyword.lower())
                    inner_dict[keyword] += 1
        aid2keywords[aid] = inner_dict
    if mission == 'train':
        save_pickle(aid2keywords, os.path.join(NEW_DATA_V2_DIR, 'aid2keywords.pkl'))
    elif mission == 'test':
        save_pickle(aid2keywords, os.path.join(NEW_DATA_DIR, 'aid2keywords.pkl'))

    # ------------------------------------------
    # save format: aid2orgset--> {aid: set{org_word, org_word, ...}}
    aid2orgset = {}
    for aid in tqdm.tqdm(aid2pids.keys()):
        pids = aid2pids[aid]
        inner_set = set()
        for pid_with_index in pids:
            pid, index = pid_with_index.split('-')
            author = whole_author_profile_pub[pid].get('authors')[int(index)]
            org = author.get('org', '').lower().strip()
            org_set = set(org.split())
            inner_set = inner_set | org_set
        aid2orgset[aid] = inner_set
    if mission == 'train':
        save_pickle(aid2orgset, os.path.join(NEW_DATA_V2_DIR, 'aid2orgset.pkl'))
    elif mission == 'test':
        save_pickle(aid2orgset, os.path.join(NEW_DATA_DIR, 'aid2orgset.pkl'))

    # ------------------------------------------
    # save format: aid2venueset--> {aid: set{venue_word, venue_word, ...}}
    aid2venueset = {}
    for aid in tqdm.tqdm(aid2pids.keys()):
        pids = aid2pids[aid]
        inner_set = set()
        for pid_with_index in pids:
            pid, index = pid_with_index.split('-')
            venue = whole_author_profile_pub[pid].get('venue', '').lower()
            if venue == '':
                continue
            else:
                venue_set = set(venue.replace('-', ' ').split())
                inner_set = inner_set | venue_set
        aid2venueset[aid] = inner_set
    if mission == 'train':
        save_pickle(aid2venueset, os.path.join(NEW_DATA_V2_DIR, 'aid2venueset.pkl'))
    elif mission == 'test':
        save_pickle(aid2venueset, os.path.join(NEW_DATA_DIR, 'aid2venueset.pkl'))

    # ------------------------------------------
    # save format: aid2keywordsset--> {aid: set{key_word, key_word, ...}}
    aid2keywordsset = {}
    for aid in tqdm.tqdm(aid2pids.keys()):
        pids = aid2pids[aid]
        inner_set = set()
        for pid_with_index in pids:
            pid, index = pid_with_index.split('-')
            keywords = whole_author_profile_pub[pid].get('keywords', '')
            if len(keywords) == 0:
                continue
            for keyword in keywords:
                if keyword != '':
                    keyword_set = set(keyword.lower().replace('-', ' ').split())
                    inner_set = inner_set | keyword_set
        aid2keywordsset[aid] = inner_set
    if mission == 'train':
        save_pickle(aid2keywordsset, os.path.join(NEW_DATA_V2_DIR, 'aid2keywordsset.pkl'))
    elif mission == 'test':
        save_pickle(aid2keywordsset, os.path.join(NEW_DATA_DIR, 'aid2keywordsset.pkl'))

    # ------------------------------------------
    # save format: aid2orgwithyear --> {aid: [(org, year), () ...]}
    aid2orgwithyear = {}
    for aid in tqdm.tqdm(aid2pids.keys()):
        pids = aid2pids[aid]
        inner_list = []
        for pid_with_index in pids:
            pid, index = pid_with_index.split('-')
            auhtors = whole_author_profile_pub[pid]['authors']
            org = auhtors[int(index)].get('org', '').lower()
            year = whole_author_profile_pub[pid].get('year', '0')
            if year == '':
                year = 0
            else:
                year = int(year)
            if any([year < 1500, year > 2100]):
                year = 0
            inner_list.append((org, year))
        aid2orgwithyear[aid] = inner_list
    if mission == 'train':
        save_pickle(aid2orgwithyear, os.path.join(NEW_DATA_V2_DIR, 'aid2orgwithyear.pkl'))
    elif mission == 'test':
        save_pickle(aid2orgwithyear, os.path.join(NEW_DATA_DIR, 'aid2orgwithyear.pkl'))

    # ------------------------------------------
    # save format aid2yearinfo --> {aid: {year: {
    #                                            orgs: [org, ....],
    #                                            venues: [venues, ...],
    #                                            keywords: [keyword, ...],
    #                                            coauthors: [author-name, ...],
    #                                            }}}
    aid2yearinfo = {}
    for aid in tqdm.tqdm(aid2pids.keys()):
        inner_dict = {}
        pids = aid2pids[aid]
        for pid_with_index in pids:
            pid, index = pid_with_index.split('-')
            year = whole_author_profile_pub[pid].get('year', '0')
            if year == '':
                year = 0
            else:
                year = int(year)
            if any([year < 1500, year > 2100]):
                year = 0
            authors = whole_author_profile_pub[pid]['authors']
            authors_name = [clean_name(item['name']) for item in authors]
            org = [authors[int(index)].get('org', '').lower()]
            authors_name.pop(int(index))
            coauthor = authors_name
            venue = [whole_author_profile_pub[pid].get('venue', '').lower()]
            keywords = whole_author_profile_pub[pid].get('keywords', [''])
            if len(keywords) == 0:
                keywords = ['']
            keywords = [keyword.lower() for keyword in keywords]
            tmp_dict = {
                'orgs': org,
                'venues': venue,
                'keywords': keywords,
                'coauthors': coauthor,
            }
            if year in inner_dict.keys():
                for key in tmp_dict:
                    inner_dict[year][key].extend(tmp_dict[key])
            else:
                inner_dict[year] = tmp_dict
        aid2yearinfo[aid] = inner_dict
    if mission == 'train':
        save_pickle(aid2yearinfo, os.path.join(NEW_DATA_V2_DIR, 'aid2yearinfo.pkl'))
    elif mission == 'test':
        save_pickle(aid2yearinfo, os.path.join(NEW_DATA_DIR, 'aid2yearinfo.pkl'))

    texttovec = TextToVec()
    # ------------------------------------------
    # save format: aid2titlevec --> {aid: [mean value]}
    aid2titlevec = {}
    for aid in tqdm.tqdm(aid2pids.keys()):
        papers = aid2pids[aid]
        inner_list = []
        for pid_with_index in papers:
            pid, index = pid_with_index.split('-')
            title = whole_author_profile_pub[pid]['title']
            inner_list.append(texttovec.get_vec(title))
        if len(inner_list) == 0:
            aid2titlevec[aid] = np.zeros(300)
        else:
            aid2titlevec[aid] = np.mean(np.array(inner_list), axis=0)
    if mission == 'train':
        save_pickle(aid2titlevec, os.path.join(NEW_DATA_V2_DIR, 'aid2titlevec.pkl'))
    elif mission == 'test':
        save_pickle(aid2titlevec, os.path.join(NEW_DATA_DIR, 'aid2titlevec.pkl'))

    # ------------------------------------------
    # save format: aid2abstractvec --> {aid: [mean value]}
    aid2abstractvec = {}
    for aid in tqdm.tqdm(aid2pids.keys()):
        papers = aid2pids[aid]
        inner_list = []
        for pid_with_index in papers:
            pid, index = pid_with_index.split('-')
            abstract = whole_author_profile_pub[pid].get('abstract')
            if abstract is None:
                continue
            inner_list.append(texttovec.get_vec(abstract))
        if len(inner_list) == 0:
            aid2abstractvec[aid] = np.zeros(300)
        else:
            aid2abstractvec[aid] = np.mean(np.array(inner_list), axis=0)
    if mission == 'train':
        save_pickle(aid2abstractvec, os.path.join(NEW_DATA_V2_DIR, 'aid2abstractvec.pkl'))
    elif mission == 'test':
        save_pickle(aid2abstractvec, os.path.join(NEW_DATA_DIR, 'aid2abstractvec.pkl'))


def sample_data(n_neg):
    """
    there are some 'bug' in the original sampled pair.
    try a more sensetive data sample method.
    we only choose the author's last year's paper as posi-pair, and other 2 authors as
    neg-pair.

    other posible reason is that it's a information leakage !!!!
    """
    def get_pid_with_index(whole_author_profile_pub, pid, name):
        authors = whole_author_profile_pub[pid]['authors']
        authors_names = [clean_name(item['name']) for item in authors]
        index = get_name_index(name, authors_names)
        return '%s-%d' % (pid, index)

    whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    test_profile_path = os.path.join(SPLIT_DIR, 'test_profile-last1year.json')
    name2aids = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'name2aids.pkl'))
    test_profile = load_json(test_profile_path)
    # posi-pair, neg-pair --> [(aid, pid), ...]
    posi_pair = []
    neg_pair = []
    for aid in tqdm.tqdm(test_profile):
        name = test_profile[aid]['name']
        papers = test_profile[aid]['papers']
        papers = [get_pid_with_index(whole_author_profile_pub, pid, name) for pid in papers]
        if len(papers) == 0:
            continue
        # positive pair
        posi_pair.extend([(aid, pid) for pid in papers])

        # negative pair
        candidate_aids_ = name2aids[name]
        candidate_aids = candidate_aids_.copy().tolist()
        candidate_aids.remove(aid)
        if len(candidate_aids) == 0:
            continue
        candidate_aids = np.random.choice(candidate_aids, min([len(candidate_aids), n_neg]))
        neg_pair.extend([(neg_aid, pid) for neg_aid in candidate_aids for pid in papers])
    os.makedirs(NEW_DATA_V3_DIR, exist_ok=True)
    posi_pair_path = os.path.join(NEW_DATA_V3_DIR, 'posi-pair-list-extend%s.pkl' % n_neg)
    neg_pair_path = os.path.join(NEW_DATA_V3_DIR, 'neg-pair-list-extend%s.pkl' % n_neg)
    save_pickle(posi_pair, posi_pair_path)
    save_pickle(neg_pair, neg_pair_path)


def split_pair():
    pos_pair_set = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'posi-pair-list-extend1.pkl'))
    neg_pair_set = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'neg-pair-list-extend1.pkl'))
    pos_pid2aid = {}
    for pair in tqdm.tqdm(pos_pair_set):
        pos_pid2aid[pair[1]] = pair[0]
    neg_pid2aid = {}
    for pair in tqdm.tqdm(neg_pair_set):
        neg_pid2aid[pair[1]] = pair[0]
    pos_pid = set(pos_pid2aid.keys())
    neg_pid = set(neg_pid2aid.keys())
    innter_pid = list(pos_pid & neg_pid)

    train_pid, test_pid = train_test_split(innter_pid, test_size=0.15, shuffle=True)
    print(len(train_pid))
    print(len(test_pid))
    train_posi_pair_list = []
    train_neg_pair_list = []
    test_posi_pair_list = []
    test_neg_pair_list = []
    for pid in tqdm.tqdm(train_pid):
        train_posi_pair_list.append((pos_pid2aid[pid], pid))
        train_neg_pair_list.append((neg_pid2aid[pid], pid))
    for pid in tqdm.tqdm(test_pid):
        test_posi_pair_list.append((pos_pid2aid[pid], pid))
        test_neg_pair_list.append((neg_pid2aid[pid], pid))
    save_pickle(train_posi_pair_list, os.path.join(NEW_DATA_V3_DIR, 'train-posi-pair-list.pkl'))
    save_pickle(train_neg_pair_list, os.path.join(NEW_DATA_V3_DIR, 'train-neg-pair-list.pkl'))
    save_pickle(test_posi_pair_list, os.path.join(NEW_DATA_V3_DIR, 'test-posi-pair-list.pkl'))
    save_pickle(test_neg_pair_list, os.path.join(NEW_DATA_V3_DIR, 'test-neg-pair-list.pkl'))


def get_features(aid_pid_pair, pid_info_dict, aid_author_info_dict, aid_year_info_dict, aid_venue_dict, aid_org_year_list, aid_keywords_dict, aid_year_all_info_dict, org_info_set, aid_venue_set, aid_keywords_set):
    feature = [
        *get_coauthor_v2(aid_pid_pair, aid_author_info_dict, pid_info_dict),
        *get_year_diff_v2(aid_year_info_dict, pid_info_dict),
        *get_venue_score_v2(aid_venue_dict, pid_info_dict),
        *get_org_score_v2(aid_pid_pair, aid_org_year_list, pid_info_dict),
        *get_key_word_num_v2(aid_keywords_dict, pid_info_dict),
        *get_relative_year_feature(aid_pid_pair, aid_year_all_info_dict, pid_info_dict),
        *get_org_with_set_score(aid_pid_pair, pid_info_dict, org_info_set),
        *get_venue_with_set_score(aid_venue_set, pid_info_dict),
        *get_keywords_with_set_score(aid_keywords_set, pid_info_dict),
    ]
    return feature


def create_title_abstract_vec(mission='title'):
    if mission == 'abstract':
        aid2cate = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2abstractvec.pkl'))
    elif mission == 'title':
        aid2cate = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2titlevec.pkl'))
    else:
        raise ValueError("mission value error")

    texttovec = TextToVec()
    pos_pair_set = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'posi-pair-list-extend1.pkl'))
    neg_pair_set = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'neg-pair-list-extend1.pkl'))
    whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)

    data = []
    for pair in tqdm.tqdm(pos_pair_set):
        aid = pair[0]
        pid_with_index = pair[1]
        pid, _ = pid_with_index.split('-')
        info = whole_author_profile_pub[pid].get(mission)
        if info is None:
            emb = np.zeros(300)
        else:
            emb = texttovec.get_vec(info)
        emb_pair = (aid2cate[aid], emb)
        data.append(emb_pair)
    for pair in tqdm.tqdm(neg_pair_set):
        aid = pair[0]
        pid_with_index = pair[1]
        pid, _ = pid_with_index.split('-')
        info = whole_author_profile_pub[pid].get(mission)
        if info is None:
            emb = np.zeros(300)
        else:
            emb = texttovec.get_vec(info)
        emb_pair = (aid2cate[aid], emb)
        data.append(emb_pair)
    save_pickle(data, os.path.join(NEW_DATA_V3_DIR, 'data-%s-emb-pair-list.pkl' % mission))


def emb_pair_to_distance(text_model_name, mission, original_emb, save_path):
    """
    original_emb shape: [(emb0, meb1), ...]
    type: numpy.ndarray
    """
    if mission != 'title' and mission != 'abstract':
        raise ValueError('mission value error')
    triplet_model = TripletModel()
    triplet_model.load_state_dict(torch.load(os.path.join('./text-model', text_model_name)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    triplet_model = triplet_model.to(device)

    original_emb0 = np.stack([pair[0].tolist() for pair in original_emb])
    original_emb1 = np.stack([pair[1].tolist() for pair in original_emb])
    original_emb0 = np.expand_dims(original_emb0, axis=1)
    original_emb1 = np.expand_dims(original_emb1, axis=1)

    original_emb0 = torch.from_numpy(original_emb0).to(device).to(torch.float)
    original_emb1 = torch.from_numpy(original_emb1).to(device).to(torch.float)

    triplet_model.eval()
    with torch.no_grad():
        emb0 = triplet_model.get_emb(original_emb0)
        emb1 = triplet_model.get_emb(original_emb1)

    emb_sidtance = torch.sqrt(torch.sum(torch.pow((emb0 - emb1), 2), dim=1))
    emb_sidtance = emb_sidtance.cpu().numpy()
    df = pd.DataFrame(data=emb_sidtance, columns=[mission])
    df.to_pickle(save_path)


def add_text_feature_for_train():
    # abstract_pair_list = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'data-abstract-emb-pair-list.pkl'))
    title_pair_list = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'data-title-emb-pair-list.pkl'))

    emb_pair_to_distance(
        'tm.title.1.checkpoint.pth', 'title', title_pair_list,
        os.path.join(NEW_DATA_V3_DIR, 'data-title-distance-df.pkl')
    )


def create_feature(n_neg=None, begin=None, end=None):
    """
    create paper-author wise features
    pair(pid1, pid2), pid1 is referrence to the author !

    coauthors_count, coauthors_count/all_coauthors_count, coauthors_count/this_paper_coauthor_count,
    this_paper_coauthor_count, this_coauthor_count/all_coauthor, this_coauthor_count/this_paper_author

    min-diff, max-diff, mean-diff, meadian-diff,
    min_max_avg-diff, this_year_count_by_all_year, in-range?[range is the min and the max, sequential],
    this_year_count, this_year_in?[is the category, the exact year]
    before_year_diff, last_year_diff(before and last [1 year or 2 years]),

    max_venue_score, mean_venue_score, max_keyword_score, mean_keyword_score,
    max_org_score, mean_org_score, max_org_year_abs_diff,

    before_last_year_coauthor_score,
    before_last_year_org_score, before_last_year_venue_score, before_last_year_keyword_score,
    before_last_year_coauthor_score
    (before and last [1 year or 2 years])
    """
    if n_neg is None:
        pos_pair_set = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'posi-pair-list-extend1.pkl'))
        neg_pair_set = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'neg-pair-list-extend1.pkl'))
    else:
        pos_pair_set = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'posi-pair-list-last2year-extend%s.pkl' % n_neg))
        neg_pair_set = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'neg-pair-list-last2year-extend%s.pkl' % n_neg))

    whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    aid2yearinfo = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2yearinfo.pkl'))
    aid2coauthor = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2coauthor.pkl'))
    aid2venue = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2venue.pkl'))
    aid2keywords = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2keywords.pkl'))
    aid2year = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2year.pkl'))
    aid2orgwithyear = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2orgwithyear.pkl'))
    aid2orgset = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2orgset.pkl'))
    aid2venueset = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2venueset.pkl'))
    aid2keywordsset = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2keywordsset.pkl'))

    name2aids = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'name2aids.pkl'))
    aid2pids = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2pids.pkl'))

    # aid2name --> {aid: name}, pid2aid --> {pid: aid}
    aid2name, pid2aid = {}, {}
    for name in name2aids:
        aids = name2aids[name]
        for aid in aids:
            aid2name[aid] = name
    for aid in aid2pids:
        pids = aid2pids[aid]
        for pid in pids:
            pid2aid[pid] = aid
    all_pids_len = len(pid2aid)
    # cols = [
    #     'coauthors_count', 'coauthors_count_by_all_count', 'coauthors_count_by_this_coauthor_count',
    #     'this_paper_coauthor_count', 'this_paper_coathor_count_by_all_coauthor', 'this_paper_coauthor_count_by_this_paper_coauthor_count',
    #     'min_diff', 'max_diff', 'mean_diff', 'meadian_diff', 'min_max_avg_diff', 'is_in_range',
    #     'this_year_count', 'this_year_count_by_all_year', 'is_in_cate_range', 'before_one', 'before_two',
    #     'later_one', 'later_two', 'venue_max_score', 'venue_mean_score', 'venue_max_add_score',
    #     'venue_mean_add_score', 'venue_is_match', 'venue_score_add_score', 'org_max_score', 'org_mean_score',
    #     'org_max_add_score', 'org_mean_add_score', 'org_is_match', 'org_score_add_score', 'org_year_abs_diff',
    #     'keywords_max_score', 'keywords_mean_score', 'rela_year_diff', 'rela_coauthor_count',
    #     'rela_coauthor_count_by1', 'rela_coauthor_count_by2', 'rela_org_max_score',
    #     'rela_org_mean_score', 'rela_org_max_add_score', 'rela_org_mean_add_score',
    #     'rela_org_score_add_score', 'rela_venue_max_score', 'rela_venue_mean_score',
    #     'rela_venue_max_add_score', 'rela_venue_mean_add_score', 'rela_venue_score_add_score',
    #     'rela_keyword_max_score', 'rela_keyword_mean_score', 'label'
    # ]
    # original
    # length: 50 + 1
    cols = [
        'coauthors_count', 'coauthors_count_by_all_count', 'coauthors_count_by_this_coauthor_count',
        'this_paper_coauthor_count', 'this_paper_coathor_count_by_all_coauthor', 'this_paper_coauthor_count_by_this_paper_coauthor_count',
        'min_diff', 'max_diff', 'mean_diff', 'meadian_diff', 'min_max_avg_diff', 'is_in_range',
        'this_year_count', 'this_year_count_by_all_year', 'is_in_cate_range', 'before_one', 'before_two',
        'later_one', 'later_two', 'venue_max_score', 'venue_mean_score', 'venue_max_add_score',
        'venue_mean_add_score', 'venue_is_match', 'venue_score_add_score', 'org_max_score', 'org_mean_score',
        'org_max_add_score', 'org_mean_add_score', 'org_is_match', 'org_score_add_score', 'org_year_abs_diff',
        'keywords_max_score', 'keywords_mean_score', 'rela_year_diff', 'rela_coauthor_count',
        'rela_coauthor_count_by1', 'rela_coauthor_count_by2', 'rela_org_max_score',
        'rela_org_mean_score', 'rela_org_max_add_score', 'rela_org_mean_add_score',
        'rela_org_score_add_score', 'rela_venue_max_score', 'rela_venue_mean_score',
        'rela_venue_max_add_score', 'rela_venue_mean_add_score', 'rela_venue_score_add_score',
        'rela_keyword_max_score', 'rela_keyword_mean_score', 'org_set_count', 'org_set_count_by_all_count',
        'org_set_count_by_this_count', 'venue_word_count', 'venue_word_count_by_all_count',
        'venue_word_count_by_this_venue_count', 'keyword_count', 'keyword_count_by_all_count',
        'keyword_count_by_this_keyword_count', 'label'
    ]
    # new, with set score, last 9 columns
    # length: 59 + 1
    data = []
    for pair in tqdm.tqdm(pos_pair_set):
        aid = pair[0]
        pair_pid = pair[1]
        # new_pair = (aid, pair_pid)
        new_pair = pair
        pid_info_dict = whole_author_profile_pub[pair_pid.split('-')[0]]
        aid_author_info_dict = aid2coauthor[aid]
        aid_year_info_dict = aid2year[aid]
        aid_venue_dict = aid2venue[aid]
        aid_org_year_list = aid2orgwithyear[aid]
        aid_keywords_dict = aid2keywords[aid]
        aid_year_all_info_dict = aid2yearinfo[aid]
        org_info_set = aid2orgset[aid]
        aid_venue_set = aid2venueset[aid]
        aid_keywords_set = aid2keywordsset[aid]
        feature = get_features(new_pair, pid_info_dict, aid_author_info_dict, aid_year_info_dict, aid_venue_dict, aid_org_year_list, aid_keywords_dict, aid_year_all_info_dict, org_info_set, aid_venue_set, aid_keywords_set)
        # feature.append(len(aid2pids[aid]) / all_pids_len)  # add paper count
        feature.append(1)
        data.append(feature)
    print('Posi length: ', len(data))
    for pair in tqdm.tqdm(neg_pair_set):
        aid = pair[0]
        pair_pid = pair[1]
        # new_pair = (aid, pair_pid)
        new_pair = pair
        pid_info_dict = whole_author_profile_pub[pair_pid.split('-')[0]]
        aid_author_info_dict = aid2coauthor[aid]
        aid_year_info_dict = aid2year[aid]
        aid_venue_dict = aid2venue[aid]
        aid_org_year_list = aid2orgwithyear[aid]
        aid_keywords_dict = aid2keywords[aid]
        aid_year_all_info_dict = aid2yearinfo[aid]
        org_info_dict = aid2orgset[aid]
        aid_venue_set = aid2venueset[aid]
        aid_keywords_set = aid2keywordsset[aid]
        feature = get_features(new_pair, pid_info_dict, aid_author_info_dict, aid_year_info_dict, aid_venue_dict, aid_org_year_list, aid_keywords_dict, aid_year_all_info_dict, org_info_dict, aid_venue_set, aid_keywords_set)
        # feature.append(len(aid2pids[aid]) / all_pids_len)
        feature.append(0)
        data.append(feature)
    print('All length: ', len(data))
    df = pd.DataFrame(data=data, columns=cols)
    print(df.head())
    if n_neg is None:
        df.to_pickle(os.path.join(NEW_DATA_V3_DIR, 'data-last1year-withsetinfo-extend1.pkl'))
    else:
        df.to_pickle(os.path.join(NEW_DATA_V3_DIR, 'data-last2year-withsetinfo-extend%s.pkl' % n_neg))


def gen_test_title_abstract_vec(mission='title'):
    if mission == 'title':
        aid2cate = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2titlevec.pkl'))
    elif mission == 'abstract':
        aid2cate = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2abstractvec.pkl'))
    else:
        raise ValueError('mission value error')

    valid_nuass = load_json(VALID_UNASS_PATH)
    valid_pub = load_json(VALID_PUB_PATH)
    name2aids = load_pickle(os.path.join(NEW_DATA_DIR, 'name2aids.pkl'))
    texttovec = TextToVec()

    all_authors_name = list(name2aids.keys())
    # test_cate_feature --> {pid-with-index: {candidate-aids: [...], data: [(emb0, meb1), ...]}}
    test_cate_feature = {}
    for pid_with_index in tqdm.tqdm(valid_nuass):
        inner_dict = {}
        now_pid, index = pid_with_index.split('-')
        author_name = valid_pub[now_pid]['authors'][int(index)]['name']
        author_name = clean_name(author_name)
        index = get_name_index(author_name, all_authors_name)
        author_name = all_authors_name[index]

        candidate_aids = name2aids[author_name]
        inner_dict['candidate-aids'] = candidate_aids
        data = []
        for aid in candidate_aids:
            info = valid_pub[now_pid].get(mission)
            if info is None:
                emb = np.zeros(300)
            else:
                emb = texttovec.get_vec(info)
            emb_pair = (aid2cate[aid], emb)
            data.append(emb_pair)
        inner_dict['data'] = data
        test_cate_feature[pid_with_index] = inner_dict
    save_pickle(test_cate_feature, os.path.join(TEST_FEATURE_DIR_V2, 'test-%s-emb-pair.pkl' % mission))


def add_text_feature_for_test():
    test_abstract_emb_pair = load_pickle(os.path.join(TEST_FEATURE_DIR_V2, 'test-abstract-emb-pair.pkl'))
    test_title_emb_pair = load_pickle(os.path.join(TEST_FEATURE_DIR_V2, 'test-title-emb-pair.pkl'))
    valid_nuass = load_json(VALID_UNASS_PATH)

    title_emb_pair = []
    for pid_with_index in tqdm.tqdm(valid_nuass):
        for pair in test_title_emb_pair[pid_with_index]['data']:
            title_emb_pair.append(pair)
    emb_pair_to_distance(
        'tm.title.1.checkpoint.pth', 'title', title_emb_pair,
        os.path.join(TEST_FEATURE_DIR_V2, 'test-title-distance-df.pkl')
    )


def gen_test_feature():
    # process test data and save in pickle
    # testdatafeatures --> {pid-with-index: {candidate-aids: [...], data: [[xxx], [xxx], [xxx]...]}}
    valid_nuass = load_json(VALID_UNASS_PATH)
    valid_pub = load_json(VALID_PUB_PATH)
    # whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    aid2yearinfo = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2yearinfo.pkl'))
    aid2coauthor = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2coauthor.pkl'))
    aid2venue = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2venue.pkl'))
    aid2keywords = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2keywords.pkl'))
    aid2year = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2year.pkl'))
    aid2orgwithyear = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2orgwithyear.pkl'))
    name2aids = load_pickle(os.path.join(NEW_DATA_DIR, 'name2aids.pkl'))
    aid2pids = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2pids.pkl'))
    aid2orgset = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2orgset.pkl'))
    aid2venueset = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2venueset.pkl'))
    aid2keywordsset = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2keywordsset.pkl'))

    all_pids_len = 0
    for aid in aid2pids:
        all_pids_len += len(aid2pids[aid])
    testdatafeatures = {}
    all_authors_name = list(name2aids.keys())
    # author_name_count = defaultdict(int)
    for pid_with_index in tqdm.tqdm(valid_nuass):
        inner_dict = {}
        now_pid, index = pid_with_index.split('-')
        author_name = valid_pub[now_pid]['authors'][int(index)]['name']
        author_name = clean_name(author_name)
        index = get_name_index(author_name, all_authors_name)
        author_name = all_authors_name[index]
        # author_name_count[author_name] += 1
        # continue

        candidate_aids = name2aids[author_name]
        inner_dict['candidate-aids'] = candidate_aids
        data = []
        for aid in candidate_aids:
            new_pair = (aid, pid_with_index)
            pid_info_dict = valid_pub[now_pid]
            aid_author_info_dict = aid2coauthor[aid]
            aid_year_info_dict = aid2year[aid]
            aid_venue_dict = aid2venue[aid]
            aid_org_year_list = aid2orgwithyear[aid]
            aid_keywords_dict = aid2keywords[aid]
            aid_year_all_info_dict = aid2yearinfo[aid]
            org_info_set = aid2orgset[aid]
            aid_venue_set = aid2venueset[aid]
            aid_keywords_set = aid2keywordsset[aid]
            data.append(get_features(new_pair, pid_info_dict, aid_author_info_dict, aid_year_info_dict, aid_venue_dict, aid_org_year_list, aid_keywords_dict, aid_year_all_info_dict, org_info_set, aid_venue_set, aid_keywords_set))
            data[-1].append(len(aid2pids[aid]) / all_pids_len)
        data = np.array(data)
        inner_dict['data'] = data
        testdatafeatures[pid_with_index] = inner_dict
    save_pickle(testdatafeatures, os.path.join(TEST_FEATURE_DIR_V2, 'testdatafeatures-withsetinfo-papercount.pkl'))
    # names = list(author_name_count.keys())
    # values = list(author_name_count.values())
    # sns.barplot(x=names, y=values)
    # plt.savefig('./img/test_name_count.png')


def gen_one_test_feature():
    # process test data and save in pickle
    # testdatafeatures --> {pid-with-index: {candidate-aids: [...], data: [[xxx], [xxx], [xxx]...]}}
    valid_nuass = load_json(VALID_UNASS_PATH)
    valid_pub = load_json(VALID_PUB_PATH)
    # whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    aid2yearinfo = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2yearinfo.pkl'))
    aid2coauthor = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2coauthor.pkl'))
    aid2venue = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2venue.pkl'))
    aid2keywords = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2keywords.pkl'))
    aid2year = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2year.pkl'))
    aid2orgwithyear = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2orgwithyear.pkl'))
    name2aids = load_pickle(os.path.join(NEW_DATA_DIR, 'name2aids.pkl'))
    # aid2pids = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2pids.pkl'))

    testdatafeatures = {}
    all_authors_name = list(name2aids.keys())
    all_aids = []
    for key in name2aids:
        aids = name2aids[key]
        all_aids.extend(aids.tolist())
    all_aids = np.array(all_aids)
    for pid_with_index in tqdm.tqdm(valid_nuass):
        inner_dict = {}
        now_pid, index = pid_with_index.split('-')
        author_name = valid_pub[now_pid]['authors'][int(index)]['name']
        author_name = clean_name(author_name)
        index = get_name_index(author_name, all_authors_name)
        author_name = all_authors_name[index]

        candidate_aids = name2aids[author_name]
        candidate_aids = all_aids
        inner_dict['candidate-aids'] = candidate_aids
        data = []
        for aid in candidate_aids:
            print(aid)
            new_pair = (aid, pid_with_index)
            pid_info_dict = valid_pub[now_pid]
            aid_author_info_dict = aid2coauthor[aid]
            aid_year_info_dict = aid2year[aid]
            aid_venue_dict = aid2venue[aid]
            aid_org_year_list = aid2orgwithyear[aid]
            aid_keywords_dict = aid2keywords[aid]
            aid_year_all_info_dict = aid2yearinfo[aid]
            data.append(get_features(new_pair, pid_info_dict, aid_author_info_dict, aid_year_info_dict, aid_venue_dict, aid_org_year_list, aid_keywords_dict, aid_year_all_info_dict))
        data = np.array(data)
        inner_dict['data'] = data
        testdatafeatures[pid_with_index] = inner_dict
        break
    save_pickle(testdatafeatures, './testdatafeatures_one.pkl')


def aggregate(paths, n_sample):
    dfs = [pd.read_pickle(path) for path in paths]
    datas = [df.values for df in dfs]
    datas_tuple = tuple(datas)
    data = np.concatenate(datas_tuple, axis=0)
    cols = pd.read_pickle(paths[0]).columns
    df = pd.DataFrame(data=data, columns=cols)
    df.drop_duplicates(inplace=True)
    print(df.head())
    df.to_pickle(os.path.join(NEW_DATA_V2_DIR, 'data-%d.pkl' % n_sample))


def cut_data_to_11():
    df = pd.read_pickle(os.path.join(NEW_DATA_V3_DIR, 'data-last2year-withsetinfo-extend2.pkl'))
    label_1_count = (df['label'] == 1).sum()
    df_label_0 = df[df['label'] == 0].sample(n=label_1_count, random_state=1129)
    df_label_1 = df[df['label'] == 1]
    data = np.concatenate((df_label_0.values, df_label_1.values), axis=0)
    df = pd.DataFrame(data=data, columns=df.columns)
    df.to_pickle(os.path.join(NEW_DATA_V3_DIR, 'data-last2year-withsetinfo-sample11.pkl'))

    # df = pd.read_pickle(os.path.join(NEW_DATA_V3_DIR, 'data-last2year-withsetinfo-sample11.pkl'))
    # sns.countplot(x='label', data=df)
    # plt.savefig('./img/count_v2_7.png')


def train(model_info):
    os.makedirs(STACK_MODEL_DIR_v2, exist_ok=True)
    df = pd.read_pickle(os.path.join(NEW_DATA_V3_DIR, 'data-last1year-withsetinfo-extend1.pkl'))
    data_title_distance = pd.read_pickle(os.path.join(NEW_DATA_V3_DIR, 'data-title-distance-df.pkl'))
    all_data = np.concatenate((df.values, data_title_distance.values.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(data=all_data, columns=list(df.columns) + list(data_title_distance.columns))
    print(df.head())
    df = shuffle(df, random_state=RANDOM_SEED)
    print(df.head())

    train_data = df[model_info['cols']].values
    train_y = df['label'].values
    print(train_data.shape)

    ss = StandardScaler()
    train_data = ss.fit_transform(train_data)
    save_pickle(ss, model_info['ss_path'])

    models = model_info['model']
    params = model_info['model_param']
    sm = StackModel(models, params)
    sm.fit(train_data, train_y)
    save_pickle(sm, model_info['model_path'])
    return sm


def predict(models):
    valid_nuass = load_json(VALID_UNASS_PATH)
    testdatafeatures = load_pickle(os.path.join(TEST_FEATURE_DIR_V2, 'testdatafeatures-withsetinfo-papercount.pkl'))
    title_feature_df = pd.read_pickle(os.path.join(TEST_FEATURE_DIR_V2, 'test-title-distance-df.pkl'))
    title_feature = title_feature_df.values

    ss = load_pickle(os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-withsetinfo-title-papercount-11.pkl'))
    submission = defaultdict(list)
    # num = 0
    for pid_with_index in tqdm.tqdm(valid_nuass):
        candidate_aids = testdatafeatures[pid_with_index]['candidate-aids']
        data = testdatafeatures[pid_with_index]['data']
        data_length = len(candidate_aids)
        title_data = title_feature[:data_length]
        title_feature = title_feature[data_length:]
        data = np.concatenate((data, title_data), axis=1)

        data = ss.transform(data)
        output = models.predict_proba(data)
        # print('-'*50)
        # print(pid_with_index)
        # print(candidate_aids)
        # print(output)
        # print(np.max(output))
        # print(np.sort(output)[-10:])
        predict_author = candidate_aids[np.argmax(output)]
        submission[predict_author].append(pid_with_index.split('-')[0])
        # print(predict_author)
        # num += 1
        # if num == 20:
        #     break
    save_json(submission, os.path.join(OUT_DIR_v2, 'submission-stack-withsetinfo-title-last1year-11-papercount.json'))


def convert_glove():
    print('convert glove begin !')
    word2vec = {}
    with open('./glove.840B.300d.txt', 'r') as f:
        for line in tqdm.tqdm(f):
            split_line = line.split()
            key = ' '.join(split_line[:-300])
            value = np.array(split_line[-300:], dtype=np.float)
            word2vec[key] = value
    print('Done')
    save_pickle(word2vec, './glove.word2vec.dict.pkl')


if __name__ == "__main__":
    convert_glove()
    split_data(1)
    preprocessing('train')
    preprocessing('test')
    for i in [1, 2, 3]:
        sample_data(i)
    split_pair()
    create_title_abstract_vec('title')
