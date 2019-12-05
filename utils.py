#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2019/11/14 22:02:18
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
import json
import pickle
from pyjarowinkler import distance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string


class TextToVec:
    def __init__(self):
        super().__init__()
        self.my_stopwords = set(stopwords.words('english'))
        self.num_pattern = re.compile(r'\d+')
        self.remove_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.word2vec_dict = load_pickle('./glove.word2vec.dict.pkl')

    def clean_text(self, str_info):
        str_lower = str_info.lower().strip()
        result = str_lower.translate(self.remove_punctuation)
        result = self.num_pattern.sub('', result)
        tokens = word_tokenize(result)
        result = [word for word in tokens if word not in self.my_stopwords]
        result = [self.lemmatizer.lemmatize(word) for word in result]
        # print(result)
        return result

    def get_vec(self, str_info):
        result = self.clean_text(str_info)
        data = []
        for word in result:
            data.append(self.word2vec_dict.get(word, np.zeros(300)).tolist())
        if len(data) == 0:
            data = np.zeros(300)
        else:
            data = np.mean(np.array(data), axis=0)
        return data


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def clean_name(name):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", "").replace("-", " ").replace("_", ' ').split()]
    full_name = '_'.join(x)
    return full_name


def get_name_index(target_name, name_list):
    scores = []
    for name in name_list:
        if name == '':
            scores.append(0)
            continue
        score = distance.get_jaro_distance(target_name, name, winkler=True, scaling=0.1)
        target_component = set(target_name.split('_'))
        name_component = set(name.split('_'))
        add_score = len(target_component & name_component) / len(target_component | name_component)
        score = score + add_score
        scores.append(score)
    # print('-'*50)
    # index = np.argsort(scores)
    # print(target_name)
    # print(np.array(name_list)[index][-5:])
    return np.argmax(scores)


def get_coauthor(pid_pair, info_pair):
    authors = (info_pair[0]['authors'], info_pair[1]['authors'])
    authors0 = [clean_name(item['name']) for item in authors[0]]
    authors1 = [clean_name(item['name']) for item in authors[1]]
    authors0.pop(int(pid_pair[0].split('-')[1]))
    authors1.pop(int(pid_pair[1].split('-')[1]))
    authors0_set = set(authors0)
    authors1_set = set(authors1)
    coau = len(authors0_set & authors1_set)
    if len(authors0_set) == 0:
        coau_by_authors0 = 0
    else:
        coau_by_authors0 = coau / len(authors0_set)

    if len(authors1_set) == 0:
        coau_by_authors1 = 0
    else:
        coau_by_authors1 = coau / len(authors1_set)
    return coau, coau_by_authors0, coau_by_authors1


def get_org_score(pid_pair, info_pair):
    authors = (info_pair[0]['authors'], info_pair[1]['authors'])
    author = (authors[0][int(pid_pair[0].split('-')[1])], authors[1][int(pid_pair[1].split('-')[1])])
    org0 = author[0].get('org', '').lower()
    org1 = author[1].get('org', '').lower()

    if all([org0 == '', org1 == '']):           # TODO better strategy
        score = -999
        add_score = 0
    elif any([org0 == '', org1 == '']):
        score = 0
        add_score = 0
    else:
        score = distance.get_jaro_distance(org0, org1)
        org0_set = set(org0.split())
        org1_set = set(org1.split())
        if len(org0_set | org1_set) == 0:
            add_score = 0
        else:
            add_score = len(org0_set & org1_set) / len(org0_set | org1_set)
        # score += add_score
    return score, add_score, score + add_score


def get_year_diff(info_pair):
    # TODO abs-diff, min-diff, max-diff, mean-diff, meadian-diff, min_max_avg-diff, in-range?
    year0 = info_pair[0].get('year', '0')
    year1 = info_pair[1].get('year', '0')
    if any([year0 == '', year1 == '']):
        return -999

    year0, year1 = int(year0), int(year1)
    if any([year0 < 1500, year1 < 1500, year0 > 2100, year1 > 2100]):
        return -999
    return abs(year0 - year1)


def get_key_word_num(info_pair):
    # 貌似不靠谱 ！！！
    keywords0 = info_pair[0].get('keywords')
    keywords1 = info_pair[1].get('keywords')
    flag = [keywords0 is None, keywords1 is None]
    if all(flag):
        return -999, -999
    elif any(flag):
        return 0, 0
    else:
        pairs = [(a.lower(), b.lower()) for a in keywords0 if a != '' for b in keywords1 if b != '']
        if len(pairs) == 0:
            return -999, -999
        scores = [distance.get_jaro_distance(pair[0], pair[1]) for pair in pairs]
        return max(scores), np.mean(scores)


def get_venue_score(info_pair):
    venue0 = info_pair[0].get('venue', '')
    venue1 = info_pair[1].get('venue', '')
    flag = [venue0 == '', venue1 == '']
    if all(flag):
        score = -999
        add_score = 0
    elif any(flag):
        score = 0
        add_score = 0
    else:
        venue0, venue1 = venue0.lower(), venue1.lower()
        score = distance.get_jaro_distance(venue0, venue1)
        venue0_set = set(venue0.split())
        venue1_set = set(venue1.split())
        if len(venue0_set | venue1_set) == 0:
            add_score = 0
        else:
            add_score = len(venue0_set & venue1_set) / len(venue0_set | venue1_set)
        # score += add_score
    return score, add_score, score + add_score


def get_coauthor_v2(aid_pid_pair, aid_author_info_dict, pid_info_dict):
    index = int(aid_pid_pair[1].split('-')[1])
    authors = pid_info_dict['authors']
    authors = [clean_name(item['name']) for item in authors]
    authors.pop(index)
    count = 0
    this_paper_count = 0
    for author_name in authors:
        if author_name in aid_author_info_dict.keys():
            this_paper_count += 1
            count += aid_author_info_dict[author_name]
    all_count = np.sum(list(aid_author_info_dict.values()))
    if all_count == 0:
        count_by_all_count = -999
    else:
        count_by_all_count = count / all_count
    if len(aid_author_info_dict) == 0:
        this_paper_count_by_author_coauthors = -999
    else:
        this_paper_count_by_author_coauthors = this_paper_count / len(aid_author_info_dict)
    if len(authors) == 0:
        # count_by_this_paper_author = 0
        # this_paper_count_by_this_paper_author = 0
        count_by_this_paper_author = -999
        this_paper_count_by_this_paper_author = -999
    else:
        count_by_this_paper_author = count / len(authors)
        this_paper_count_by_this_paper_author = this_paper_count / len(authors)
    return count, count_by_all_count, count_by_this_paper_author, this_paper_count, this_paper_count_by_author_coauthors, this_paper_count_by_this_paper_author


def get_year_diff_v2(aid_year_info_dict, pid_info_dict):
    if aid_year_info_dict is None:
        return [-999] * 11 + [999] * 2
    year = pid_info_dict.get('year', '0')
    if year == '':
        year = 0
    else:
        year = int(year)
    if year <= 1500 or year >= 2100:
        year = 0
    if year == 0:
        return [-999] * 11 + [999] * 2
    # else:                     # 测试year !!! 猜测 !!!
    #     year = year + 4
    min_diff = aid_year_info_dict['min'] - year
    max_diff = aid_year_info_dict['max'] - year
    mean_diff = aid_year_info_dict['mean'] - year
    meadian_diff = aid_year_info_dict['median'] - year
    min_max_avg_diff = aid_year_info_dict['min_max_avg'] - year
    is_in_range = 0
    if min_diff <= 0 and max_diff >= 0:
        is_in_range = 1
    year_list = aid_year_info_dict['year_list']
    year_array = np.array(year_list)

    this_year_count = np.sum(year_array == year)
    this_year_count_by_all_year = this_year_count / len(year_list)
    if this_year_count > 0:
        is_in_cate_range = 1
    else:
        is_in_cate_range = 0
    year_unique = np.unique(year_array)
    year_unique_diff = year_unique - year

    # sort --> small to big
    year_smaller = np.sort(year_unique_diff[year_unique_diff < 0])
    year_bigger = np.sort(year_unique_diff[year_unique_diff > 0])
    if len(year_smaller) < 2:
        if len(year_smaller) < 1:
            before_one, before_two = -999, -999
        else:
            before_one = year_smaller[-1]
            before_two = -999
    else:
        before_one = year_smaller[-1]
        before_two = year_smaller[-2]

    if len(year_bigger) < 2:
        if len(year_bigger) < 1:
            later_one, later_two = 999, 999
        else:
            later_one = year_bigger[0]
            later_two = 999
    else:
        later_one = year_bigger[0]
        later_two = year_bigger[1]
    return [
        min_diff, max_diff, mean_diff, meadian_diff, min_max_avg_diff, is_in_range,
        this_year_count, this_year_count_by_all_year, is_in_cate_range, before_one, before_two,
        later_one, later_two
    ]


def get_venue_with_set_score(aid_venue_set, pid_info_dict):
    target_venue = pid_info_dict.get('venue', '').lower()
    target_venue_set = set(target_venue.replace('-', ' ').split())
    if len(target_venue_set) == 0:
        venue_word_count = -999
        venue_word_count_by_this_venue_count = -999
    else:
        venue_word_count = len(target_venue_set & aid_venue_set)
        venue_word_count_by_this_venue_count = venue_word_count / len(target_venue_set)

    if len(aid_venue_set) == 0:
        venue_word_count_by_all_count = -999
    else:
        if venue_word_count == -999:
            venue_word_count_by_all_count = -999
        else:
            venue_word_count_by_all_count = venue_word_count / len(aid_venue_set)
    return venue_word_count, venue_word_count_by_all_count, venue_word_count_by_this_venue_count


def get_venue_score_v2(aid_venue_dict, pid_info_dict):
    venue = pid_info_dict.get('venue', '').lower()
    if venue == '':
        max_score = -999
        mean_score = -999
        max_add_score = 0
        mean_add_score = 0
        is_match = 1
        score_add_score = -999
    else:
        add_scores = []
        venue_list = list(aid_venue_dict.keys())
        # venue_count_list = list(aid_venue_dict.values())
        scores = [distance.get_jaro_distance(venue, item) for item in venue_list]
        if len(scores) == 0:
            max_score = -999
            mean_score = -999
            max_add_score = 0
            mean_add_score = 0
            is_match = 1
            score_add_score = -999
        else:
            for item in venue_list:
                venue_set = set(venue.split())
                item_set = set(item.split())
                if len(venue_set | item_set) == 0:
                    add_score = 0
                else:
                    add_score = len(venue_set & item_set) / len(venue_set | item_set)
                add_scores.append(add_score)
            max_score_index = np.argmax(scores)
            max_add_score_index = np.argmax(add_scores)
            max_score = scores[max_score_index]
            mean_score = np.mean(scores)
            max_add_score = add_scores[max_add_score_index]
            mean_add_score = np.mean(add_scores)
            is_match = int(max_score_index == max_add_score_index)
            if is_match:
                score_add_score = max_score + max_add_score
            else:
                score_add_score = max_score + add_scores[max_score_index]
    return max_score, mean_score, max_add_score, mean_add_score, is_match, score_add_score


def get_org_with_set_score(aid_pid_pair, pid_info_dict, org_info_set):
    pid, index = aid_pid_pair[1].split('-')
    author = pid_info_dict['authors'][int(index)]
    target_org = author.get('org', '').lower().strip()
    target_org_set = set(target_org.split())
    if len(target_org_set) == 0:
        org_word_count = -999
        org_word_count_by_this_org_count = -999
    else:
        org_word_count = len(target_org_set & org_info_set)
        org_word_count_by_this_org_count = org_word_count / len(target_org_set)

    if len(org_info_set) == 0:
        org_word_count_by_all_count = -999
    else:
        if org_word_count == -999:
            org_word_count_by_all_count = -999
        else:
            org_word_count_by_all_count = org_word_count / len(org_info_set)
    return org_word_count, org_word_count_by_all_count, org_word_count_by_this_org_count


def get_org_score_v2(aid_pid_pair, aid_org_year_list, pid_info_dict):
    index = int(aid_pid_pair[1].split('-')[1])
    author = pid_info_dict['authors'][index]
    org = author.get('org', '').lower()
    year = pid_info_dict.get('year', '0')
    if year == '':
        year = 0
    else:
        year = int(year)
    if year <= 1500 or year >= 2100:
        year = 0
    if org == '' or all([item[0] == '' for item in aid_org_year_list]):
        max_score = -999
        mean_score = -999
        max_add_score = 0
        mean_add_score = 0
        is_match = 1
        score_add_score = -999
        year_abs_diff = -999
    else:
        org_list = [item[0] for item in aid_org_year_list if item[0] != '']
        year_list = [item[1] for item in aid_org_year_list if item[0] != '']
        scores = [distance.get_jaro_distance(org, item) for item in org_list]
        add_scores = []
        for item in org_list:
            org_set = set(org.split())
            item_set = set(item.split())
            if len(org_set | item_set) == 0:
                add_score = 0
            else:
                add_score = len(org_set & item_set) / len(org_set | item_set)
            add_scores.append(add_score)
        max_score_index = np.argmax(scores)
        max_add_score_index = np.argmax(add_scores)
        max_score = scores[max_score_index]
        max_add_score = add_scores[max_add_score_index]
        mean_score = np.mean(scores)
        mean_add_score = np.mean(add_scores)
        is_match = int(max_score_index == max_add_score_index)
        if is_match:
            score_add_score = max_score + max_add_score
        else:
            score_add_score = max_score + add_scores[max_score_index]
        org_array = np.array(org_list)
        year_array = np.array(year_list)
        if year != 0:
            max_match_year = year_array[org_array == org_array[max_score_index]]
            if all(max_match_year == 0):
                year_abs_diff = -999
            else:
                year_abs_diff_array = np.abs(year - max_match_year)
                year_abs_diff = np.min(year_abs_diff_array)
        else:
            year_abs_diff = -999
    return max_score, mean_score, max_add_score, mean_add_score, is_match, score_add_score, year_abs_diff


def get_keywords_with_set_score(aid_keywords_set, pid_info_dict):
    keywords = pid_info_dict.get('keywords', '')
    if len(keywords) == 0:
        keywords = []

    target_keyword_set = set()
    for keyword in keywords:
        target_keyword_set = target_keyword_set | set(keyword.lower().replace('-', ' ').split())
    if len(target_keyword_set) == 0:
        keyword_count = -999
        keyword_count_by_this_keyword_count = -999
    else:
        keyword_count = len(target_keyword_set & aid_keywords_set)
        keyword_count_by_this_keyword_count = keyword_count / len(target_keyword_set)

    if len(aid_keywords_set) == 0:
        keyword_count_by_all_count = -999
    else:
        if keyword_count == -999:
            keyword_count_by_all_count = -999
        else:
            keyword_count_by_all_count = keyword_count / len(aid_keywords_set)
    return keyword_count, keyword_count_by_all_count, keyword_count_by_this_keyword_count


def get_key_word_num_v2(aid_keywords_dict, pid_info_dict):
    keywords_list = list(aid_keywords_dict.keys())
    keywords_list = [item.lower() for item in keywords_list]
    keywords = pid_info_dict.get('keywords')
    if keywords is None:
        return -999, -999
    else:
        keywords = [item.lower() for item in keywords]
        pairs = [(a, b) for a in keywords_list if a != '' for b in keywords if b != '']
        if len(pairs) == 0:
            return -999, -999
        scores = [distance.get_jaro_distance(pair[0], pair[1]) for pair in pairs]
        max_score = np.max(scores)
        mean_score = np.mean(scores)
        return max_score, mean_score


def get_relative_year_feature(aid_pid_pair, aid_year_all_info_dict, pid_info_dict):
    year = pid_info_dict.get('year', '0')
    if year == '':
        year = 0
    else:
        year = int(year)
    if year <= 1500 or year >= 2100:
        year = 0

    if year == 0:
        year_diff = -999
        coauthor_count = 0
        coauthor_count_by1 = 0
        coauthor_count_by2 = 0
        org_max_score = -999
        org_mean_score = -999
        org_max_add_score = 0
        org_mean_add_score = 0
        org_score_add_score = -999
        venue_max_score = -999
        venue_mean_score = -999
        venue_max_add_score = 0
        venue_mean_add_score = 0
        venue_score_add_score = -999
        keyword_max_score = -999
        keyword_mean_score = -999
    else:
        year_list = list(aid_year_all_info_dict.keys())
        year_array = np.array(year_list)
        year_diff_array = np.abs(year_array - year)
        # sort --> small to big
        sort_index = np.argsort(year_diff_array)
        relative_year = year_array[sort_index[0]]
        year_diff = year_diff_array[sort_index[0]]

        index = int(aid_pid_pair[1].split('-')[1])
        authors = pid_info_dict['authors']
        authors_name = [clean_name(item['name']) for item in authors]
        coauthor_name = authors_name.pop(index)
        org = authors[index].get('org', '').lower()
        venue = pid_info_dict.get('venue', '').lower()
        keywords = pid_info_dict.get('keywords', [''])
        if len(keywords) == 0:
            keywords = ['']

        coauthor_name_list = aid_year_all_info_dict[relative_year]['coauthors']
        org_list = aid_year_all_info_dict[relative_year]['orgs']
        venue_list = aid_year_all_info_dict[relative_year]['venues']
        keywords_list = aid_year_all_info_dict[relative_year]['keywords']
        # coauthor score
        coauthor_name_set = set(coauthor_name)
        coauthor_name_list_set = set(coauthor_name_list)
        coauthor_count = len(coauthor_name_set & coauthor_name_list_set)
        if len(coauthor_name_set) == 0:
            coauthor_count_by1 = 0
        else:
            coauthor_count_by1 = coauthor_count / len(coauthor_name_set)
        if len(coauthor_name_list_set) == 0:
            coauthor_count_by2 = 0
        else:
            coauthor_count_by2 = coauthor_count / len(coauthor_name_list_set)
        # org score
        pairs = [(org, item) for item in org_list if item != '' and org != '']
        if len(pairs) == 0:
            org_max_score = -999
            org_mean_score = -999
            org_max_add_score = 0
            org_mean_add_score = 0
            org_score_add_score = -999
        else:
            scores = [distance.get_jaro_distance(pair[0], pair[1]) for pair in pairs]
            org_max_score = np.max(scores)
            org_mean_score = np.mean(scores)
            add_scores = []
            for pair in pairs:
                org1_set = set(pair[0].split())
                org2_set = set(pair[1].split())
                add_score = len(org1_set & org2_set) / len(org1_set | org2_set)
                add_scores.append(add_score)
            org_max_add_score = np.max(add_scores)
            org_mean_add_score = np.mean(add_scores)
            org_score_add_score = org_max_score + org_max_add_score
        # venue score
        pairs = [(venue, item) for item in venue_list if item != '' and venue != '']
        if len(pairs) == 0:
            venue_max_score = -999
            venue_mean_score = -999
            venue_max_add_score = 0
            venue_mean_add_score = 0
            venue_score_add_score = -999
        else:
            scores = [distance.get_jaro_distance(pair[0], pair[1]) for pair in pairs]
            venue_max_score = np.max(scores)
            venue_mean_score = np.mean(scores)
            add_scores = []
            for pair in pairs:
                venue1_set = set(pair[0].split())
                venue2_set = set(pair[1].split())
                add_score = len(venue1_set & venue2_set) / len(venue1_set | venue2_set)
                add_scores.append(add_score)
            venue_max_add_score = np.max(add_scores)
            venue_mean_add_score = np.mean(add_scores)
            venue_score_add_score = venue_max_score + venue_max_add_score
        # keyword score
        pairs = [(a, b) for a in keywords if a != '' for b in keywords_list if b != '']
        if len(pairs) == 0:
            keyword_max_score = -999
            keyword_mean_score = -999
        else:
            scores = [distance.get_jaro_distance(pair[0], pair[1]) for pair in pairs]
            keyword_max_score = np.max(scores)
            keyword_mean_score = np.mean(scores)
    return [
        year_diff, coauthor_count, coauthor_count_by1, coauthor_count_by2,
        org_max_score, org_mean_score, org_max_add_score, org_mean_add_score,
        org_score_add_score, venue_max_score, venue_mean_score, venue_max_add_score,
        venue_mean_add_score, venue_score_add_score, keyword_max_score, keyword_mean_score
    ]


class MLP_2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 1)
        # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.drop1(x)
        x = torch.sigmoid(self.fc2(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = torch.sigmoid(self.fc4(x))
        return x


class MLP_3(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 256)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = torch.sigmoid(self.fc3(x))
        return x


class SK_MLP:
    def __init__(self, in_dim, layer=2):
        super().__init__()
        if layer == 2:
            self.model = MLP_2(in_dim)
        elif layer == 3:
            self.model = MLP_3(in_dim)
        else:
            raise ValueError("Don't implement layer %s" % layer)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.device_cpu = torch.device('cpu')

    def fit(self, x, y, epochs=3000, lr=0.1, eval_set=None, batch_size=64, verbose=True):
        x, y = torch.tensor(x).to(self.device).to(torch.float), torch.tensor(y).to(self.device).to(torch.float)
        # dataset = TensorDataset(x, y)
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if eval_set is not None:
            eval_x = torch.tensor(eval_set[0]).to(self.device).to(torch.float)
            eval_y = torch.tensor(eval_set[1]).to(self.device).to(torch.float)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        train_loss, eval_loss = [], []

        for epoch in range(epochs):
            self.model.train()
            # tmp_train_loss, tmp_val_loss = [], []
            # for train_x, train_y in dataloader:
            optimizer.zero_grad()
            output = self.model(x).squeeze()
            # print(output)
            # print(output.shape)
            # print(train_y.shape)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # tmp_train_loss.append(loss.item())
            # train_loss.append(np.mean(tmp_train_loss))

            if eval_set is not None:
                self.model.eval()
                with torch.no_grad():
                    output = self.model(eval_x)
                    loss = criterion(output, eval_y)
                    eval_loss.append(loss.item())
                if verbose:
                    print('Epoch: %d/%d, Train loss: %f, Eval loss: %f' % (epoch + 1, epochs, train_loss[-1], eval_loss[-1]))
            else:
                if verbose:
                    print('Epoch: %d/%d, Train loss: %f' % (epoch+1, epochs, train_loss[-1]))
        x.to(self.device_cpu).detach()
        y.to(self.device_cpu).detach()

    def predict_proba(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(x).to(self.device).to(torch.float)
            result = self.model(x)
        result = result.cpu().numpy()
        return np.concatenate((1-result, result), axis=1)

    def predict(self, x):
        proba = self.predict_proba(x)[:, 1]
        prediction = (proba > 0.5).astype(np.float)
        return prediction
