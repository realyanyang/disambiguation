#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   final_subpipe.py
@Time    :   2019/11/28 20:05:43
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
from collections import defaultdict
import numpy as np
import tqdm
import torch
from triplet_model import TripletModel
import pandas as pd
from utils import load_json, load_pickle, save_json, save_pickle, clean_name, get_name_index, SK_MLP
import time
from utils import get_coauthor_v2, get_year_diff_v2, get_venue_score_v2, get_org_score_v2, get_key_word_num_v2, get_relative_year_feature
from utils import get_org_with_set_score, get_venue_with_set_score, get_keywords_with_set_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import TextToVec
from multiprocessing import Pool
import math
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
sns.set()


TEST_PUB_PATH = './final_dir/data/cna_test_pub.json'
TEST_UNASS_PATH = './final_dir/data/cna_test_unass_competition.json'
TEST_FEATURE_DIR = './final_dir/feature'
FINAL_DIR = './final_dir'
RESULT_SAVE_DIR = './final_dir/save'

NEW_DATA_DIR = './new-data'          # original info, for test
STACK_MODEL_DIR_v2 = './stack_model_v2'
RANDOM_SEED = 1129
np.random.seed(RANDOM_SEED)

os.makedirs(TEST_FEATURE_DIR, exist_ok=True)
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

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
np.random.seed(RANDOM_SEED)


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


def gen_base_feature(index, multi_size):
    # process test data and save in pickle
    # testdatafeatures --> {pid-with-index: {candidate-aids: [...], data: [[xxx], [xxx], [xxx]...]}}
    test_unass = load_json(TEST_UNASS_PATH)
    test_pub = load_json(TEST_PUB_PATH)
    # whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    aid2yearinfo = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2yearinfo.pkl'))
    aid2coauthor = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2coauthor.pkl'))
    aid2venue = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2venue.pkl'))
    aid2keywords = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2keywords.pkl'))
    aid2year = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2year.pkl'))
    aid2orgwithyear = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2orgwithyear.pkl'))
    name2aids = load_pickle(os.path.join(NEW_DATA_DIR, 'name2aids.pkl'))
    # aid2pids = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2pids.pkl'))
    aid2orgset = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2orgset.pkl'))
    aid2venueset = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2venueset.pkl'))
    aid2keywordsset = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2keywordsset.pkl'))

    name_map = load_json(os.path.join(FINAL_DIR, 'name.different.modified.json'))
    original_name = [pair[0] for pair in name_map]
    changed_name = [pair[1] for pair in name_map]
    name_map2 = load_json(os.path.join(FINAL_DIR, 'name.different.2.modified.json'))
    original_name2 = [pair[0] for pair in name_map2]
    changed_name2 = [pair[1] for pair in name_map2]

    single_range = math.ceil(len(test_unass) / multi_size)
    start = index * single_range
    end = (index + 1) * single_range if (index + 1) * single_range < len(test_unass) else len(test_unass)

    testdatafeatures = {}
    all_authors_name = list(name2aids.keys())
    print('Gen test features ...')
    for pid_with_index in tqdm.tqdm(test_unass[start:end]):
        inner_dict = {}
        now_pid, index = pid_with_index.split('-')
        author_name = test_pub[now_pid]['authors'][int(index)]['name']
        author_name = clean_name(author_name)
        if pid_with_index == 'ToCcabLT-1':
            author_name = 'junliang_wang'
        if pid_with_index == 'cVvvcFzj-1':
            author_name = 'xiaojun_liu'

        if author_name in original_name2:
            name_index = original_name2.index(author_name)
            author_name = changed_name2[name_index]
        elif author_name in original_name:
            name_index = original_name.index(author_name)
            author_name = changed_name[name_index]
        else:
            index = get_name_index(author_name, all_authors_name)
            author_name = all_authors_name[index]

        if isinstance(author_name, str):
            candidate_aids = name2aids[author_name]
        elif isinstance(author_name, list):
            candidate_aids = []
            for name in author_name:
                candidate_aids.extend(name2aids[name].tolist())
            candidate_aids = np.array(candidate_aids)
        else:
            raise ValueError("check author name ! ! !")

        inner_dict['candidate-aids'] = candidate_aids
        data = []
        for aid in candidate_aids:
            new_pair = (aid, pid_with_index)
            pid_info_dict = test_pub[now_pid]
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
        data = np.array(data)
        inner_dict['data'] = data
        testdatafeatures[pid_with_index] = inner_dict
    # save_pickle(testdatafeatures, os.path.join(TEST_FEATURE_DIR, 'u6uRzaff-5.pkl'))
    return testdatafeatures


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


def gen_title_feature():
    aid2titlevec = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2titlevec.pkl'))

    test_unass = load_json(TEST_UNASS_PATH)
    test_pub = load_json(TEST_PUB_PATH)
    name2aids = load_pickle(os.path.join(NEW_DATA_DIR, 'name2aids.pkl'))
    texttovec = TextToVec()

    name_map = load_json(os.path.join(FINAL_DIR, 'name.different.modified.json'))
    original_name = [pair[0] for pair in name_map]
    changed_name = [pair[1] for pair in name_map]
    name_map2 = load_json(os.path.join(FINAL_DIR, 'name.different.2.modified.json'))
    original_name2 = [pair[0] for pair in name_map2]
    changed_name2 = [pair[1] for pair in name_map2]

    all_authors_name = list(name2aids.keys())
    # test_title_feature --> {pid-with-index: {candidate-aids: [...], data: [(emb0, meb1), ...]}}
    test_title_feature = {}
    print('Gen title emb pair ...')
    for pid_with_index in tqdm.tqdm(test_unass):
        inner_dict = {}
        now_pid, index = pid_with_index.split('-')
        author_name = test_pub[now_pid]['authors'][int(index)]['name']
        author_name = clean_name(author_name)
        if pid_with_index == 'ToCcabLT-1':
            author_name = 'junliang_wang'
        if pid_with_index == 'cVvvcFzj-1':
            author_name = 'xiaojun_liu'

        if author_name in original_name2:
            name_index = original_name2.index(author_name)
            author_name = changed_name2[name_index]
        elif author_name in original_name:
            name_index = original_name.index(author_name)
            author_name = changed_name[name_index]
        else:
            index = get_name_index(author_name, all_authors_name)
            author_name = all_authors_name[index]

        if isinstance(author_name, str):
            candidate_aids = name2aids[author_name]
        elif isinstance(author_name, list):
            candidate_aids = []
            for name in author_name:
                candidate_aids.extend(name2aids[name].tolist())
            candidate_aids = np.array(candidate_aids)
        else:
            raise ValueError("check author name !!!")

        inner_dict['candidate-aids'] = candidate_aids
        info = test_pub[now_pid].get('title')
        if info is None:
            emb = np.zeros(300)
        else:
            emb = texttovec.get_vec(info)
        data = []
        for aid in candidate_aids:
            emb_pair = (aid2titlevec[aid], emb)
            data.append(emb_pair)
        inner_dict['data'] = data
        test_title_feature[pid_with_index] = inner_dict
    save_pickle(test_title_feature, os.path.join(TEST_FEATURE_DIR, 'test-title-emb-pair-name-clean-2.pkl'))

    print('Gen title distance ...')
    test_title_emb_pair = load_pickle(os.path.join(TEST_FEATURE_DIR, 'test-title-emb-pair-name-clean-2.pkl'))
    test_unass = load_json(TEST_UNASS_PATH)
    title_emb_pair = []
    for pid_with_index in tqdm.tqdm(test_unass):
        for pair in test_title_emb_pair[pid_with_index]['data']:
            title_emb_pair.append(pair)
    emb_pair_to_distance(
        'tm.title.1.checkpoint.pth', 'title', title_emb_pair,
        os.path.join(TEST_FEATURE_DIR, 'test-title-distance-df-name-clean-2.pkl')
    )


def predict(models):
    test_unass = load_json(TEST_UNASS_PATH)
    testdatafeatures = load_pickle(os.path.join(TEST_FEATURE_DIR, 'testdatafeatures-withsetinfo.pkl'))
    title_feature_df = pd.read_pickle(os.path.join(TEST_FEATURE_DIR, 'test-title-distance-df.pkl'))
    title_feature = title_feature_df.values

    models_loaded = []
    for model_info in models:
        model = {
            'model': load_pickle(model_info['model']),
            'ss': load_pickle(model_info['ss']),
            'cols': model_info['cols'],
            'score': model_info['score']
        }
        models_loaded.append(model)

    scores = [model_info['score'] for model_info in models_loaded]
    weights = [score / sum(scores) for score in scores]
    weights = np.array(weights).reshape(1, len(models_loaded))
    print(weights)

    submission = defaultdict(list)
    for pid_with_index in tqdm.tqdm(test_unass):
        candidate_aids = testdatafeatures[pid_with_index]['candidate-aids']
        data = testdatafeatures[pid_with_index]['data']
        data_length = len(candidate_aids)
        title_data = title_feature[:data_length]
        title_feature = title_feature[data_length:]
        data = np.concatenate((data, title_data), axis=1)
        default_cols = BASE_COLS + SET_INFO_COLS + TITLE_COLS
        df = pd.DataFrame(data=data, columns=default_cols)

        inner_data = np.zeros((len(candidate_aids), len(models_loaded)))
        for num, model_info in enumerate(models_loaded):
            model = model_info['model']
            ss = model_info['ss']
            data = df[model_info['cols']].values
            data = ss.transform(data)
            output = model.predict_proba(data)
            inner_data[:, num] = output

        final_output = np.sum((inner_data * weights), axis=1)
        predict_author = candidate_aids[np.argmax(final_output)]
        submission[predict_author].append(pid_with_index.split('-')[0])
    save_json(submission, os.path.join(FINAL_DIR, 'result-top3models.json'))


def see_year_distribution():
    # test_pub = load_json(TEST_PUB_PATH)
    # # test_pub = load_json('./data2/cna_data/whole_author_profile_pub.json')
    # year_count = []
    # for pid in test_pub:
    #     year = test_pub[pid].get('year', '0')
    #     if year == '':
    #         year = 0
    #     else:
    #         year = int(year)
    #     if year <= 1500 or year >= 2100:
    #         year = 0
    #     if year != 0:
    #         year_count.append(year)
    # df = pd.DataFrame(data=year_count, columns=['year'])
    # plt.figure(figsize=(10, 5))
    # sns.countplot(x='year', data=df)
    # plt.xticks(rotation='vertical')
    # plt.savefig(os.path.join(FINAL_DIR, 'test.year.png'))

    diff_year = []
    df_dict = load_pickle('./final_dir/feature/testdatafeatures-withsetinfo.pkl')
    for pid_with_index in df_dict:
        data = df_dict[pid_with_index]['data']
        default_cols = BASE_COLS + SET_INFO_COLS
        df = pd.DataFrame(data=data, columns=default_cols)
        diff_year.extend(df['max_diff'].values.tolist())
    df = pd.DataFrame(data=diff_year, columns=['diff_year'])
    plt.figure(figsize=(15, 5))
    sns.countplot(x='diff_year', data=df)
    plt.xticks(rotation='vertical')
    plt.savefig(os.path.join(FINAL_DIR, 'test.max.diff.year.png'))


def multi_gen_base_feature(multi_size):
    result = []
    p = Pool(multi_size)
    for index in range(multi_size):
        result.append(p.apply_async(gen_base_feature, args=(index, multi_size)))
        print('Process %d start' % index)
    p.close()
    p.join()
    testdatafeatures = {}
    for sub_dict in result:
        testdatafeatures.update(sub_dict.get())
    save_pickle(testdatafeatures, os.path.join(TEST_FEATURE_DIR, 'testdatafeatures-withsetinfo-name-clean-2.pkl'))


def save_time(model):
    test_unass = load_json(TEST_UNASS_PATH)
    testdatafeatures = load_pickle(os.path.join(TEST_FEATURE_DIR, 'testdatafeatures-withsetinfo-name-clean-2.pkl'))
    title_feature_df = pd.read_pickle(os.path.join(TEST_FEATURE_DIR, 'test-title-distance-df-name-clean-2.pkl'))
    title_feature = title_feature_df.values

    models_loaded = {
        'model': load_pickle(model['model']),
        'ss': load_pickle(model['ss']),
        'cols': model['cols'],
        'score': model['score'],
        'name': model['name'],
    }
    print(models_loaded['name'])

    model_result = {}
    for pid_with_index in tqdm.tqdm(test_unass):
        inner_dict = {}
        candidate_aids = testdatafeatures[pid_with_index]['candidate-aids']
        data = testdatafeatures[pid_with_index]['data']
        data_length = len(candidate_aids)
        title_data = title_feature[:data_length]
        title_feature = title_feature[data_length:]
        data = np.concatenate((data, title_data), axis=1)
        default_cols = BASE_COLS + SET_INFO_COLS + TITLE_COLS
        df = pd.DataFrame(data=data, columns=default_cols)

        model = models_loaded['model']
        ss = models_loaded['ss']
        data = df[models_loaded['cols']].values
        data = ss.transform(data)
        output = model.predict_proba(data)
        inner_dict['candidate-aids'] = candidate_aids
        inner_dict['result-score'] = output
        model_result[pid_with_index] = inner_dict
    save_pickle(model_result, os.path.join(RESULT_SAVE_DIR, 'name.clean.2.%s.result.score.pkl' % models_loaded['name']))


def get_coauthor_count_for_enhence(aid_pid_pair, aid_author_info_dict, pid_info_dict):
    index = int(aid_pid_pair[1].split('-')[1])
    authors = pid_info_dict['authors']
    authors = [clean_name(item['name']) for item in authors]
    authors.pop(index)
    count = 0
    for author_name in authors:
        if author_name in aid_author_info_dict.keys():
            count += 1
    return count


def get_org_score_for_enhence(aid_pid_pair, aid_org_year_list, pid_info_dict, my_stopwords, num_pattern, remove_punctuation, lemmatizer):
    def clean_text(str_info):
        str_lower = str_info.lower().strip()
        result = str_lower.translate(remove_punctuation)
        result = num_pattern.sub('', result)
        tokens = word_tokenize(result)
        result = [word for word in tokens if word not in my_stopwords]
        result = [lemmatizer.lemmatize(word) for word in result]
        result = [word for word in result if len(word) > 1]
        return ' '.join(list(set(result)))

    def get_score(corpus):
        vectorizer = CountVectorizer()
        org_vectors = vectorizer.fit_transform(corpus)
        score = cosine_similarity(org_vectors[0], org_vectors[1])
        return score.item()

    index = int(aid_pid_pair[1].split('-')[1])
    author = pid_info_dict['authors'][index]
    org = author.get('org', '').lower()
    if org == '':
        return 0
    aid_org = [pair[0] for pair in aid_org_year_list if pair[0] != '']
    if aid_org == []:
        return 0

    aid_org_clean = list(map(clean_text, aid_org))
    org_clean = clean_text(org)
    corpus_list = [[org_clean, target_org] for target_org in aid_org_clean]
    scores = list(map(get_score, corpus_list))
    return np.max(scores)


def get_mean_final_score(model_result_paths):
    print('Mean score begin ...')
    test_unass = load_json(TEST_UNASS_PATH)
    # aid2coauthor = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2coauthor.pkl'))
    # test_pub = load_json(TEST_PUB_PATH)
    # aid2orgwithyear = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2orgwithyear.pkl'))
    # title_feature_df = pd.read_pickle(os.path.join(TEST_FEATURE_DIR, 'test-title-distance-df.pkl'))
    # title_feature = title_feature_df.values

    # org_text_process_dict = {
    #     'my_stopwords': set(stopwords.words('english')),
    #     'num_pattern': re.compile(r'\d+'),
    #     'remove_punctuation': str.maketrans(string.punctuation, ' '*len(string.punctuation)),
    #     'lemmatizer': WordNetLemmatizer(),
    # }
    result_dict_list = [load_pickle(path) for path in model_result_paths]
    submission = defaultdict(list)
    # count = 0
    # problem_pids = []
    for pid_with_index in tqdm.tqdm(test_unass):
        candidate_aids = result_dict_list[0][pid_with_index]['candidate-aids']
        inner_data = np.zeros((len(candidate_aids), len(result_dict_list)))
        for num, result_dict in enumerate(result_dict_list):
            data = result_dict[pid_with_index]['result-score']
            inner_data[:, num] = data
        final_output = np.mean(inner_data, axis=1)
        predict_author = candidate_aids[np.argmax(final_output)]
        submission[predict_author].append(pid_with_index.split('-')[0])
    save_json(submission, os.path.join(FINAL_DIR, 'name-clean-2-mean-result-%d.json' % len(result_dict_list)))
    #     if np.max(final_output) < 0.5:
    #         print('-'*50)
    #         print(pid_with_index)

    #     if np.max(final_output) < 0.5:
    #         count += 1
    #         problem_pids.append(pid_with_index)
    #         coauthor_info = []
    #         for aid in candidate_aids:
    #             coauthor_info.append(get_coauthor_count_for_enhence((aid, pid_with_index), aid2coauthor[aid], test_pub[pid_with_index.split('-')[0]]))
    #         coauthor_info_array = np.array(coauthor_info)
    #         if np.max(coauthor_info_array) > 0:
    #             predict_author = candidate_aids[np.argmax(coauthor_info_array)]
    #         else:
    #             org_info = []
    #             for aid in candidate_aids:
    #                 org_info.append(get_org_score_for_enhence((aid, pid_with_index), aid2orgwithyear[aid], test_pub[pid_with_index.split('-')[0]], **org_text_process_dict))
    #             org_info_array = np.array(org_info)
    #             if np.max(org_info_array) > 0.5:
    #                 predict_author = candidate_aids[np.argmax(org_info_array)]
    #             else:
    #                 predict_author = candidate_aids[np.argmax(final_output)]
    #     else:
    #         predict_author = candidate_aids[np.argmax(final_output)]
    #     submission[predict_author].append(pid_with_index.split('-')[0])
    # save_json(problem_pids, os.path.join(FINAL_DIR, 'problem.pids.3.json'))
    # save_json(submission, os.path.join(FINAL_DIR, 'name-clean-2-enhance-mean-result-%d.json' % len(result_dict_list)))


def check_name():
    problem_pids = load_json(os.path.join(FINAL_DIR, 'problem.pids.3.json'))
    name2aids = load_pickle(os.path.join(NEW_DATA_DIR, 'name2aids.pkl'))
    test_pub = load_json(TEST_PUB_PATH)
    all_authors_name = list(name2aids.keys())

    name_map = []
    for pid_with_index in tqdm.tqdm(problem_pids):
        now_pid, index = pid_with_index.split('-')
        author_name_no_clean = test_pub[now_pid]['authors'][int(index)]['name']
        author_name = clean_name(author_name_no_clean)
        if pid_with_index == 'ToCcabLT-1':
            author_name = 'junliang_wang'
        if pid_with_index == 'cVvvcFzj-1':
            author_name = 'xiaojun_liu'

        index = get_name_index(author_name, all_authors_name)
        author_name_inlist = all_authors_name[index]
        # if author_name_inlist != author_name:
        name_map.append((pid_with_index, author_name_no_clean, author_name, author_name_inlist))
    name_map = list(set(name_map))
    print(len(name_map))
    save_json(name_map, os.path.join(FINAL_DIR, 'name.different.3.json'))


def find_name(name):
    name2aids = load_pickle(os.path.join(NEW_DATA_DIR, 'name2aids.pkl'))
    aid2orgset = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2orgset.pkl'))
    all_authors_name = list(name2aids.keys())
    for name_in_list in all_authors_name:
        if name in name_in_list:
            print('-'*50)
            print(name_in_list)
            for aid in name2aids[name_in_list]:
                # print(aid2orgset[aid])
                if 'beijing' in aid2orgset[aid]:
                    print(aid2orgset[aid])


if __name__ == "__main__":
    models = [
        {   # 0.85926333738039 original
            'model': os.path.join(STACK_MODEL_DIR_v2, 'sm-191125-nosetinfo-extend3-sample11.pkl'),
            'ss': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-nosetinfo-extend3-sample11.pkl'),
            'cols': BASE_COLS,
            'score': 0.85926333738039,
            'name': 'sm-191125-nosetinfo-extend3-sample11.pkl',
        },
        {   # 0.858031834386063 with set info
            'model': os.path.join(STACK_MODEL_DIR_v2, 'test-2-sm-191127-withsetinfo-sample11.pkl'),
            'ss': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-withsetinfo-sample11.pkl'),
            'cols': BASE_COLS + SET_INFO_COLS,
            'score': 0.858031834386063,
            'name': 'test-2-sm-191127-withsetinfo-sample11.pkl',
        },
        {   # 0.856180351089599 with set info
            'model': os.path.join(STACK_MODEL_DIR_v2, 'sm-191127-withsetinfo-11.pkl'),
            'ss': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-withsetinfo-11.pkl'),
            'cols': BASE_COLS + SET_INFO_COLS,
            'score': 0.856180351089599,
            'name': 'sm-191127-withsetinfo-11.pkl',
        },
        {   # 0.855763586778158 with set info and title info
            'model': os.path.join(STACK_MODEL_DIR_v2, 'sm-191128-withsetinfo-title-11-norf.pkl'),
            'ss': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-withsetinfo-title-11.pkl'),
            'cols': BASE_COLS + SET_INFO_COLS + TITLE_COLS,
            'score': 0.855763586778158,
            'name': 'sm-191128-withsetinfo-title-11-norf.pkl',
        },
        {   # 0.85364791527539
            'model': os.path.join(STACK_MODEL_DIR_v2, 'sm-191126-withsetinfo-sample11.pkl'),
            'ss': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-withsetinfo-sample11.pkl'),
            'cols': BASE_COLS + SET_INFO_COLS,
            'score': 0.85364791527539,
            'name': 'sm-191126-withsetinfo-sample11.pkl',
        },
        {   # 0.855538436984147
            'model': os.path.join(STACK_MODEL_DIR_v2, 'sm-191128-withsetinfo-title-11.pkl'),
            'ss': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-withsetinfo-title-11.pkl'),
            'cols': BASE_COLS + SET_INFO_COLS + TITLE_COLS,
            'score': 0.855538436984147,
            'name': 'sm-191128-withsetinfo-title-11.pkl',
        },
    ]
    # see_year_distribution()

    process_size = 22
    multi_gen_base_feature(process_size)

    gen_title_feature()

    for i in range(len(models)):   # 建议手动多进程，节省时间
        save_time(models[i])       # 无法代码直接多进程，会与模型的多进程冲突
    # save_time(models[5])

    model_result_paths = [
        os.path.join(RESULT_SAVE_DIR, 'name.clean.2.sm-191125-nosetinfo-extend3-sample11.pkl.result.score.pkl'),
        os.path.join(RESULT_SAVE_DIR, 'name.clean.2.sm-191126-withsetinfo-sample11.pkl.result.score.pkl'),
        os.path.join(RESULT_SAVE_DIR, 'name.clean.2.sm-191127-withsetinfo-11.pkl.result.score.pkl'),
        os.path.join(RESULT_SAVE_DIR, 'name.clean.2.sm-191128-withsetinfo-title-11-norf.pkl.result.score.pkl'),
        os.path.join(RESULT_SAVE_DIR, 'name.clean.2.sm-191128-withsetinfo-title-11.pkl.result.score.pkl'),
        os.path.join(RESULT_SAVE_DIR, 'name.clean.2.test-2-sm-191127-withsetinfo-sample11.pkl.result.score.pkl'),
    ]
    get_mean_final_score(model_result_paths)
    # check_name()
