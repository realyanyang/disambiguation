#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   train_model.py
@Time    :   2019/12/05 13:49:58
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
from channel2_v2 import *
import os
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


add_text_feature_for_train()
create_feature()
models = [
    {   # 0.85926333738039 original best
        'model_path': os.path.join(STACK_MODEL_DIR_v2, 'sm-191125-nosetinfo-extend3-sample11.pkl'),
        'ss_path': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-nosetinfo-extend3-sample11.pkl'),
        'cols': BASE_COLS,
        'score': 0.85926333738039,
        'name': 'sm-191125-nosetinfo-extend3-sample11.pkl',
        'model': [
            [
                CatBoostClassifier(
                    iterations=180, learning_rate=0.1, depth=7, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                CatBoostClassifier(
                    iterations=500, learning_rate=0.1, depth=4, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=7, learning_rate=0.05, n_estimators=180, subsample=0.8,
                    n_jobs=-1, min_child_weight=6, random_state=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=4, learning_rate=0.05, n_estimators=350, subsample=0.8,
                    n_jobs=-1, min_child_weight=6, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=7, learning_rate=0.01, n_estimators=800, objective='binary',
                    subsample=0.8, n_jobs=23, num_leaves=82, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=4, learning_rate=0.01, n_estimators=2000, objective='binary',
                    subsample=0.8, n_jobs=23, num_leaves=12, random_state=RANDOM_SEED
                ),
                RandomForestClassifier(
                    n_estimators=1000, max_depth=35, n_jobs=-1, verbose=0, random_state=RANDOM_SEED
                ),
            ],
            [
                CatBoostClassifier(
                    iterations=150, learning_rate=0.1, depth=2, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
            ],
        ],
        'model_param': [
            [
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {},
            ],
            [
                {'verbose': False},
            ],
        ],
    },
    {   # 0.858031834386063 with set info
        'model_path': os.path.join(STACK_MODEL_DIR_v2, 'test-2-sm-191127-withsetinfo-sample11.pkl'),
        'ss_path': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-withsetinfo-sample11.pkl'),
        'cols': BASE_COLS + SET_INFO_COLS,
        'score': 0.858031834386063,
        'name': 'test-2-sm-191127-withsetinfo-sample11.pkl',
        'model': [
            [
                CatBoostClassifier(
                    iterations=400, learning_rate=0.05, depth=7, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                CatBoostClassifier(
                    iterations=1000, learning_rate=0.05, depth=4, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=7, learning_rate=0.05, n_estimators=180, subsample=0.8,
                    n_jobs=-1, min_child_weight=4, random_state=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=4, learning_rate=0.03, n_estimators=500, subsample=0.8,
                    n_jobs=-1, min_child_weight=6, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=7, learning_rate=0.01, n_estimators=1000, objective='binary',
                    subsample=0.8, n_jobs=23, num_leaves=35, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=4, learning_rate=0.01, n_estimators=3500, objective='binary',
                    subsample=0.8, n_jobs=23, num_leaves=5, random_state=RANDOM_SEED
                ),
                RandomForestClassifier(
                    n_estimators=1000, max_depth=35, n_jobs=-1, verbose=0, random_state=RANDOM_SEED
                ),
            ],
            [
                CatBoostClassifier(
                    iterations=800, learning_rate=0.01, depth=3, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
            ],
        ],
        'model_param': [
            [
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {},
            ],
            [
                {'verbose': False},
            ],
        ],
    },
    {   # 0.856180351089599 with set info
        'model_path': os.path.join(STACK_MODEL_DIR_v2, 'sm-191127-withsetinfo-11.pkl'),
        'ss_path': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-withsetinfo-11.pkl'),
        'cols': BASE_COLS + SET_INFO_COLS,
        'score': 0.856180351089599,
        'name': 'sm-191127-withsetinfo-11.pkl',
        'model': [
            [
                CatBoostClassifier(
                    iterations=400, learning_rate=0.05, depth=7, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                CatBoostClassifier(
                    iterations=1000, learning_rate=0.05, depth=4, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=7, learning_rate=0.05, n_estimators=180, subsample=0.8,
                    n_jobs=-1, min_child_weight=4, random_state=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=4, learning_rate=0.03, n_estimators=500, subsample=0.8,
                    n_jobs=-1, min_child_weight=6, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=7, learning_rate=0.01, n_estimators=1000, objective='binary',
                    subsample=0.8, n_jobs=23, num_leaves=35, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=4, learning_rate=0.01, n_estimators=3500, objective='binary',
                    subsample=0.8, n_jobs=23, num_leaves=5, random_state=RANDOM_SEED
                ),
                RandomForestClassifier(
                    n_estimators=1000, max_depth=35, n_jobs=-1, verbose=0, random_state=RANDOM_SEED
                ),
            ],
            [
                CatBoostClassifier(
                    iterations=800, learning_rate=0.01, depth=3, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
            ],
        ],
        'model_param': [
            [
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {},
            ],
            [
                {'verbose': False},
            ],
        ],
    },
    {   # 0.855763586778158 with set info and title info
        'model_path': os.path.join(STACK_MODEL_DIR_v2, 'sm-191128-withsetinfo-title-11-norf.pkl'),
        'ss_path': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-withsetinfo-title-11.pkl'),
        'cols': BASE_COLS + SET_INFO_COLS + TITLE_COLS,
        'score': 0.855763586778158,
        'name': 'sm-191128-withsetinfo-title-11-norf.pkl',
        'model': [
            [
                CatBoostClassifier(
                    iterations=320, learning_rate=0.05, depth=7, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                CatBoostClassifier(
                    iterations=900, learning_rate=0.05, depth=4, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=7, learning_rate=0.05, n_estimators=180, subsample=0.8,
                    n_jobs=-1, min_child_weight=6, random_state=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=4, learning_rate=0.03, n_estimators=500, subsample=0.8,
                    n_jobs=-1, min_child_weight=6, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=7, learning_rate=0.01, n_estimators=1000, objective='binary',
                    subsample=0.8, n_jobs=-1, num_leaves=82, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=4, learning_rate=0.01, n_estimators=3500, objective='binary',
                    subsample=0.8, n_jobs=-1, num_leaves=5, random_state=RANDOM_SEED
                ),
            ],
            [
                CatBoostClassifier(
                    iterations=1200, learning_rate=0.01, depth=2, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
            ],
        ],
        'model_param': [
            [
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
            ],
            [
                {'verbose': False},
            ],
        ],
    },
    {   # 0.85364791527539
        'model_path': os.path.join(STACK_MODEL_DIR_v2, 'sm-191126-withsetinfo-sample11.pkl'),
        'ss_path': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-withsetinfo-sample11.pkl'),
        'cols': BASE_COLS + SET_INFO_COLS,
        'score': 0.85364791527539,
        'name': 'sm-191126-withsetinfo-sample11.pkl',
        'model': [
            [
                CatBoostClassifier(
                    iterations=180, learning_rate=0.1, depth=7, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                CatBoostClassifier(
                    iterations=500, learning_rate=0.1, depth=4, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=7, learning_rate=0.05, n_estimators=180, subsample=0.8,
                    n_jobs=-1, min_child_weight=6, random_state=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=4, learning_rate=0.05, n_estimators=350, subsample=0.8,
                    n_jobs=-1, min_child_weight=6, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=7, learning_rate=0.01, n_estimators=800, objective='binary',
                    subsample=0.8, n_jobs=-1, num_leaves=82, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=4, learning_rate=0.01, n_estimators=2000, objective='binary',
                    subsample=0.8, n_jobs=-1, num_leaves=12, random_state=RANDOM_SEED
                ),
                RandomForestClassifier(
                    n_estimators=1000, max_depth=35, n_jobs=-1, verbose=0, random_state=RANDOM_SEED
                ),
            ],
            [
                CatBoostClassifier(
                    iterations=150, learning_rate=0.1, depth=2, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
            ],
        ],
        'model_param': [
            [
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {},
            ],
            [
                {'verbose': False},
            ],
        ],
    },
    {   # 0.855538436984147
        'model_path': os.path.join(STACK_MODEL_DIR_v2, 'sm-191128-withsetinfo-title-11.pkl'),
        'ss_path': os.path.join(STACK_MODEL_DIR_v2, 'standardscaler-last1year-withsetinfo-title-11.pkl'),
        'cols': BASE_COLS + SET_INFO_COLS + TITLE_COLS,
        'score': 0.855538436984147,
        'name': 'sm-191128-withsetinfo-title-11.pkl',
        'model': [
            [
                CatBoostClassifier(
                    iterations=320, learning_rate=0.05, depth=7, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                CatBoostClassifier(
                    iterations=900, learning_rate=0.05, depth=4, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=7, learning_rate=0.05, n_estimators=180, subsample=0.8,
                    n_jobs=-1, min_child_weight=6, random_state=RANDOM_SEED
                ),
                XGBClassifier(
                    max_depth=4, learning_rate=0.03, n_estimators=500, subsample=0.8,
                    n_jobs=-1, min_child_weight=6, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=7, learning_rate=0.01, n_estimators=1000, objective='binary',
                    subsample=0.8, n_jobs=-1, num_leaves=82, random_state=RANDOM_SEED
                ),
                LGBMClassifier(
                    max_depth=4, learning_rate=0.01, n_estimators=3500, objective='binary',
                    subsample=0.8, n_jobs=-1, num_leaves=5, random_state=RANDOM_SEED
                ),
                RandomForestClassifier(
                    n_estimators=1000, max_depth=60, n_jobs=-1, verbose=0, random_state=RANDOM_SEED
                ),
            ],
            [
                CatBoostClassifier(
                    iterations=1200, learning_rate=0.01, depth=2, loss_function='Logloss',
                    eval_metric='Logloss', task_type='GPU', random_seed=RANDOM_SEED
                ),
            ],
        ],
        'model_param': [
            [
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {'verbose': False},
                {},
            ],
            [
                {'verbose': False},
            ],
        ],
    },
]
for model_info in models:
    print('--'*50)
    train(model_info)
