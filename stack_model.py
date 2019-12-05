#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   stack_model.py
@Time    :   2019/11/18 16:20:56
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
import numpy as np
import copy
from sklearn.model_selection import KFold
# from utils import SK_MLP
from sklearn.metrics import log_loss
from utils import load_json, load_pickle, save_json, save_pickle
import time
RANDOM_SEED = 1129

np.random.seed(RANDOM_SEED)


class StackModel:
    def __init__(self, models, fit_params, folds=5, verbose=False):
        super().__init__()
        self.models = models
        self.params = fit_params
        assert len(self.models) == len(self.params)
        for model, param in zip(models, fit_params):
            assert len(model) == len(param)
        self.level_layers = len(models)
        self.folds = folds
        self.verbose = verbose
        self.kfold = KFold(n_splits=self.folds, random_state=RANDOM_SEED, shuffle=True)
        print('Total level: %d' % self.level_layers)

    def fit(self, x, y):
        begin_time = time.time()
        assert len(x) == len(y)
        for level_num, (level_model, level_params) in enumerate(zip(self.models, self.params)):
            level_begin_time = time.time()
            print('=='*10, 'level num:', level_num, '==' * 10)
            level_inner_predict = np.zeros((x.shape[0], len(level_model)))

            for model_num, (model, params) in enumerate(zip(level_model, level_params)):
                model_begin_time = time.time()
                print(model)
                print('--'*10, 'model num:', model_num, '--'*10)
                for train_index, val_index in self.kfold.split(x):
                    train_x, train_y = x[train_index], y[train_index]
                    val_x, val_y = x[val_index], y[val_index]
                    tmp_model = copy.copy(model)
                    tmp_model.fit(train_x, train_y, **params)
                    tmp_model_prediction = tmp_model.predict_proba(val_x)[:, 1]
                    print('Log loss', log_loss(val_y, tmp_model_prediction, eps=1e-7))

                    for num, index in enumerate(val_index):
                        level_inner_predict[index][model_num] = tmp_model_prediction[num]
                    del tmp_model
                model.fit(x, y, **params)
                print('Model train time: %f s' % (time.time() - model_begin_time))
            x = level_inner_predict
            print('Level train time: %f s' % (time.time() - level_begin_time))
            # to_save = np.concatenate((x, y.reshape(-1, 1)), axis=1)
            # save_pickle(to_save, './tmp/inner-%d.pkl' % level_num)
        print('Final log loss:', log_loss(y, x))
        print('Total train time: %f s' % (time.time() - begin_time))

    def predict_proba(self, x):
        begin_time = time.time()
        for level_num, level_model in enumerate(self.models):
            level_begin_time = time.time()
            if self.verbose:
                print('=='*10, 'level num:', level_num, '==' * 10)
            level_inner_predict = np.zeros((x.shape[0], len(level_model)))
            for model_num, model in enumerate(level_model):
                model_begin_time = time.time()
                if self.verbose:
                    print('--'*10, 'model num:', model_num, '--'*10)
                model_prediction = model.predict_proba(x)[:, 1]
                level_inner_predict[:, model_num] = model_prediction
                if self.verbose:
                    print('Model predict time: %f s' % (time.time() - model_begin_time))
            x = level_inner_predict
            if self.verbose:
                print('Level predict time: %f s' % (time.time() - level_begin_time))
        if self.verbose:
            print('Total predict time: %f s' % (time.time() - begin_time))
        return x.squeeze()

    def predict(self, x):
        result = self.predict_proba(x)
        result = (result > 0.5).astype(np.float)
        return result


if __name__ == "__main__":
    pass
    # models = [
    #     [
    #         # SK_MLP(3),
    #         # XGBClassifier(
    #         #     max_depth=7, learning_rate=0.1, n_estimators=1500, subsample=0.8,
    #         #     n_jobs=-1, min_child_weight=2, random_state=RANDOM_SEED
    #         # ),
    #         # XGBClassifier(
    #         #     max_depth=4, learning_rate=0.1, n_estimators=5000, subsample=0.8,
    #         #     n_jobs=-1, min_child_weight=2, random_state=RANDOM_SEED,
    #         # ),
    #         # CatBoostClassifier(
    #         #     iterations=4000, learning_rate=0.1, depth=7, loss_function='Logloss',
    #         #     eval_metric='Logloss', task_type='CPU', random_seed=RANDOM_SEED
    #         # ),
    #         # CatBoostClassifier(
    #         #     iterations=6000, learning_rate=0.1, depth=4, loss_function='Logloss',
    #         #     eval_metric='Logloss', task_type='CPU', random_seed=RANDOM_SEED
    #         # ),
    #         LGBMClassifier(
    #             max_depth=7, learning_rate=0.1, n_estimators=4000, objective='binary',
    #             subsample=0.8, n_jobs=-1, num_leaves=82
    #         ),
    #         LGBMClassifier(
    #             max_depth=4, learning_rate=0.1, n_estimators=6000, objective='binary',
    #             subsample=0.8, n_jobs=-1, num_leaves=12
    #         ),
    #     ],
    #     [
    #         # CatBoostClassifier(
    #         #     iterations=6000, learning_rate=0.1, depth=2, loss_function='Logloss',
    #         #     eval_metric='Logloss', task_type='CPU', random_seed=RANDOM_SEED
    #         # ),
    #         LGBMClassifier(
    #             max_depth=4, learning_rate=0.1, n_estimators=6000, objective='binary',
    #             subsample=0.8, n_jobs=-1, num_leaves=12
    #         ),
    #     ],
    # ]
    # params = [
    #     [
    #         # {'lr': 0.01, 'epochs': 100, 'verbose': False},
    #         # {'verbose': False},
    #         # {'verbose': False},
    #         # {'verbose': False},
    #         # {'verbose': False},
    #         {'verbose': False},
    #         {'verbose': False},
    #     ],
    #     [
    #         {'verbose': False},
    #     ],
    # ]

    # # train_x = np.random.randint(0, 10, (20, 3))
    # train_x = np.random.randn(5000, 3)
    # train_y = (np.sum(train_x, axis=1) % 2).astype(np.int)
    # print(train_x)
    # print(train_y)

    # # model = SK_MLP(3, layer=2)
    # # model.fit(train_x, train_y, epochs=300, verbose=True)
    # # save_pickle(model, './mlp.pkl')
    # # model = load_pickle('./mlp.pkl')
    # # y = model.predict(train_x)
    # # print(y)
    # # print((y > 0.5).cpu().numpy().astype(np.float))

    # # sm = StackModel(models, params)
    # # sm.fit(train_x, train_y)
    # # save_pickle(sm, './stack_model/sm.pkl')

    # # sm = load_pickle('./stack_model/sm.pkl')
    # # pre = sm.predict_proba(train_x)
