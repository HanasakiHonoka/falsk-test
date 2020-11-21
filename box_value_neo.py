# encoding: utf-8
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('pre.csv')

df['year'] = df['year'] - 2000

Y0 = df['boxoffice']
Y1 = np.log1p(df['boxoffice'] * 10000)
Y2 = df['boxoffice'] * 10000
Y = Y0

top_genres = ['剧情', '爱情', '喜剧', '动作', '惊悚', '悬疑', '犯罪', '战争', '恐怖', '纪录片', '青春', '奇幻', '冒险', '儿童', '传记', '古装', '励志',
              '亲情', '科幻', '历史']
genre_cols = []
for g in top_genres:
    genre_cols.append("genre_" + g)

df_input = df.copy(deep=True)
df_input.drop(
    ['id', 'title', 'technology', 'type', 'boxoffice', 'actor', 'issue_company', 'manu_company', 'release_date',
     'type_list', 'technology_list'], axis=1, inplace=True)
df_input.drop(
    ['director_1_id', 'director_1_name', 'director_2_id', 'director_2_name', 'scenarist_1_id', 'scenarist_1_name',
     'scenarist_2_id', 'scenarist_2_name', 'actor_1_id', 'actor_1_name', 'actor_2_id', 'actor_2_name'], axis=1,
    inplace=True)
df_input.drop(genre_cols, axis=1, inplace=True)
df_input.drop(['duration'], axis=1, inplace=True)
df_input.drop(['issue_company_1_id', 'manu_company_1_id'], axis=1, inplace=True)

df_input.drop(['first_boxoffice', 'douban_rating', 'baidu_index'], axis=1, inplace=True)
# df_input.drop(['director_1_pr_boxrank','director_2_pr_boxrank','actor_1_pr_boxrank','actor_2_pr_boxrank','scenarist_1_pr_boxrank','scenarist_2_pr_boxrank'],axis=1,inplace=True)
df_input.drop(['director_1_baidu', 'director_2_baidu', 'actor_1_baidu', 'actor_2_baidu', 'scenarist_1_baidu',
               'scenarist_2_baidu'], axis=1, inplace=True)
df_input.drop(['actor_1_fans', 'actor_2_fans'], axis=1, inplace=True)
df_input.drop(['pr_movie', 'bt_movie', 'cl_movie', 'de_movie'], axis=1, inplace=True)

from scipy.stats import norm, skew

df_input.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

df_input.drop(['rating_numbers', 'comment_numbers', 'review_numbers'], axis=1, inplace=True)

person_cols = ['director_1_amount', 'director_1_boxoffice', 'director_2_amount', 'director_2_boxoffice', \
               'scenarist_1_amount', 'scenarist_1_boxoffice', 'scenarist_2_amount', 'scenarist_2_boxoffice', \
               'actor_1_amount', 'actor_1_boxoffice', 'actor_2_amount', 'actor_2_boxoffice']

neo4j_cols = ['pr_box_rank', 'director_1_pr_boxrank', 'director_2_pr_boxrank', 'actor_1_pr_boxrank',
              'actor_2_pr_boxrank', 'scenarist_1_pr_boxrank', 'scenarist_2_pr_boxrank']

df_advanced_input = df_input.copy(deep=True)
df_advanced_input.drop(['day', 'day_of_week', 'week', 'nationalday', 'newyear'], axis=1, inplace=True)

# df_advanced_input['duration'] = (df_advanced_input['duration'] - df_advanced_input['duration'].min()) / (df_advanced_input['duration'].max() - df_advanced_input['duration'].min())

df_advanced_input['has_2_director'] = df_advanced_input['director_2_amount'] > 0
df_advanced_input['has_2_scenarist'] = df_advanced_input['scenarist_2_amount'] > 0
df_advanced_input['director_boxoffice_average'] = df_advanced_input['director_1_boxoffice'] / df_advanced_input[
    'director_1_amount']
df_advanced_input['director_boxoffice_average'] = df_advanced_input['director_boxoffice_average'].fillna(0)
df_advanced_input['director_boxoffice_average'] = df_advanced_input['director_boxoffice_average'].astype(int)
df_advanced_input['scenarist_boxoffice_average'] = df_advanced_input['scenarist_1_boxoffice'] / df_advanced_input[
    'scenarist_1_amount']
df_advanced_input['scenarist_boxoffice_average'] = df_advanced_input['scenarist_boxoffice_average'].fillna(0)
df_advanced_input['scenarist_boxoffice_average'] = df_advanced_input['scenarist_boxoffice_average'].astype(int)
df_advanced_input['actor_1_boxoffice_average'] = df_advanced_input['actor_1_boxoffice'] / df_advanced_input[
    'actor_1_amount']
df_advanced_input['actor_1_boxoffice_average'] = df_advanced_input['actor_1_boxoffice_average'].fillna(0)
df_advanced_input['actor_1_boxoffice_average'] = df_advanced_input['actor_1_boxoffice_average'].astype(int)
df_advanced_input['actor_2_boxoffice_average'] = df_advanced_input['actor_2_boxoffice'] / df_advanced_input[
    'actor_2_amount']
df_advanced_input['actor_2_boxoffice_average'] = df_advanced_input['actor_2_boxoffice_average'].fillna(0)
df_advanced_input['actor_2_boxoffice_average'] = df_advanced_input['actor_2_boxoffice_average'].astype(int)
df_advanced_input['director_amount'] = df_advanced_input['director_1_amount']
df_advanced_input['scenarist_amount'] = df_advanced_input['scenarist_1_amount']
df_advanced_input['actor_amount'] = df_advanced_input['actor_1_amount'] + df_advanced_input['actor_2_amount']
df_advanced_input.drop(person_cols, axis=1, inplace=True)

df_neo4j_input = df_advanced_input.copy(deep=True)

from sklearn.metrics import mean_squared_error


def rmse(y, y_pred):
    if mean_squared_error(y, y_pred) > 0:
        return np.sqrt(mean_squared_error(y, y_pred))
    else:
        return np.sqrt(-mean_squared_error(y, y_pred))


from sklearn.model_selection import train_test_split

random_seed = 38

X_train_neo, X_val_neo, y_train_neo, y_val_neo = train_test_split(df_neo4j_input, Y, test_size=0.2)

LR = LinearRegression()

LR.fit(X_train_neo, y_train_neo)

print('训练集准确率：\n', LR.score(X_train_neo, y_train_neo))  # 分数
print('验证集准确率：\n', LR.score(X_val_neo, y_val_neo))

y_neo4j_pred_LR = (LR.predict(X_val_neo)).astype(int)

a = (y_val_neo.values).astype(int) / 1
b = y_neo4j_pred_LR / 1
rmse(a, b)

import xgboost as xgb

dtrain = xgb.DMatrix(X_train_neo, label=y_train_neo)
dtest = xgb.DMatrix(X_val_neo)

params = {'max_depth': 7,
          'eta': 1,
          'silent': 1,
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'learning_rate': 0.05
          }
num_rounds = 50

xb = xgb.train(params, dtrain, num_rounds)

y_neo4j_pred_xgb = (xb.predict(dtest)).astype(int)

a = (y_val_neo.values).astype(int) / 1
b = y_neo4j_pred_xgb / 1
rmse(a, b)

import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression', boosting='gbdt', metric='l2',
                              max_depth=3, num_leaves=8, max_bin = 21,
                              learning_rate=0.05, n_estimators=1900,
                              bagging_fraction = 0.7, bagging_freq = 8, bagging_seed=9,
                              feature_fraction = 0.9, feature_fraction_seed=9,
                              min_data_in_leaf =10, min_sum_hessian_in_leaf = 11,
                              num_iterations=10000,lambda_l2=0.00095)
# model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
#                               learning_rate=0.05, n_estimators=720,
#                               max_bin=55, bagging_fraction=0.8,
#                               bagging_freq=5, feature_fraction=0.2319,
#                               feature_fraction_seed=9, bagging_seed=9,
#                               min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

model_lgb.fit(X_train_neo, y_train_neo)
y_neo4j_pred_lgb = model_lgb.predict(X_val_neo)

a = (y_val_neo.values).astype(int) / 1
b = y_neo4j_pred_lgb / 1
rmse(a, b)

from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(max_depth=7)

gbm.fit(X_train_neo, y_train_neo)

print('训练集准确率：\n', gbm.score(X_train_neo, y_train_neo))  # 分数
print('验证集准确率：\n', gbm.score(X_val_neo, y_val_neo))

y_neo4j_pred_gbm = (gbm.predict(X_val_neo)).astype(int)

a = (y_val_neo.values).astype(int)
b = y_neo4j_pred_gbm
rmse(a, b)


def load_LRN_model():
    return LR


def load_XGBN_model():
    return xb


def load_LGBN_model():
    return model_lgb


def load_GBMN_model():
    return gbm
