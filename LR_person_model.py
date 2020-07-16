# encoding: utf-8
# @FileName  :LR_person_model.py
# @Time      :2020/7/16 2:11
# @Author    :XZX
import numpy as np
import pandas as pd
from scipy.stats import norm, skew
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.set_option('display.width', 1200)

pd.set_option('display.max_rows', 4000)

pd.set_option('display.max_columns', 200)
df_train = pd.read_csv('person_premodel.csv')

df = df_train.copy(deep=True)
# df = df[df['year'] >= 2014]
df = df[df['boxoffice'] >= 100]

Y0 = df['boxoffice']
Y1 = np.log1p(df['boxoffice'] * 10000)
Y2 = df['boxoffice'] * 10000
Y = Y0

df_input = df.copy(deep=True)
df_input.drop(
    ['id', 'title', 'technology', 'type', 'boxoffice', 'first_boxoffice', 'actor', 'issue_company', 'manu_company',
     'release_date', 'type_list', 'technology_list'], axis=1, inplace=True)
df_input.drop(
    ['director_1_id', 'director_1_name', 'director_2_id', 'director_2_name', 'scenarist_1_id', 'scenarist_1_name',
     'scenarist_2_id', 'scenarist_2_name', 'actor_1_id', 'actor_1_name', 'actor_2_id', 'actor_2_name'], axis=1,
    inplace=True)

year_cols = ['year_2000', 'year_2001', 'year_2002', 'year_2003', 'year_2004', \
             'year_2005', 'year_2006', 'year_2007', 'year_2008', 'year_2009', \
             'year_2010', 'year_2011', 'year_2012', 'year_2013', 'year_2014', \
             'year_2015', 'year_2016', 'year_2017', 'year_2018', 'year_2019']

genre_cols = ['genre_剧情', 'genre_爱情', 'genre_喜剧', 'genre_动作', 'genre_惊悚', \
              'genre_动画', 'genre_悬疑', 'genre_冒险', 'genre_犯罪', 'genre_战争', \
              'genre_恐怖', 'genre_奇幻', 'genre_儿童', 'genre_纪录片', 'genre_青春']

person_cols = ['director_1_amount', 'director_1_boxoffice', 'director_2_amount', 'director_2_boxoffice', \
               'scenarist_1_amount', 'scenarist_1_boxoffice', 'scenarist_2_amount', 'scenarist_2_boxoffice', \
               'actor_1_amount', 'actor_1_boxoffice', 'actor_2_amount', 'actor_2_boxoffice']
df_baseline_input = df_input.copy(deep=True)
df_baseline_input.drop(year_cols, axis=1, inplace=True)
df_baseline_input.drop(['is_weekend', 'springfestival', 'nationalday', 'summer'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df_baseline_input, Y, test_size=0.3, random_state=78)

from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR.fit(X_train, y_train)

print('LR_person_model训练集准确率：\n', LR.score(X_train, y_train))  # 分数
print('LR_person_model验证集准确率：\n', LR.score(X_val, y_val))


def load_LR_person_model():
    return LR
