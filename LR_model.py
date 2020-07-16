# encoding: utf-8
import numpy as np
import pandas as pd
from scipy.stats import norm, skew
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('premodel.csv')

df = df_train.copy(deep=True)
df = df[df['boxoffice'] >= 100]

Y0 = df['boxoffice']
Y1 = np.log1p(df['boxoffice'] * 10000)
Y2 = df['boxoffice'] * 10000
Y = Y0

df_input = df.copy(deep=True)
df_input.drop(
    ['id', 'title', 'technology', 'type', 'boxoffice', 'first_boxoffice', 'actor', 'issue_company', 'manu_company',
     'release_date', 'type_list', 'technology_list'], axis=1, inplace=True)

df_input.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
year_cols = ['year_2000', 'year_2001', 'year_2002', 'year_2003', 'year_2004',
             'year_2005', 'year_2006', 'year_2007', 'year_2008', 'year_2009',
             'year_2010', 'year_2011', 'year_2012', 'year_2013', 'year_2014',
             'year_2015', 'year_2016', 'year_2017', 'year_2018', 'year_2019']

genre_cols = ['genre_剧情', 'genre_爱情', 'genre_喜剧', 'genre_动作', 'genre_惊悚',
              'genre_动画', 'genre_悬疑', 'genre_冒险', 'genre_犯罪', 'genre_战争',
              'genre_恐怖', 'genre_奇幻', 'genre_儿童', 'genre_纪录片', 'genre_青春']

df_baseline_input = df_input.copy(deep=True)
df_baseline_input.drop(year_cols, axis=1, inplace=True)
df_baseline_input.drop(['is_weekend', 'springfestival', 'nationalday', 'summer'], axis=1, inplace=True)

X_train, X_val, y_train, y_val = train_test_split(df_baseline_input, Y, test_size=0.3, random_state=78)

# print(X_val.head())


LR = LinearRegression()

LR.fit(X_train, y_train)

print('LR_model训练集准确率：\n', LR.score(X_train, y_train))  # 分数
print('LR_model验证集准确率：\n', LR.score(X_val, y_val))


# y_baseline_pred_LR =(LR.predict(X_val.head())).astype(int)
# print(y_baseline_pred_LR)
def load_LR_model():
    return LR
