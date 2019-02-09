# -*- coding: UTF-8 -*-
import sklearn
import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
np.set_printoptions(suppress=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
'''
boston = load_boston()
print("data shape:{}".format(boston.data.shape))
print("target shape:{}".format(boston.target.shape))
print("line head 5:\n{}".format(boston.data[:5]))
print("target head 5:\n{}".format(boston.target[:5]))
'''
train_df = pd.read_csv("/Users/wangsen/ai/03/9day_discuz/firstDiscuz/02_houseprice/data/train.csv",index_col=0)
print("train shape:{}".format(train_df.shape))
test_df = pd.read_csv("/Users/wangsen/ai/03/9day_discuz/firstDiscuz/02_houseprice/data/test.csv",index_col=0)

print("test shape:{}".format(test_df.shape))
print("train columns:\n{}".format(train_df.dtypes))
train_target = train_df.pop("SalePrice")
#train_df = train_df.drop("SalePrice",axis=1)
## read_csv加载csv文件
## index_col=0,指明第一列为id列
#print(train_df.info())
##print(train_df.describe().T)
#print(train_df['MSSubClass'].value_counts())
#print(train_df['MSSubClass'].unique())
## unique 查看数据
## value_counts 数据统计
#数据预处理，训练集和测试集一起做数据预处理
train_len = train_df.shape[0]
test_len = test_df.shape[0]


all_df = pd.concat((train_df,test_df),axis=0)

# all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
# print("all_df shape:{}".format(all_df.shape))
# '''
# print(all_df['MSSubClass'].value_counts())
# print(all_df['MSSubClass'].unique())
# sb = all_df['MSSubClass'].unique()
# print(pd.get_dummies(sb))
# print(pd.concat((all_df['MSSubClass'][:5],pd.get_dummies(all_   df['MSSubClass'], prefix='MSSubClass')[:5]),axis=1).T)
# '''
all_dummy_df = pd.get_dummies(all_df)
# print(all_dummy_df.head())
print(all_dummy_df.shape)
print("查看空值情况：\n{}".format(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)))
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)
# numeric_cols = all_df.columns[all_df.dtypes != 'object']
# print("全局编码后的列名:\n{}".format(all_dummy_df.dtypes.value_counts()))
# print("全局编码后的列信息:\n{}".format(all_dummy_df.dtypes))


dummy_train_df = all_dummy_df[:train_len]

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#lr.fit(dummy_train_df,train_target)
#print("训练集评分：{}".format(lr.score(dummy_train_df,train_target,scor)))

kf = KFold(n_splits=5, shuffle=True)
score_ndarray = np.sqrt(-cross_val_score(lr, dummy_train_df, train_target, cv=kf,scoring="neg_mean_squared_error"))
print(score_ndarray.mean())

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=200, max_features=3)
score_ndarray = np.sqrt(-cross_val_score(clf, dummy_train_df, train_target, cv=kf,scoring="neg_mean_squared_error"))
print(score_ndarray.mean())

clf.fit(dummy_train_df,train_target)
train_predict = clf.predict(dummy_train_df)
from sklearn.metrics import mean_squared_error
print("随机森林算法的误差:",np.sqrt(mean_squared_error(train_target,train_predict)))

lr.fit(dummy_train_df,train_target)
train_predict = lr.predict(dummy_train_df)
from sklearn.metrics import mean_squared_error
print("线性回归的误差：",np.sqrt(mean_squared_error(train_target,train_predict)))