"""xbgoost模型做二分类

"""
import os
import warnings

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from aiMachineFactor.dataPrepare.sample_handle import SubSample, feature_label, binary_partly
from aiMachineFactor.utils.const import FEATURE_FIELD
from aiMachineFactor.utils.logger import machine_log


def rolling_generate():
    """滚动回测的函数"""
    # 设置环境
    warnings.filterwarnings("ignore")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # 设置参数
    samplein_n = 12  # 12个月训练期，1个月预测期
    sampleout_n = 1

    g_params = {
        'subsample': np.arange(0.5, 1, 0.1),
        'n_estimators': np.arange(100, 600, 100),
    }

    # TODO 必须初始化？其它参数
    common_params = {
        'learning_rate': 0.1,
        'n_estimators': 500,
        'max_depth': 5,
        'min_child_weight': 1,
        'seed': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 1,
        'reg_lambda': 1,
    }

    # 设置特征和标签
    feature = FEATURE_FIELD
    label = ['ex_rate_20d']

    # 读取数据
    cachepath = 'E:/aiMachineFactor-master/aiMachineFactor/cache/cache_data/'
    universe = pd.read_pickle(cachepath + 'universe')
    classone = pd.read_pickle(cachepath + 'classone')
    indexdata = pd.read_pickle(cachepath + 'indexdata')
    total_stock = pd.read_pickle(cachepath + 'total_filled')

    # 去除空值
    total_stock = total_stock.dropna(subset=feature+label)
    if total_stock.isnull().any().any():  # 检查
        machine_log.warn('仍然有缺失值')
        print(total_stock[total_stock.isnull().value is True])

    # 设置时间索引
    total_stock.set_index('date', drop=False, inplace=True)
    resample_ind = total_stock['date'].groupby([total_stock['date'].dt.year,
                                                total_stock['date'].dt.month]
                                               ).max()

    # 设置子集
    indexdata.set_index('date', drop=False, inplace=True)
    bench_ind = SubSample.bench_huge(indexdata, 1.5)
    last_ind = SubSample.month_last(total_stock)
    sub = list(bench_ind)
    sub.extend(last_ind)
    sub = list(set(sub))
    sub.sort()
    sub = pd.to_datetime(sub)

    # 滚动回测
    for ind, trade_date in resample_ind.iteritems():
        # 时间标签datetime.datetime对象，或timestamp都行
        trade_date = pd.Timestamp.to_pydatetime(trade_date)  #
        temp = trade_date - relativedelta(months=sampleout_n-1)
        out_start = datetime.datetime(temp.year, temp.month, 1)  # 月初
        out_end = trade_date  # 当月最后一个有交易的日期，但不一定是最后的自然日
        in_end = out_start-relativedelta(days=1)
        in_start = out_start-relativedelta(months=samplein_n)

        if in_start < total_stock['date'].min():
            continue

        # 划分数据集
        samplein = total_stock[in_start:in_end]
        samplein_data = samplein[samplein.index.isin(sub)]

        # 筛选部分正负例
        samplein_data = binary_partly(samplein_data, 'ex_rate_20d')
        sampleout = total_stock[out_start:out_end]

        try:
            # 调用模型
            model = GridXgbBinary(out_end, g_params, common_params)
            model.fit(samplein_data, feature, ['binary_partly'])  # 记录预测概率，所以标签有所不同
            model.predict(sampleout, feature, ['ex_rate_20d'])
            # 保存结果
            model.cache_importance()
            model.cache_predict()
        except Exception as e:
            machine_log.exception(out_end, e)
            continue

        machine_log.info('完成截面{}>>>{}'.format(out_start, out_end))


class GridXgbBinary(object):
    def __init__(self, period, g_params, common_params):
        self.flag = period
        # 参数
        self.g_params = g_params
        self.common_params = common_params
        # 结果
        self.gsearch = None
        self.importance = None
        self.pre_df = None
        machine_log.info('模型初始化{}'.format(self.flag.strftime('%Y-%m-%d')))

    def fit(self, samplein, feature, label, cv=5):
        # 划分特征和标签
        x_train, y_train = feature_label(samplein, feature, label)
        y_train = y_train.values.ravel()  # TODO 解决DataConversionWarning
        # 训练
        model = xgb.XGBClassifier(**self.common_params)
        gsearch = GridSearchCV(estimator=model,
                               param_grid=self.g_params,
                               scoring='r2',
                               cv=cv,
                               n_jobs=-1,
                               )
        gsearch.fit(x_train, y_train)

        self.gsearch = gsearch
        self.importance = pd.DataFrame(self.gsearch.best_estimator_.feature_importances_,
                                       index=feature, columns=['importance'],
                                       )

        machine_log.debug('训练数据{}得分{}'.format(len(x_train), gsearch.best_score_))

    def predict(self, sampleout, feature, label):
        # 划分特征和标签
        x_test, y_test = feature_label(sampleout, feature, label)
        # 训练
        result = self.gsearch.predict_proba(x_test)  # 记录概率,0 1 概率相加为1
        temp = sampleout.copy()
        temp['pre'] = result[:, 1]  # 只保留1的概率

        self.pre_df = temp[['date', 'windcode']+label+['pre']]
        # evs = metrics.explained_variance_score(y_test, result)  # FIXME 会出错
        # mae = metrics.median_absolute_error(y_test, result)
        # machine_log.debug('完成预测evs{}>>>>mae{}'.format(evs, mae))

    def cache_importance(self, filepath=None):
        if filepath is None:
            filepath = os.path.splitext(__file__)[0] + '/importance/'

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        self.importance.to_pickle(filepath+'{}'.format(self.flag.strftime('%Y-%m-%d')))

    def cache_predict(self, filepath=None):
        if filepath is None:
            filepath = os.path.splitext(__file__)[0] + '/predict/'

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        self.pre_df.to_pickle(filepath+'{}'.format(self.flag.strftime('%Y-%m-%d')))


if __name__ == '__main__':
    rolling_generate()
