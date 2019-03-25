"""样本处理

"""
import numpy as np
import pandas as pd


def binary_partly(df, field, percent=0.3):
    """根据df的field列，选取部分正例和负例

    :param df:
    :param field: string,单个字段
    :param percent:
    :return:
    """
    try:
        df[field]
    except Exception:
        RuntimeError('df{}不存在'.format(field))

    temp = df.sort_values(by=field, ascending=False)
    temp['binary_partly'] = np.nan
    n_select = np.around(np.multiply(percent, len(df[field]))).astype(int)

    # 正负例
    temp['binary_partly'][0:n_select] = 1
    temp['binary_partly'][-n_select:] = 0

    return temp[temp['binary_partly'].notnull()]


def feature_label(df, feature, label):
    """划分特征和标签

    :param df:
    :param feature:
    :param label:
    :return: tuple
    """
    return df[feature], df[label]


class SubSample(object):
    """日线数据取子样本"""

    @staticmethod
    def bench_huge(indexdata, chg_filter=1.5):
        """筛选基准涨跌幅大于一定比例的交易日

        :param indexdata:
        :param chg_filter:
        :return:index
        """
        chg_hug = indexdata['pct_chg'].groupby([indexdata.index.year, indexdata.index.month]
                                               ).apply(lambda x: x[x.abs() > chg_filter])
        bench_ind = indexdata[indexdata['pct_chg'].isin(chg_hug)]
        return bench_ind.index

    @staticmethod
    def month_last(df):
        """筛选df的月末交易日

        :param df:
        :return: index
        """
        resample_ind = df['date'].groupby([df['date'].dt.year,
                                           df['date'].dt.month]
                                          ).max()
        return pd.DatetimeIndex(resample_ind)

