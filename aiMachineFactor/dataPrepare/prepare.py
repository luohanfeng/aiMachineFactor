"""数据准备的处理函数

"""
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from aiMachineFactor.utils import system_log


def ex_rate(df, index_df):
    """
    计算当期超额收益，前复权方式
    :param df:
    :param index_df:
    :return:
    """
    try:
        df[['close', 'adjfactor']]
        index_df[['date', 'pct_chg', 'windcode']]
    except KeyError:
        return None

    temp_df = df[df['trade_status'] == '交易'][['close', 'adjfactor']].copy()
    temp_df['qfq_close'] = temp_df['adjfactor'] / temp_df['adjfactor'].max() * temp_df['close']
    temp_df['qfq_pct_chg'] = (temp_df['qfq_close'] / temp_df['qfq_close'].shift(1) - 1) * 100

    # 聚合
    temp_ind = index_df[['date', 'pct_chg', 'windcode']].set_index('date')
    temp_ind.rename(columns={'pct_chg': 'pct_chg_bench', 'windcode': 'windcode_bench'}, inplace=True)
    temp_df = temp_df.join(temp_ind)
    temp_df['ex_rate'] = temp_df['qfq_pct_chg']-temp_df['pct_chg_bench']

    df['ex_rate'] = temp_df['ex_rate']
    system_log.info('增加列数据{}'.format('ex_rate'))

    return 'ex_rate'


def ex_rate_Nd(df, ndays=20):
    """

    :param df: 要求index为日期形式，并升序
    :param ndays:
    :return:
    """
    try:
        df[['trade_status', 'adjfactor', 'ex_rate']]
    except KeyError:
        return None

    offset = relativedelta(days=ndays)

    temp_df = df[df['trade_status'] == '交易'][['close', 'ex_rate']].copy()
    temp_df['trade_date'] = pd.to_datetime(temp_df.index)
    col_name = 'ex_rate_{}d'.format(ndays)
    temp_df[col_name] = np.NaN

    # 识别每个日期的未来天数
    for iloc in range(len(temp_df)):
        trade_date = pd.to_datetime(temp_df['trade_date'][iloc])
        if trade_date+offset > pd.to_datetime(temp_df['trade_date'][-1]):
            break
        future_date = temp_df[trade_date+offset:]['trade_date'].min()  # 不在交易时段，继续往下取
        temp_df.loc[trade_date, col_name] = temp_df['ex_rate'][trade_date: future_date].sum()

    df[col_name] = temp_df[col_name]
    system_log.info('增加列数据{}'.format(col_name))
    return col_name


# ------------------------------------------------------------------------------------------------
def fill_by_industry(df, industry_name, field_list):
    """

    :param df:
    :param industry_name:
    :param field_list:
    :return:
    """
    df.loc[:, field_list] = df.loc[:, field_list].fillna(df.groupby(industry_name)[field_list].transform('mean'))


def winsorize_standard(df, field_list):

    def winsorize(each):
        """去极值"""
        if each.isnull().all():
            return each

        median = each.median()
        mad = (each-median).abs().median()
        each[each > median+5*mad] = median+5*mad
        each[each < median-5*mad] = median-5*mad

        return each

    def standard(each):
        """标准化"""
        if each.isnull().all():
            return each

        mean = each.mean()
        std = each.std()
        return (each-mean)/std

    for each_field in field_list:
        df[each_field] = winsorize(df[each_field])
        df[each_field] = standard(df[each_field])


def neutralize():
    # TODO 行业市值，中性化
    pass
