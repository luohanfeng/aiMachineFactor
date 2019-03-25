"""时序处理

"""
from aiDataSource.const.field import FieldTrading, FieldValuation, FieldMomentum
from aiMachineFactor.dataPrepare.prepare import ex_rate, ex_rate_Nd


def add_field(df, index_df):
    """增加dataframe的字段。

    :param df: 要求group之前的index不能重复
    :param index_df:
    :return:
    """
    # 筛选
    print(df['windcode'].iloc[0])
    df = df[df['trade_status'] == '交易']
    df = df[(df['date'] - df['ipo_date']).dt.days >= 365]
    if df.empty:
        return df

    # 设置索引为日期型
    stock_set = df.set_index('date')
    stock_set.sort_index(ascending=True, inplace=True)

    # 新增字段
    FieldTrading.turn_Nd(stock_set, 5)
    FieldTrading.std_turn_Nd(stock_set, 30)

    FieldValuation.bp(stock_set)
    FieldValuation.sp(stock_set)
    FieldValuation.ncfp(stock_set)
    FieldValuation.ocfp(stock_set)

    FieldMomentum.return_Nm(stock_set, 3)
    FieldMomentum.return_Nm(stock_set, 6)
    FieldMomentum.wgt_return_Nm(stock_set, 1)
    FieldMomentum.wgt_return_Nm(stock_set, 3)
    FieldMomentum.exp_wgt_return_Nm(stock_set, 3)
    FieldMomentum.exp_wgt_return_Nm(stock_set, 6)

    # 计算超额收益，未来n期累计超额收益
    ex_rate(stock_set, index_df)
    ex_rate_Nd(stock_set, 20)

    # 再转换回df的数值索引
    stock_set.reset_index(drop=False, inplace=True)
    stock_set.index = df.index

    return stock_set


def ts_ffill(df, field_list):
    """
    df要求group之前的index不能重复
    :param df:
    :param field_list:
    :return:
    """
    df[field_list] = df[field_list].ffill()
    return df

