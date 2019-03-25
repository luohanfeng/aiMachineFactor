"""截面数据处理

"""
from aiMachineFactor.dataPrepare.prepare import fill_by_industry, winsorize_standard
from aiMachineFactor.utils.const import CS_FIELD


def fill_standard(df, classone, industry_name):
    """数据填充，标准化

    :param df:
    :param classone:
    :return:
    """
    # 筛选
    print(df['date'].iloc[0])
    df = df[df['trade_status'] == '交易']
    df = df[(df['date'] - df['ipo_date']).dt.days >= 365]
    if df.empty:
        return df

    # 添加行业分类
    stock_set = df.merge(classone[['windcode', industry_name]],
                         on=['windcode'],
                         how='left',
                         )
    # 行业填充
    field_list = CS_FIELD
    fill_by_industry(stock_set, industry_name, field_list)
    winsorize_standard(stock_set, field_list)

    return stock_set



