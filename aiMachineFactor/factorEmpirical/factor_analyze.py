"""因子性能测试

"""
import os
import numpy as np
import pandas as pd


def cache2factor(factorfield='pre',
                 dirpath='E:/aiMachineFactor-master/aiMachineFactor/factorGenerate/xgb_binary/predict/'):
    # 数据列表
    factor_list = []

    # 遍历文件
    file_list = os.listdir(dirpath)
    for each_file in file_list:
        file = os.path.join(dirpath, each_file)
        if os.path.isdir(file):
            continue

        df = pd.read_pickle(file)
        factor_list.append(df)

    # 数据转换
    factor_data = pd.concat(factor_list)
    return factor_data.set_index(['date', 'windcode'])[factorfield]


def cache2price(pricefield='close',
                path='E:/aiMachineFactor-master/aiMachineFactor/cache/cache_data/',
                filename='total_stock'):
    # 读取数据
    file = path+filename
    temp_df = pd.read_pickle(file)

    # fixme 一使用复权价格，二某些日期存在nan
    return temp_df.pivot(index='date', columns='windcode', values=pricefield)


def cache2group_dict(groupfield='industry_sw',
                     path='E:/aiMachineFactor-master/aiMachineFactor/cache/cache_data/',
                     filename='classone'):
    # 读取数据
    file = path+filename
    temp_df = pd.read_pickle(file)

    # 数据转换
    temp = temp_df[['windcode', groupfield]].set_index('windcode')
    return temp[groupfield].to_dict()


if __name__ == '__main__':
    factor = cache2factor()
    prices = cache2price()
    factor_groups = cache2group_dict()

    prices.plot()
