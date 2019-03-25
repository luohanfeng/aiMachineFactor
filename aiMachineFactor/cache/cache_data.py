"""从数据库获取原始数据，
进行时序处理、截面处理，缓存到本地

"""
import os
# import warnings
from aiDataSource.gateway import MongoGateway

from aiMachineFactor.dataPrepare.cs_handle import fill_standard
from aiMachineFactor.dataPrepare.ts_handle import add_field, ts_ffill
from aiMachineFactor.utils import cache2pickle
import pandas as pd


# -------------------------------------------------------------------------------------
# 基本设置
# warnings.filterwarnings("ignore")
from aiMachineFactor.utils.const import FEATURE_FIELD

index_code = '000016.SH'
beginTime = '20100908'
endTime = '20181231'

cachepath = os.path.splitext(__file__)[0] + '/'
if not os.path.exists(cachepath):
    os.makedirs(cachepath)


def from_mongo():
    mggw = MongoGateway()
    universe = mggw.indexconstituent_weight(index_code, beginTime, endTime)
    allstock = mggw.universe_stock(universe, beginTime, endTime)
    classone = mggw.class_one(endTime)
    indexdata = mggw.single_index(index_code, beginTime, endTime)

    # 缓存本地
    cache2pickle(cachepath, universe, allstock, classone, indexdata)


def handling_ts_cs():
    universe = pd.read_pickle(cachepath+'universe')
    allstock = pd.read_pickle(cachepath+'allstock')
    classone = pd.read_pickle(cachepath+'classone')
    indexdata = pd.read_pickle(cachepath+'indexdata')

    # 时序处理，增加字段
    allstock.reset_index(drop=True, inplace=True)
    total_stock = allstock.groupby('windcode', as_index=False, group_keys=False).apply(add_field, indexdata)

    # 截面处理
    total_stock = total_stock.groupby('date', as_index=False, group_keys=False).apply(fill_standard,
                                                                                      classone,
                                                                                      'industry_sw',
                                                                                      )
    cache2pickle(cachepath, total_stock)


def handling_ffill():
    # 最后做填充处理
    total_stock = pd.read_pickle(cachepath + 'total_stock')
    total_stock.reset_index(drop=True, inplace=True)
    filled = total_stock.groupby('windcode', as_index=False, group_keys=False).apply(ts_ffill, FEATURE_FIELD)
    total_stock[FEATURE_FIELD] = filled[FEATURE_FIELD]
    total_filled = total_stock

    cache2pickle(cachepath, total_filled)


if __name__ == '__main__':
    handling_ffill()
