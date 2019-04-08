import numpy as np
import pandas as pd
from scipy.stats import mstats
from scipy import stats
from datetime import datetime
from .plotting import *
from pyecharts import *
from .utils import *
from .performance import *
import os
online()

"""
序列数据的多图布局
"""
TS_LAYOUT = {2: [{'grid_bottom': "60%"}, {'grid_top': "60%"}],
             3: [{'grid_bottom': 550, 'grid_top': 50},
                 {'grid_bottom': 300, 'grid_top': 300},
                 {'grid_bottom': 50, 'grid_top': 550}],
             4: [{'grid_bottom': 610, 'grid_top': 40},
                 {'grid_bottom': 420, 'grid_top': 230},
                 {'grid_bottom': 230, 'grid_top': 420},
                 {'grid_bottom': 40, 'grid_top': 610}]}

"""
热力图的多图布局
"""
HEAT_LAYOUT = {2: [{'grid_left': "60%"}, {'grid_right': "60%"}],
               3: [{'grid_right': '55%', 'grid_bottom': '55%'},
                   {'grid_left': '55%', 'grid_bottom': '55%'},
                   {'grid_right': '55%', 'grid_top': '55%'}],
               4: [{'grid_right': '58%', 'grid_bottom': '58%'},
                   {'grid_left': '58%', 'grid_bottom': '58%'},
                   {'grid_right': '58%', 'grid_top': '58%'},
                   {'grid_left': '58%', 'grid_top': '58%'}]}



LEGEND_LAYOUT = {'ts':[{'legend_pos': 550, 'legend_top': 40,'legend_bottom':610},
                 {'legend_pos': 550,'legend_top': 230,'legend_bottom':420},
                 {'legend_pos': 550,'legend_top': 420,'legend_bottom':230},
                 {'legend_pos': 550, 'legend_top': 610,'legend_bottom':40}],
                 'ht':[{'legend_pos':'24%', 'legend_top':'48%'},
                 {'legend_pos': '76%','legend_top':'48%'},
                 {'legend_pos': '24%','legend_top':'52%'},
                 { 'legend_pos': '76%','legend_top':'52%'}]}

NAME_DICT = {'1D': '一天', '5D': '一周', '10D': '半月', '20D': '一月'}

TITLE_SETTING = [{'title_pos':80,'title_top':60,'title_text_size':12},
                 {'title_pos':465,'title_top':60,'title_text_size':12},
                 {'title_pos':80,'title_top':465,'title_text_size':12},
                 {'title_pos':465,'title_top':465,'title_text_size':12}]

def mov_ts(se, title, col, legend_pos):
    """
    带移动平均的时间序列图模板,
    """
    line_se = Line(title=title, title_pos='center')
    line_se.add(NAME_DICT[se.name], se.index.strftime("%Y-%m-%d").to_list(), se.values,
                **legend_pos,yaxis_max=round(se.max(), 2))
    ma_se = se.rolling(window=22).mean()
    line_se.add('月移动平均{}'.format(col),ma_se.index.strftime("%Y-%m-%d").to_list(), ma_se.values,
                **legend_pos)
    return line_se


def hist(se,legend_pos,title_setting):
    """
    pyecharts的分布图的模拟
    """
    if isinstance(se, pd.Series):
        #估计序列的概率密度,默认用gaussian_kde即高斯核估计
        #scott Rule 估计bins,并计算se的均值标准差
        kde = stats.gaussian_kde(se)
        # mean = se.mean().round(3)
        mean = round(se.mean(), 3)
        # std = se.std().round(3)
        std = round(se.std(),3)
        low = np.floor(se.min())
        high = np.ceil(se.max())
        bw = kde.scotts_factor()*se.std(ddof=1)

        #分区间统计频率，并归一化
        cut = np.arange(low,high,bw)
        se = se.groupby(pd.cut(se, cut)).count()
        se = se / (se.sum()*bw)

        #构造密度曲线数据和直方图数据
        se_kde = [np.asscalar(kde(x.left)) for x in se.index]
        df = pd.DataFrame({'hist': se, 'kde': se_kde})
        # index = [str(x.left.round(2)) for x in df.index]
        index = [str(round(x.left, 2)) for x in df.index]
        df.index = index
        bar = Bar(title='均值:{}\n标准差:{}'.format(mean, std),
                      **title_setting)
        bar._colorlst = ['#98f5ff', '#4169e1', '#98f5fe','#4169e2',
                         '#98f5fd', '#4169e3', '#98f5fc','#4169e4']
        bar.add(se.name, df.index, df['hist'].values, bar_category_gap='2%',**legend_pos)
        line = Line('信息系数分布图')
        line.add('', df.index, df['kde'].values, symbol_size=1)
        overlap = Overlap()
        overlap.add(bar)
        overlap.add(line)
        return overlap
    else:
        raise TypeError("se expected to be pd.Series")


def extreme_se(se):
    """
    去极值,极值用端点值替代
    :param se: pd.Series
    """
    q = se.quantile([0.025, 0.975])
    if isinstance(q, pd.Series) and len(q) == 2:
        se[se < q.iloc[0]] = q.iloc[0]
        se[se > q.iloc[1]] = q.iloc[1]
    return se


def std_se(se):
    """
    标准化，减均值除标准差
    :param se: pd.Series
    """
    se_mean = se.mean()
    se_std = se.std()
    se = (se - se_mean) / se_std
    return se


def clean_data(factor, df):
    """
    从原始股票中挑选与因子股票的股票名称、交易日期相同的股票数据
    :param factor:pd.DataFrame,含因子数据的股票的数据框
    :param df: pd.DataFrame,原始股票数据框
    """
    # index = pd.MultiIndex.from_frame(df[['date', 'windcode']])
    index = pd.MultiIndex.from_arrays([df['date'], df['windcode']])
    df.index = index
    index = factor.index
    df = df.loc[index]
    return df


def path_generator(dir):
    """
    路径生成器，生成所有预测因子的路径
    :param dir:因子路径，文件夹名
    """
    g = os.walk(dir)
    for path, _, file_list in g:
        for file_name in file_list:
            yield os.path.join(path, file_name)


def series_unstack(se):
    y_axis = [index.year for index in se.index]
    x_axis = [index.month for index in se.index]
    index = pd.MultiIndex.from_arrays([y_axis, x_axis], names=['year', 'month'])
    temp_se = se.copy()
    temp_se.index = index
    df = temp_se.unstack()
    return df


def split_by_len(s, iterable):
    """
    this func split the char str by length list or ...
    """
    result = []
    for l in iterable:
        result.append(s[:l])
        s = s[l:]
    return result


def connect_strs(strs, symbol=None):
    """
    this func connects a str list using symbol(if specified)
    """
    result = ''
    if symbol:
        symbols = [symbol]*(len(strs)-1)
        symbols.append('')
        for k, v in zip(strs, symbols):
            result += (k+v)
    else:
        for _str in strs:
            result += _str
    return result


def cumulative_return_q(df, period):
    """
    计算固定周期换仓的累积收益
    :param df:mean_return_q_daily
    :param period:
    """
    freq = df.index.levels[1].freq
    ret_wide = df.unstack('factor_quantile')
    cum_ret = ret_wide.apply(cumulative_returns, period=period, freq=freq)
    cum_ret.loc[:, ::-1]
    return cum_ret


def grid_charts(charts, pattern='ts'):
    """
    序列数据的多图排列
    """
    if isinstance(charts, (set, list, tuple)):
        assert len(charts) in (2, 3, 4), "no such layout subplots 2, 3 or 4"
        if len(charts) == 2:
            grid = Grid(width=800, height=600)
            layouts = TS_LAYOUT[2] if pattern == 'ts' else HEAT_LAYOUT[2]
            for chart, layout in zip(charts, layouts):
                grid.add(chart, **layout)
        if len(charts) == 3:
            grid = Grid(weight=800, height=800)
            layouts = TS_LAYOUT[3] if pattern == 'ts' else HEAT_LAYOUT[3]
            for chart, layout in zip(charts, layouts):
                grid.add(chart, **layout)
        if len(charts) == 4:
            grid = Grid(width=800, height=800)
            layouts = TS_LAYOUT[4] if pattern == 'ts' else HEAT_LAYOUT[4]
            for chart, layout in zip(charts, layouts):
                grid.add(chart, **layout)
    else:
        raise TypeError("charts should be a list, tuple or set")
    return grid


class PlotData(object):
    """
    所有图形的数据接口,最原始读取的数据私有,函数计算
    得到的因子数据受保护
    """
    def __init__(self, factor, stock, industry):
        """
        factor:pd.DataFrame
        stock: pd.DataFrame
        industry: pd.DataFrame
        """
        self.factor = factor
        self.stock = stock
        self.industry = industry
        self._factor = pd.DataFrame()
        self.feature = pd.DataFrame()

    @property
    def factor(self):
        return self.__factor

    @property
    def stock(self):
        return self.__stock

    @property
    def industry(self):
        return self.__industry

    @factor.setter
    def factor(self, factor):
        if not isinstance(factor, pd.DataFrame):
            raise TypeError("factor data should be a dataframe")
        else:
            try:
                factor[['windcode', 'pre', 'date']]
            except AttributeError:
                print("factor data should contain pre, date, and windcode")
            else:
                self.__factor = factor[['date', 'windcode', 'pre', ]].copy()
        return self.__factor

    @stock.setter
    def stock(self, stock):
        if not isinstance(stock, pd.DataFrame):
            raise TypeError("stock should be dataframe")
        else:
            try:
                stock[['date', 'windcode', 'close']]
            except AttributeError:
                print("stock data contain data, windcode, close")
            else:
                stock = stock[['date', 'windcode', 'close']].copy()
                self.__stock = stock
        return self.__stock

    @industry.setter
    def industry(self, industry):
        if isinstance(industry, pd.DataFrame):
            self.__industry = industry
        return self.__industry

    @classmethod
    def from_pickle(cls, factor_dir, stock_dir, classone_dir):
        paths = path_generator(factor_dir)
        factor = pd.DataFrame()
        for path in paths:
            new_factor = pd.read_pickle(path)
            factor = pd.concat([factor, new_factor])
        stock = pd.read_pickle(stock_dir)
        classone = pd.read_pickle(classone_dir)
        return cls(factor, stock, classone)


    def read_feature(self, dir):
        paths = path_generator(dir)
        feature = pd.DataFrame()
        for path in paths:
            new_feature = pd.read_pickle(path)
            feature = pd.concat([feature, new_feature])
        self.feature = feature


    def plot_feature(self, num_feature=30):
        feature = self.feature.groupby(level=0).sum().sort_values('importance',ascending=False)
        df = feature.iloc[0:num_feature]
        bar = Bar("因子重要性排名", title_pos='center')
        bar._colorlst=['#0000cd']
        bar.add("",df.index, df['importance'],xaxis_rotate=90)
        return bar


    def clean_factor(self):
        """
        整理因子数据格式以配合函数get_clean_factor_and_forward_returns
        """
        if not isinstance(self.__factor.index, pd.MultiIndex):
            self.__factor.index = self.__factor.date
            self.__factor = self.__factor.sort_index()
            # index = pd.MultiIndex.from_frame(self.__factor[['date', 'windcode']])
            index = pd.MultiIndex.from_arrays([self.__factor['date'],self.__factor['windcode']])
            factor = self.__factor.pre
            factor.index = index
            self.__factor = factor

    def clean_stock(self):
        """
        整理股票数据格式,使其股票和日期与因子数据对齐
        """
        if not self.__stock.columns.name == 'windcode':
            if isinstance(self.__factor.index, pd.MultiIndex):
                self.__stock = clean_data(self.__factor, self.__stock)
                self.__stock = self.__stock.pivot(index='date',
                                                  columns='windcode', values='close')
            else:
                print("clean factor first")

    def clean_industry(self):
        """
        整理股票行业成字典格式
        """
        if not isinstance(self.__industry, dict):
            grouper_df = self.__industry[['industry_citic', 'windcode']]
            if isinstance(self.__factor.index, pd.MultiIndex):
                stock_list = self.__factor.index.levels[1].to_list()
                temp = grouper_df[grouper_df['windcode'].isin(stock_list)]
                grouper = dict(zip(temp.windcode, temp.industry_citic))
                self.__industry = grouper
            else:
                print("clean factor first")

    def get_factor(self):
        self.clean_factor()
        self.clean_stock()
        self.clean_industry()
        self._factor = get_clean_factor_and_forward_returns(
            self.__factor, self.__stock, self.__industry, periods=(1, 5, 10, 20))


class ReturnPlot(object):
    """
    因子收益分析图
    """

    def __init__(self, plot_data):
        self.plot_data = plot_data
        self.mean_return_q = None
        self.std_err_q = None
        self.mean_return_q_daily = None
        self.std_err_q_daily = None
        self.q_return_spread = None
        self.std_err_spread = None

    def compute_mean_return(self):
        self.mean_return_q, self.std_err_q = mean_return_by_quantile(
            self.plot_data._factor, by_group=False)
        self.mean_return_q_daily, self.std_err_q_daily = mean_return_by_quantile(
            self.plot_data._factor, by_group=False, by_date=True)
        self.q_return_spread, self.std_err_spread = compute_mean_returns_spread(
            self.mean_return_q_daily, 5, 1, self.std_err_q_daily)

    def plot_q_bar(self):
        """
        各分位数组合的各周期的平均超额收益柱形图
        """
        bar = Bar('分位数组合平均超额收益', title_pos='center')
        df = self.mean_return_q
        periods = df.columns.to_list()
        quantile = df.index.to_list()
        bar._colorlst = ['#ff4500', '#8470ff', '#008b00', '#4169e1']
        for period in periods:
            bar.add(period, quantile, df[period],
                    legend_pos='center', legend_top='bottom')
        return bar

    def plot_q_spread(self):
        """
        固定周期（1天，5天，10天, 20天）的收益率最好分位数与最差分位数收益价差图
        """
        charts = []
        for col, legend_pos in zip(self.q_return_spread.columns, LEGEND_LAYOUT['ts']):
            se = self.q_return_spread[col]
            ts = mov_ts(se,title='分位数收益价差',col=col,legend_pos=legend_pos)
            charts.append(ts)
        spread = grid_charts(charts)
        return spread

    def plot_cum_ret_q(self, period='20D'):
        """
        分位数组合累积收益图
        :param period:pandas.Timedelta or Str('1D','1h','5D')
        :return:
        """
        df = cumulative_return_q(self.mean_return_q_daily, period)
        line = Line("分位数组合累积收益""({}前瞻收益)".format(period))
        df_plot = df[period].copy()
        for col in df_plot.columns:
            line.add(df_plot[col].name, df_plot[col].index.strftime("%Y-%m-%d").to_list(),
                     df_plot[col].values)
        return line

    def plot_cum_ret_p(self, periods=('1D', '5D', '10D', '20D')):
        """
        :param periods:
        :return:
        """
        df = factor_returns(self.plot_data._factor)
        freq = df.index.freq
        line = Line("因子加权的多空组合累积收益", title_pos='center')
        line._colorlst = ['#ff4500', '#8470ff', '#008b00', '#4169e1']
        for period in periods:
            cum_ret = cumulative_returns(df[period], period, freq)
            line.add(period, cum_ret.index.strftime("%Y-%m-%d").to_list(),
                     cum_ret.values,)
                     # legend_pos='center', legend_top='bottom')
        return line


class ICPlot(object):

    def __init__(self, plot_data):
        self.plot_data = plot_data
        self.mean_monthly_ic = None
        self.ic = None

    def compute_ic(self):
        self.mean_monthly_ic = mean_information_coefficient(self.plot_data._factor, by_time='M')
        self.ic = factor_information_coefficient(self.plot_data._factor)

    def plot_ic_heatmap(self):
        charts = []
        for col, legend_pos in zip(self.mean_monthly_ic.columns, LEGEND_LAYOUT['ht']):
            se = self.mean_monthly_ic[col]
            year = sorted(list({index.year for index in se.index}))
            month = sorted(list({index.month for index in se.index}))
            df = series_unstack(se)
            data = [[i, j, df.iloc[i, j]] for i in range(len(year)) for j in range(len(month))]
            heatmap = HeatMap("信息系数热力图", width=300, height=300, title_pos='center')
            heatmap.add(col,
                        year, month, data,
                        is_visualmap=True,
                        visual_range=[df.min().min(), df.max().max()],
                        visual_orient="horizontal", **legend_pos)
            charts.append(heatmap)
        heat = grid_charts(charts, pattern='ht')
        return heat

    def plot_ic_ts(self):
        charts = []
        for col, legend_pos in zip(self.ic.columns, LEGEND_LAYOUT['ts']):
            se = self.ic[col]
            ts = mov_ts(se, title='信息系数IC', col=col, legend_pos=legend_pos)
            # ts._colorlst = ['#98f5fa', '#008B00', '#98f5fb', '#008b01',
            #                 '#98f5fc', '#008b02', '#98f5fd', '#008b03']
            charts.append(ts)
        spread = grid_charts(charts)
        return spread

    def plot_ic_hist(self):
        charts = []
        for col, legend_pos, title_setting in zip(self.ic.columns, LEGEND_LAYOUT['ht'], TITLE_SETTING):
            se = self.ic[col]
            hs = hist(se,legend_pos,title_setting)
            charts.append(hs)
        ic_hist = grid_charts(charts,pattern='ht')
        return ic_hist


class TurnoverPlot(object):

    def __init__(self, plot_data):
        self.plot_data = plot_data
        self.q_factor = plot_data._factor['factor_quantile']
        self.factor_ac = None
        self.q_turnover = None

    def compute_q_turnover(self):
        turnover_periods = utils.get_forward_returns_columns(
            self.plot_data._factor.columns)
        self.q_turnover = \
            {p: pd.concat([quantile_turnover(self.q_factor, q, p)
                           for q in range(1, int(self.q_factor.max()) + 1)],
                          axis=1)
             for p in turnover_periods}
        self.factor_ac = pd.concat(
            [factor_rank_autocorrelation(self.plot_data._factor, period) for period in
             turnover_periods], axis=1)

    def plot_turnover(self, period='20D'):
        df = self.q_turnover[period]
        top, bottom = 5, 1
        line = Line("{}最高与最低分位数组合换手率".format(period),
                    title_pos='center')
        line._colorlst = ['#008B45', '#4169E1']
        line.add('5', df[top].index.strftime("%Y-%m-%d").to_list(),
                 df[top].values,
                 legend_pos='center', legend_top='bottom')
        line.add('1', df[bottom].index.strftime("%Y-%m-%d").to_list(),
                 df[bottom].values,
                 legend_pos='center', legend_top='bottom')
        return line

    def plot_ac(self, period='20D'):
        line = Line("因子自相关系数({})".format(period), title_pos='center')
        line._colorlst = ['#4169E1']
        se = self.factor_ac[period]
        line.add(period, se.index.strftime("%Y-%m-%d").to_list(),
                 se.values, legend_pos='right')
        return line


