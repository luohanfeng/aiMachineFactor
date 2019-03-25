# 

## 项目结构
bin-----执行脚本

cache----缓存数据到本地，存储到cache_data文件夹，pd.pickle格式。

> 第一种：
* universe：按日期组成的指数股票池。
* allstock：按照指数股票池，取出各个股票在数据库中的字段。
* classone: 证券分类。
* indexdata：指数在数据库中的字段。

> 第二种：
* total_stock：对allstock进行时间序列上字段扩展，截面上行业分类、行业填充、去极值、标准化。
* total_filled：对total_stock进行向前填充。

dataPrepare----数据处理模块。

factorEmpirical----因子分析模块，包括alphalens包，生成alphalens数据格式，echart绘图函数。

factorGenerate----存储机器学习的脚本，并生成同名的缓存文件。

* 一个算法，对应一个py执行文件，和同名文件夹缓存运行结果。
* 同名文件夹，包括importance因子重要性排名，predict预测值，同名ipnb因子检验的脚本，同名html因子检验报告。
*算法py文件，包括相关参数、滚动回测函数、算法的类。

utils----常用的功能函数。


## 变量命名含义

根据滚动测试，每一期数据分为samplein和sampleout。

根据数据集和使用的有效数据，分为set和data。

根据交叉验证，samplein分为train和val，sampleout为test。

根据各列特性，字段分为x(或feature)和y(或label)。

