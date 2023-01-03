import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from os.path import expanduser
import h5py
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import tee
from itertools import compress
current_time = datetime.now().strftime('%Y-%m-%d')
import scipy.stats
def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

if os.name == 'posix':
    plt.rcParams['font.sans-serif'] = ['Songti SC']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False


HOME = expanduser("~")
rawpath = f'{HOME}/.rqalpha/bundle'
if not os.path.exists(rawpath):
    print('数据源路径不存在, 请在terminal中依次输入\n1. pip install rqalpha \n2. rqalpha download-bundle\n完成数据导入')
    
refresh_rate_map = {
    'Monthly(-1)': 'is_monthend',
    'Weekly(-1)': 'is_weekend',
    'Yearly(-1)': 'is_yearend',
}
name2code_index = {
    'SH50': '000016.XSHG',
    'HS300': '000300.XSHG',
    'ZZ500': '000905.XSHG',
    'ZZ800': '000906.XSHG',
    'ZZ1000': '000852.XSHG'
}

def get_TradeDate(start_date: str, 
                  end_date: str) -> pd.core.indexes.datetimes.DatetimeIndex:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    test = pd.to_datetime(np.array(np.load(f'{rawpath}/trading_dates.npy'),dtype='U8'))
    trade_dates = pd.Series(0,index=test)[start_date:end_date].index
    return trade_dates

def get_TransferDate(start_date: str, 
                  end_date: str, end=True, freq='month') -> pd.core.indexes.datetimes.DatetimeIndex:
    def pairwise(iterable):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    s = get_TradeDate(start_date, end_date)[0]
    e = get_TradeDate(start_date, end_date)[-1]
    if end:
        date_s = pd.to_datetime(s)
        date_e = pd.to_datetime(get_NextTradeDate(current_date=e,period=1))
        trade_date = get_TradeDate(date_s,date_e)
        exec(f'lst = list(map(lambda x:x[0]!=x[1],pairwise(map(lambda x:x.{freq},trade_date))))',locals())
        res = pd.to_datetime(list(compress(trade_date, locals()['lst'])))
        return res
    if not end:
        date_s = pd.to_datetime(get_NextTradeDate(current_date=s,period=-1))
        date_e = pd.to_datetime(e)
        trade_date = get_TradeDate(date_s,date_e)
        exec(f'lst = list(map(lambda x:x[0]!=x[1],pairwise(map(lambda x:x.{freq},trade_date))))',locals())
        res = pd.to_datetime(list(compress(trade_date, locals()['lst'])))
        return res   

def get_ex_factor(secID, start_date, end_date):
    trade_date = get_TradeDate(start_date, end_date)
    with h5py.File(f'{rawpath}/ex_cum_factor.h5',mode='r') as hf:
        df = pd.DataFrame(hf[secID][:])
    df.iloc[0,0] = '19900101000000'
    df.index = pd.to_datetime(np.array(df['start_date'],dtype='U8'))
    df = df['ex_cum_factor']
    true_time = [df.index.asof(t) for t in trade_date]
    ex_factor = df[-1]/df[true_time]
    ex_factor.index = pd.to_datetime(trade_date)
    return ex_factor

def get_NextTradeDate(current_date: str,period:int = -1) ->pd._libs.tslibs.timestamps.Timestamp:
    # 目前的版本要求current_date是一个tradedate
    current_date =  pd.to_datetime(current_date)
    shifts = 3*period + int(np.sign(period)*10)
    next = current_date + timedelta(days=shifts)
    days = get_TradeDate(min(current_date,next),max(current_date, next))
    index = np.where(days == current_date)[0] + period
    result = days[int(index)]
    return result

def get_data_indexes(secID: str, fields: str,
                       start_date: str, end_date: str):
    '''
    secID: 传入标准的指数代码, 例如 000300.XSHG
    fields: open, high, low, close, volumn, total_turnover
    '''
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    with h5py.File(f'{rawpath}/indexes.h5',mode='r') as hf:
        df = pd.DataFrame(hf[secID][:])
    data = df[fields]
    data.index = pd.to_datetime(np.array(df['datetime'],dtype='U8'))
    df = data.loc[start_date:end_date]
    return df

def get_data_stocks(secID, fields, start_date, end_date, exclude_ds=False):
    with h5py.File(f'{rawpath}/stocks.h5',mode='r') as hf:
        df = pd.DataFrame(hf[secID][:])
        data = df.loc[:,fields]
        data.index = pd.to_datetime(np.array(df['datetime'],dtype='U8'))     
    if not exclude_ds:
        df = data.loc[start_date:end_date]
        return df
    else:
        data = data.loc[start_date:end_date]
        ex_factor = get_ex_factor(secID,start_date,end_date)
        ex_data = data/ex_factor
        return ex_data
    
def get_data_index_components(secID='000300.XSHG', current_date=None) -> list:
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./data/index_components'):
        os.mkdir('./data/index_components')
    try:
        res = pd.read_csv(f'./data/index_components/{secID}.csv',index_col=0)
        res.index = pd.to_datetime(res.index)
        truetime = res.index.asof(f'{current_date}')
        return list(res.loc[truetime])
    except:
        print('本地数据不存在, 尝试下载')
        res = pd.read_csv(f'https://raw.githubusercontent.com/nymath/financial_data/main/data/index_components/{secID}.csv',index_col=0,encoding='utf8')
        res.index = pd.to_datetime(res.index)
        print('下载成功')
        res.index = pd.to_datetime(res.index)
        res.to_csv(f'./data/index_components/{secID}.csv',encoding='utf8')
        print('保存成功')
        truetime = res.index.asof(f'{current_date}')
        return list(res.loc[current_date])

def get_data_InterestRate(stard_date: str,
                          end_date: str,period: str ='10Y') -> pd.Series:
    with h5py.File(f'{rawpath}/yield_curve.h5',mode='r') as hf:
        df = pd.DataFrame(hf['data'][:])
    data = df[period]
    data.index = pd.to_datetime(np.array(df['date'],dtype='U8'))
    df = data[stard_date:end_date]
    return df

def RiskAssessment(simple_return_series: pd.DataFrame,
                   simple_riskfree_series: pd.DataFrame) -> dict:
    trade_date = simple_return_series.index.to_list()
    riskfree = simple_riskfree_series.loc[trade_date]
    portfolio_names0 = simple_return_series.columns
    portfolio_names1 = portfolio_names0.to_list().copy()
    portfolio_names1.remove('benchmark')
    
    template0 = pd.Series(data=0,index=portfolio_names0)
    template1 = pd.Series(data=0,index=portfolio_names1)
    
    log_return = np.log(1+simple_return_series)
    
    pnl = (1+simple_return_series).cumprod()
    assess = dict()
    # mean return  
    all_log_return_mean = log_return.mean() * 252
    all_log_return_std = log_return.std() * np.sqrt(252)
    assess['annulized_return'] = all_log_return_mean
    assess['annulized_volatility'] = all_log_return_std
    
    # Sharpe ratio
    all_riskfree_mean = riskfree.mean() * 252
    all_log_return_Sharpe = ( all_log_return_mean - np.array(all_riskfree_mean))/ all_log_return_std
    assess['sharpe'] = all_log_return_Sharpe
    
    # max_drawdown
    MDD_list = template0.copy()
    for s in portfolio_names0:
        temp_list = pnl[s]
        MDD_list[s] = np.max((temp_list.cummax() - temp_list)/temp_list.cummax())
    assess['max_drawdown'] = MDD_list

    # Alpha Beta Information Ratio
    X = log_return['benchmark'].values.reshape(-1,1) - riskfree.values.reshape(-1,1)
    X = sm.add_constant(X,prepend=True)
    Y = log_return.loc[:,portfolio_names1] - riskfree.values.reshape(-1,1)
    beta_list = template1.copy()
    alpha_list = template1.copy()
    sigma_hat_list = template1.copy()
    for s in portfolio_names1:
        model = sm.OLS(Y[s],X)
        res = model.fit()
        alpha_list[s] = res.params[0] 
        beta_list[s] = res.params[1]
        sigma_hat_list[s] = np.sqrt(res.mse_resid)
    alpha_list = alpha_list * 252 
    sigma_hat_list = sigma_hat_list * np.sqrt(252) 
    assess['alpha'] = alpha_list
    assess['beta'] = beta_list
    assess['information_ratio'] = alpha_list / sigma_hat_list
    return assess

def cal_ReturnSeries(assets_series: pd.DataFrame, weight_series: pd.DataFrame) -> pd.Series:
                    #  trade_date: pd.core.indexes.datetimes.DatetimeIndex, 
                    #  transfer_date: pd.core.indexes.datetimes.DatetimeIndex
    trade_date = assets_series.index
    transfer_date = weight_series.index
    simple_return_series = pd.Series(index=trade_date, name='test', dtype='float64')
    
    for i in range(len(transfer_date)):
        if i < len(transfer_date)-1:     
            current_time = transfer_date[i]
            next_time = transfer_date[i+1]
        else:
            current_time = transfer_date[i]
            next_time = trade_date[-1]   
        assets_data = assets_series.loc[current_time:next_time]
        assets_pnl = assets_data / assets_data.iloc[0,:]
        future_portfolio_return = (assets_pnl * np.array(weight_series.loc[current_time]).reshape(1,-1)).sum(axis=1) # 用一次broadcast
        future_portfolio_return = future_portfolio_return / future_portfolio_return.shift(1) - 1
        future_portfolio_return.dropna(inplace=True)
        simple_return_series[future_portfolio_return.index] = future_portfolio_return     
    # zero padding
    simple_return_series.loc[transfer_date[0]] = 0
    simple_return_series.dropna(inplace=True)
    temp = pd.Series(simple_return_series,index=trade_date)
    temp.fillna(0,inplace=True)
    return temp

def StratifiedBackTest(assets_series, factor_series, transfer_date, n=5):
    '''
    单因子分层回测
    '''
    print('回测开始')
    num_assets = assets_series.shape[1]
    num_base = int(num_assets/n)
    portfolios_name = [f'G{i}' for i in range(1,n+1)]
    template = pd.DataFrame(0,index=transfer_date,columns=assets_series.columns, dtype='float64')
    dict_weight_series = {}
    for s in portfolios_name:
        dict_weight_series[s] = template.copy()
    for i in range(len(transfer_date)):
        current_time = transfer_date[i]
        print(current_time)
        factor_current = factor_series.loc[current_time,:]
        for i in range(1,n+1):
            temp_componets = list(factor_current.sort_values(ascending=True)[(i-1)*num_base:i*num_base].index)
            dict_weight_series[f'G{i}'].loc[current_time,temp_componets] = 1 / len(temp_componets)
    print('回测完成, 开始计算收益')
    simple_return_series = pd.concat((cal_ReturnSeries(assets_series,dict_weight_series[s]) for s in dict_weight_series.keys()),axis=1)
    return dict_weight_series

def cal_IC_series(assets_series: pd.DataFrame, factor_series: pd.DataFrame, 
                  transfer_date: pd.DatetimeIndex) -> pd.Series:
    IC_series = pd.DataFrame(columns=['Pearson','pvalue'], dtype='float64')
    for i in range(len(transfer_date)-1):
        current_time = transfer_date[i]
        next_time = transfer_date[i+1]
        factor_current = factor_series.loc[current_time,:]
        next_monthly_return = assets_series.loc[next_time] / assets_series.loc[current_time] - 1
        res = scipy.stats.pearsonr(factor_current,next_monthly_return)
        IC_series.loc[current_time] = np.array(res)
    return IC_series


def cal_turnover(weight_series: pd.DataFrame) -> np.ndarray:
    weight_series_l1 = weight_series.shift(1)
    weight_series_l1.fillna(0,inplace=True)
    _ = np.abs(weight_series-weight_series_l1).mean(axis=0).sum()*250
    return _

# 接下来就剩一个因子库构建了


# 海龟交易系统搭建
# class VirtualAccount(__builtin__.object):
#     def __init__(self):
#         pass
#     self.current_date
#     self.previous_date
    
#     def get_universe(self, asset_type, exclude_halt=False):
#         pass
    
# class RiskAssessment():
#     def __init__(self):
#         pass


# def frequency_convert(multi_series, frequency_list):
#     try:
#         if '__iter__' in dir(frequency_list):
#             frequency_list = list(iter(frequency_list))
#             if multi_series.shape[0] < len(frequency_list):
#                 xx = pd.Series(data=range(len(frequency_list)),index=frequency_list)
#                 temp = pd.concat([multi_series,xx],axis=1)
#                 temp = temp.fillna(method='ffill').dropna()
#                 return temp.loc[:,list(multi_series.columns)]
#             else:
#                 return 0
#         else:
#             print(f'ERROR: {frequency_list} is not iterable.')
#     except:
#         print('请传入DataFrame以及iterator')