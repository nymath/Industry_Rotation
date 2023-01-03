import pandas as pd
import numpy as np
import statsmodels.api as sm
import os 




refresh_rate_map = {
    'Monthly(-1)': 'is_monthend',
    'Weekly(-1)': 'is_weekend',
    'Yearly(-1)': 'is_yearend',
}

def get_TradeDate(start_date: str, 
                  end_date: str) -> pd.core.indexes.datetimes.DatetimeIndex:
    try:
        test = pd.to_datetime(np.array(np.load(f'{rawpath}/trading_dates.npy'),dtype='U8'))
    except:
        '数据源路径不存在, 请输入rqalpha download-bundle安装'
    trade_dates = pd.Series(0,index=test)[start_date:end_date].index
    return trade_dates


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
    all_log_return_Sharpe = ( all_log_return_mean - all_riskfree_mean.values ) / all_log_return_std
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
    # Turnover Rate
    return assess

def cal_turnover(weight_series: pd.DataFrame) -> np.ndarray:
    weight_series_l1 = weight_series.shift(1)
    weight_series_l1.fillna(0,inplace=True)
    _ = np.abs(weight_series-weight_series_l1).mean(axis=0).sum()*250
    return _




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