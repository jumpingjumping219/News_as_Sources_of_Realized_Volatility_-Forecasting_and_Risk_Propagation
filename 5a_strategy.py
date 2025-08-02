"""
This module aims to compare the return spreads  
of the variance risk premium (VRP) strategy based on different RV forecasting models.

input: 

output:
intraday strategy
overnight strategy
"""
import os
from os.path import *

import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def load_data(tickers,stock_df):
    data = stock_df.copy()   #
    data['base']=pd.to_datetime('2000-01-30')
    data['date']=data['Time'].apply(lambda x:x[0:10])
    data['date'] = pd.to_datetime(data['date'].astype(str), format='%Y-%m-%d')
    data.day=(data['date']-data['base'])
    data['dd']=data.day.dt.days
    data.fillna(method="ffill", inplace=True)
    
    #vol_ticker_list=[i for i in list(set([i[0:-4] for i in data.columns.tolist()[1:]])) if i!='']
    #vol_ticker_list.sort()
    
    #ticker_list=[i for i in vol_ticker_list if i in news_ticker_list]

    #daily_rv=stock_df.groupby(['dd']).sum().reset_index()
    
    #daily_first_rv=stock_df.drop_duplicates(subset='dd',keep='first')
    
    daily_stock=data.groupby(['dd']).sum().reset_index()
    #daily_rv = daily_stock[[col for col in daily_stock.columns if col.endswith('vol') or col.endswith('dd')]]
    #daily_rv.set_index('dd', inplace=True)
    
    
    daily_stock.set_index('dd', inplace=True)
    #data = data[data.index <= '2021-07-01']
    #data = data[data.index >= '2011-06-30']
    daily_stock = daily_stock.sort_index(axis=1)

    for clm in daily_stock.columns:
        max_p = np.percentile(daily_stock[clm], 99)
        min_p = np.percentile(daily_stock[clm], 1)

        daily_stock.loc[daily_stock[clm] > max_p, clm] = max_p
        daily_stock.loc[daily_stock[clm] < min_p, clm] = min_p

        
    #daily_rv = daily_stock[[col for col in daily_stock.columns if col.endswith('vol') or col.endswith('dd')]]
    #daily_rv.set_index('dd', inplace=True)
    
    
    vol_clms = [i for i in daily_stock.columns if '_vol' in i]
    var_df = daily_stock[vol_clms] * 1e4
    var_df.columns = [i[:-4] for i in vol_clms]
    var_df = var_df[tickers]
    ret_clms = [i for i in daily_stock.columns if '_ret' in i]
    ret_df = daily_stock[ret_clms] * 1e2
    ret_df.columns = [i[:-4] for i in ret_clms]
    ret_df = ret_df[tickers]
    return var_df, ret_df


def load_prediction_data(version_name, data_version, horizon):
    result_files = [i for i in files if version_name in i and '_pred' in i and data_version in i]
   # result_files = [i for i in result_files if 'Meta' not in i and 'Sig' not in i]
    #result_files = [i for i in result_files if 'scheduled' not in i and '__' not in i]

    result_files.sort()
    for (i, item) in enumerate(result_files):
        print(i, item)

    mdldict = {}
    for filename in result_files:
        test_pred_df = pd.read_csv(join(sum_path, filename), index_col=0) ####the first column is the 
        test_pred_df = test_pred_df.sort_index(axis=1)
        test_pred_df=test_pred_df[tickers.tolist()]
        print(filename)

        # check the prediction carefully
        test_pred_df=np.exp(2*test_pred_df) 
        # transform to annualized monthly expected realized variance from forecasting model m. depends on the forecasting horizon
        test_pred_df=12* test_pred_df

        test_pred_df[test_pred_df<=0] = np.nan
        test_pred_df = test_pred_df.ffill()
        mdldict[filename] = test_pred_df
    
    return mdldict


def build_vrp(pred_df, iv_df):
    dates = pred_df.index.intersection(iv_df.index)
    tickers = pred_df.columns.intersection(iv_df.columns)
    p = pred_df.loc[dates, tickers]
    iv = iv_df.loc[dates, tickers]

    return iv - p

def VRP_strategy(vrp, ret, H = 21):
    """
    output:
    index: time
    cols=['intraday_long', 'intraday_short','intraday_spread',
        'overnight_long','overnight_short','overnight_spread']
    """
    dates   = vrp.index
    T, N    = len(dates), vrp.shape[1]
    columns = vrp.columns

    rec = []
    for i in range(0, T, H):
        reb_date = dates[i]           
        end_idx  = min(i+H, T)       
        period   = dates[i:end_idx]

        # 1) 当日分位信号
        today = vrp.loc[reb_date]
        q_low, q_high = today.quantile([0.1, 0.9])
        longs  = today[today>=q_high].index
        shorts = today[today<=q_low ].index

        # 2) 拆日内/隔夜：用 ret_df

        intraday = ret.loc[period, :]
        overnight = pd.DataFrame(0.0, index=period, columns=columns)

        # 3) 组合等权
        long_ret_in  = intraday[longs].mean(axis=1)
        short_ret_in = intraday[shorts].mean(axis=1)
        spread_in    = long_ret_in - short_ret_in

        long_ret_ov  = overnight[longs].mean(axis=1)
        short_ret_ov = overnight[shorts].mean(axis=1)
        spread_ov    = long_ret_ov - short_ret_ov

        df = pd.DataFrame({
            'intraday_long':  long_ret_in,
            'intraday_short': short_ret_in,
            'intraday_spread':spread_in,
            'overnight_long':  long_ret_ov,
            'overnight_short': short_ret_ov,
            'overnight_spread':spread_ov,
        })
        rec.append(df)

    return pd.concat(rec)


def statistics(ret):
    summary = {}
    for model, df in ret.items():
        rec = {}
        for name in ['intraday_spread','overnight_spread']:
            r = df[name].dropna()
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            rec[f"{name}_Sharpe"] = ann_ret / ann_vol if ann_vol>0 else np.nan
        summary[model] = rec

    sharpe_df = pd.DataFrame(summary).T  
    return sharpe_df

def plot(ret):
    plt.figure(figsize=(10,5))
    for model, df in strat_returns.items():
        plt.plot(df['intraday_spread'].cumsum(),  label=f"{model} intraday strategy")
        plt.plot(df['overnight_spread'].cumsum(), label=f"{model} overnight strategy", linestyle='--')

    plt.legend()
    plt.title("VRP")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    horizon = 21
    data_version = 'raw'
    # hf_data_stock_65min=pd.read_csv('E:/Data_100stock_rv/hf_data_stock_65min.csv')
    ana_path = './backup/Results_Backup/News_Var_20250729/Results_Analysis'
    sum_path = join('./backup/Results_Backup/News_Var_20250729/Var_Results_Sum')
    # stock_df=hf_data_stock_65min.copy()
    tickers =np.array(['AAPL', 'ABT', 'ACN', 'ADBE', 'ADP', 'AMGN', 'AMT', 'AMZN', 'AVGO',
       'AXP', 'BA', 'BAC', 'BDX', 'BLK', 'BMY', 'BRK.B', 'BSX', 'C',
       'CAT', 'CB', 'CCI', 'CI', 'CL', 'CMCSA', 'CME', 'COP', 'COST',
       'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DHR', 'DUK', 'FIS',
       'FISV', 'GE', 'GILD', 'GOOG', 'GS', 'HD', 'HON', 'IBM', 'INTC',
       'INTU', 'ISRG', 'JNJ', 'JPM', 'KO', 'LLY', 'LMT', 'LOW', 'MA',
       'MCD', 'MDT', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU',
       'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'PNC',
       'QCOM', 'SBUX', 'SCHW', 'SO', 'SYK', 'T', 'TGT', 'TJX', 'TXN',
       'UNH', 'UNP', 'UPS', 'USB', 'V', 'VRTX', 'VZ', 'WFC', 'WMT', 'XOM'])
    
    
    hf_data_stock_65min=pd.read_csv('.back_up/Data_100stock_rv/hf_data_stock_65min.csv')
    stock_df=hf_data_stock_65min.copy()

    # 1 load return data
    var_df, ret_df = load_data(tickers,stock_df)

    # 2 obtain all models prediction file
    files = os.listdir(sum_path)
    files.sort()
    pred_dfs = load_prediction_data('Forecast_Var', data_version, horizon)

    # 3 load IV
    IV = pd.read_csv('./backup/stock_100_news_attributes/1m_IV.csv')
    IV['date'] = pd.to_datetime(IV['date'], format="%d%b%Y")
    P_IV = IV.pivot(
        index='date', columns='ticker', values = 'impl_volatility'
    )

    # 4 constuct VRP and perform , H represents the rebalace frequency
    vrp_dfs = {m: build_vrp(pred, P_IV) for m, pred in pred_dfs.items()}
    strat_returns = {
        model: VRP_strategy(vrp_dfs[model], ret_df.loc[vrp_dfs[model].index], H = 21) for model in vrp_dfs
    }

    print(statistics(strat_returns))
    plot(strat_returns)
