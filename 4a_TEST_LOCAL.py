# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:24:11 2023

@author: Liying
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 12:33:07 2023

@author: LiyingW
"""

#The structure of testing###################

import os
from os.path import *
from MCS import *

import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy import stats
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

#raw_net = 'raw'

#stock_df=hf_data_stock_65min.copy()


def load_data(tickers,stock_df):
    data = stock_df.copy()   #
    data['base']=pd.to_datetime('2000-01-30')
    data['date']=data['Time'].apply(lambda x:x[0:10])
    data['date'] = pd.to_datetime(data['date'].astype(str), format='%Y-%m-%d')
    data['day']=(data['date']-data['base'])
    data['dd']=data['day'].dt.days
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


#def load_adjacency_news(date):
#    file_df = pd.read_csv(join('/data01/Chao_ESG_News/SP100_CoCoverage_Graphs/filename_match.csv'), index_col=0)
#    dd_idx = file_df[file_df['et_date'] == date]['dd'].values[0]
#    adj_df = pd.read_csv(join('/data01/Chao_ESG_News/SP100_CoCoverage_Graphs/%d.csv' % dd_idx), index_col=0)
#    adj_df = adj_df.sort_index(axis=1)
#    d_sqrt_inv = np.diag(np.sqrt(1/(adj_df.sum(1)+1e-8)))
#    adj_df = pd.DataFrame(np.dot(np.dot(d_sqrt_inv, adj_df), d_sqrt_inv), index=adj_df.index, columns=adj_df.columns)
#    return adj_df


def QLIKE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


def Loss(var_df, test_pred_df):   #test_pred_df should have the index
    test_df = var_df.loc[test_pred_df.index]
    test_df = test_df.ffill()

    #####################################
    ###transform######################
    
    ticker_l = var_df.columns.tolist()
    test_pred_df.columns = ticker_l
    df_l = []

    for ticker in ticker_l:
        y_true = test_df[ticker].values
        y_pred = test_pred_df[ticker].values
        assert (y_pred > 0).all()
        mse = mean_squared_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        qlike = QLIKE(y_true, y_pred)

        df_l.append([np.round(mse, 4), np.round(mape, 4), np.round(qlike, 4)])

    df = pd.DataFrame(np.array(df_l), index=ticker_l, columns=['MSE', 'MAPE', 'QLIKE'])
    return df


#Forecast_Var_HAR_firm_reduced_firm-levelraw_W1_F1

def Result(var_df, version_name, data_version, horizon):
    #result_files = [i for i in files if
    #                ('_pred' in i) and version_name in i and '_' + data_name + '_' in i and data_version in i]# and 'F%d' % horizon in i and 'W22' in i
    result_files = [i for i in files if version_name in i and '_pred' in i and data_version in i]
   # result_files = [i for i in result_files if 'Meta' not in i and 'Sig' not in i]
    #result_files = [i for i in result_files if 'scheduled' not in i and '__' not in i]

    result_files.sort()
    for (i, item) in enumerate(result_files):
        print(i, item)

    E_df_l = []
    F_df_l = []
    Q_df_l = []
    files_l = []


    for filename in result_files:
        test_pred_df = pd.read_csv(join(sum_path, filename), index_col=0) ####the first column is the 
        test_pred_df = test_pred_df.sort_index(axis=1)
        test_pred_df=test_pred_df[tickers.tolist()]############################
        print(filename)
        # print(test_pred_df)
        #######converting  the test_pred_df is log()/2
        # test_pred_df=np.exp(2*test_pred_df)*1e4 ####################
        test_pred_df=np.exp(2*test_pred_df) ####################
        test_pred_df[test_pred_df<=0] = np.nan
        test_pred_df = test_pred_df.ffill()
        print(test_pred_df.isna().sum().sum())


        df = Loss(var_df, test_pred_df)
        E_df_l.append(df['MSE'])
        F_df_l.append(df['MAPE'])
        Q_df_l.append(df['QLIKE'])

        file_key_name = filename.split('_')[2] + '_' + filename.split('_')[3]
        # file_key_name = filename
        files_l.append(file_key_name)

    E_df = pd.concat(E_df_l, axis=1)
    E_df.columns = files_l
    F_df = pd.concat(F_df_l, axis=1)
    F_df.columns = files_l
    Q_df = pd.concat(Q_df_l, axis=1)
    Q_df.columns = files_l

    return files_l,result_files, E_df, F_df, Q_df



def norm_loss(df):
    return df.apply(lambda x: x/df['HAR_Nonews'], axis=0)


def rank_MCS(loss_df, pval_df,files_l):
    loss_mean_df = loss_df.mean(0)
    rank_df = loss_mean_df.rank()
    pval_df = pd.DataFrame(pval_df, columns=['p-value'])
    pval_df['loss'] = loss_mean_df
    pval_df['ratio'] = loss_mean_df / loss_mean_df.loc['HAR_Nonews']
    pval_df['rank'] = rank_df
    idx_l = files_l#['GHAR+NewsPropagation_iden+global+glasso+news','GHAR_iden+global+glasso+news','GHAR+NewsPropagation_iden+global+news','GHAR_iden+global+news','GHAR+NewsPropagation_iden+glasso+news','GHAR_iden+glasso+news','GHAR+NewsPropagation_iden+global','GHAR_iden+glasso']
    
    #files_l#['HAR_stocks', 'HAR+FirmNews_stocks'] #####################################change point!!!!!!!!!!!!!!!!!!!!!!!

    return pval_df.loc[idx_l, ['ratio', 'p-value']]


if __name__ == '__main__':
    horizon = 1
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
    # var_df, ret_df = load_data(tickers,stock_df)

    # 1 obtain true realized volatility
    var_1m_df = pd.read_csv('./backup/stock_100_news_attributes/1m_predict_panel.csv')
    var_1m_df = var_1m_df[['OFTIC', 'date', '1m_Target']]
    var_df = var_1m_df.pivot(columns='OFTIC', index='date', values='1m_Target')

    # 2 obtain all models prediction file
    files = os.listdir(sum_path)
    files.sort()
    
    # 3 generate results
    files_l,result_files, E_df, F_df, Q_df = Result(var_df, 'Forecast_Var', data_version, horizon)
    
    print(' * ' * 30 + 'Un-Normalized')
    print(E_df.mean(0))
    pd.DataFrame(E_df.mean(0).sort_values().rename('MSE')).to_csv(join(ana_path, 'Un-Normalized' + '_MSE.csv'))
    print(Q_df.mean(0))
    pd.DataFrame(Q_df.mean(0).sort_values().rename('QLIKE')).to_csv(join(ana_path, 'Un-Normalized' + '_QLIKE.csv'))
    

    mcs_E = ModelConfidenceSet(E_df, 0.05, 10000, 2).run()
    sum_E = rank_MCS(E_df, mcs_E.pvalues,files_l)

    mcs_Q = ModelConfidenceSet(Q_df, 0.05, 10000, 2).run()
    sum_Q = rank_MCS(Q_df, mcs_Q.pvalues,files_l)

    print(" * * * * * MCS of MSE * * * * * ")
    print(np.round(sum_E, 3))
    pd.DataFrame(np.round(sum_E, 3)).to_csv(join(ana_path, 'Un-Normalized' + '_MCS_of_MSE.csv'))


    print(" * * * * * MCS of QLIKE * * * * * ")
    print(np.round(sum_Q, 3))
    pd.DataFrame(np.round(sum_Q, 3)).to_csv(join(ana_path, 'Un-Normalized' + '_MCS_of_QLIKE.csv'))
    
    

    print(' * ' * 30 + 'Normalized')
    print(norm_loss(E_df).mean(0).sort_values())
    pd.DataFrame(norm_loss(E_df).mean(0).sort_values().rename('MSE')).to_csv(join(ana_path, 'Normalized' + '_MSE.csv'))
    print(norm_loss(Q_df).mean(0).sort_values())
    pd.DataFrame(norm_loss(Q_df).mean(0).sort_values().rename('QLIKE')).to_csv(join(ana_path, 'Normalized' + '_QLIKE.csv'))
    mcs_E = ModelConfidenceSet(norm_loss(E_df), 0.05, 10000, 2).run()
    sum_E = rank_MCS(norm_loss(E_df), mcs_E.pvalues,files_l)

    mcs_Q = ModelConfidenceSet(norm_loss(Q_df), 0.05, 10000, 2).run()
    sum_Q = rank_MCS(norm_loss(Q_df), mcs_Q.pvalues,files_l)

    print(" * * * * * MCS of MSE * * * * * ")
    print(np.round(sum_E, 3))
    pd.DataFrame(np.round(sum_E, 3)).to_csv(join(ana_path, 'Normalized' + '_MCS_of_MSE.csv'))


    print(" * * * * * MCS of QLIKE * * * * * ")
    print(np.round(sum_Q, 3))
    pd.DataFrame(np.round(sum_Q, 3)).to_csv(join(ana_path, 'Normalized' + '_MCS_of_QLIKE.csv'))
    
    
    