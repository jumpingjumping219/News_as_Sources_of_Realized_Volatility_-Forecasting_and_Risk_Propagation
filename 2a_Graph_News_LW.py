# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:49:23 2023

@author: LiyingW

This module applies graph to predict RV at a longer future horizon, by leveraging news attributes 
More details see the methodology in  Graph-based Methods for Forecasting Realized Covariances

Here are three kind of models:

- Graph Model based on News Co-coverage(features are comprised of HAR features and Graph generated features)
    - GHAR+news

- News Propagation through Graph (add news propagation through the graph.)
    - GHAR+newspropogation

- Comprehensive Graph Model based on News (News attributes + News based graph + News propagation)
    - GHAR+NewsPropogationAttribute
"""

import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=1, help="forward-looking period")
parser.add_argument("--horizon", type=int, default=21, help="forecasting horizon", choices=[1, 4, 21])
parser.add_argument("--model_name", type=str, default='GHAR', help="model name", choices=['GHAR', 'GHAR+News', 'GHAR+NewsPropogation', 'GHAR+NewsPropogationAttribute'])
parser.add_argument("--version", type=str, default='Forecast_Var', help="version name")
parser.add_argument("--graph", type=str, default='iden+global+news', help="graph construction")


opt = parser.parse_args()
print(opt)

     
this_version = '_'.join(
    [opt.version,
     opt.model_name,   ####'GHAR+news',news as the attribute ''GHAR+newspropogation'
     opt.graph,  ######################the graph type
     'raw', 
     'W' + str(opt.window), # length of testset
     'F' + str(opt.horizon)]) # 1-day / 1-week / 1-month ahead

para_name = this_version.split('_')[3]
model_name = this_version.split('_')[2]
window=1
perc=0

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

def preprocess_HAR(news):
    # 1 match the date index with real date
    pp = pd.read_csv('./backup/stock_100_news_attributes/predict_panel.csv')
    dd2date = pd.read_csv('./backup/stock_100_news_cocoverage_graph/filename_match.csv').iloc[:,1:]
    dd2datedic = dd2date.set_index('dd')['et_date'].to_dict()
    pp['date'] = pp['dd'].map(dd2datedic)
    print(pp)

    # 2 generate horizon feature(weekly monthly) following the longer future horizons in Graph-based Methods for Forecasting Realized Covariances
    pp, prefix = add_rolling_horizon_features(pp, news, method='mean', window=opt.horizon)
    cols_1m = [col for col in pp.columns if col.startswith(prefix)] + ['date', 'var+lag1', 'var+lag5', 'var+lag22', 'OFTIC']
    pp_1m = pp[cols_1m]

    pp_1m[f'{prefix}Target_log']=np.log(pp_1m[f'{prefix}Target'])/2
    pp_1m[f'{prefix}var+lag1_log']=np.log(pp_1m['var+lag1'])/2
    pp_1m[f'{prefix}var+lag5_log']=np.log(pp_1m['var+lag5'])/2
    pp_1m[f'{prefix}var+lag22_log']=np.log(pp_1m['var+lag22'])/2
    pp_1m = pp_1m.dropna()
    pp_1m.drop(columns=['var+lag1', 'var+lag5', 'var+lag22',f'{prefix}Target'], inplace=True)
    
    df = pp_1m.copy()
    tickers=df['OFTIC'].unique()
    dd_l = list(set(df['dd'].tolist()))
    dd_l.sort()
    
    df=df.sort_values(by=['OFTIC','dd'])
    df.rename(columns={'OFTIC':'Ticker'},
              inplace=True)
    print(df.columns)
    # clms = [i for i in df.columns if i not in ['Date', 'Ticker']]
    subdf_dic = {}
    for dd in dd_l:
        subdf = df[df['dd'] == dd]
        subdf_dic[dd] = subdf
                                                        
    print('Finish preparation!')
    return subdf_dic, dd_l,tickers, prefix

def add_rolling_horizon_features(df, columns, method='mean', window=21):
    horizon = [1, 4, 21, 63]
    nxt_horizon = horizon[horizon.index(window) + 1]
    print(nxt_horizon)

    df = df.copy()
    df = df.sort_values(['Ticker', 'date'])
    if window == 21:
        prefix='1m_'
    elif window == 4:
        prefix = '1w_'
    elif window  == 1:
        prefix = '1d_'
        return df, prefix


    df[f'{prefix}Target'] = (
    df.groupby('Ticker')['Target']
      .transform(lambda s: s.rolling(window=window, min_periods=window).sum().shift(-(window - 1)))
    )

    # 2) 1m_Num_News_shock ：(近21日日均) - (近63日日均)
    news_1_avg = (
        df.groupby('Ticker')['Num_News']
        .transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    )


    news_2_avg = (
        df.groupby('Ticker')['Num_News']
        .transform(lambda s: s.rolling(window=nxt_horizon, min_periods=nxt_horizon).mean())
    )
    df[f'{prefix}_Num_News_shock'] = news_1_avg - news_2_avg

    # 3) 1m_CSS_shock ：(近21日日均) - (近63日日均)
    css_1_avg = (
        df.groupby('Ticker')['css_mean']
        .transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    )
    css_2_avg = (
        df.groupby('Ticker')['css_mean']
        .transform(lambda s: s.rolling(window=nxt_horizon, min_periods=nxt_horizon).mean())
    )
    df['1m_CSS_shock'] = css_1_avg - css_2_avg
    df['1m_CSS_shock_absolute'] = np.abs(df['1m_CSS_shock'])


    # 其他的 news feature 如 Num_offmarket_News 直接 mean
    for col in columns:
        if method == 'sum':
            df[f'{prefix}{col}'] = (
                df.groupby('Ticker')[col]
                  .transform(lambda s: s.rolling(window, min_periods=window).sum())
            )
        elif method == 'mean':
            df[f'{prefix}{col}'] = (
                df.groupby('Ticker')[col]
                  .transform(lambda s: s.rolling(window, min_periods=window).mean())
            )
    
    return df, prefix



def preprocess_adj_l(dd_l,ret_df,subdf_dic,tickers,prefix):#################generate the regressors
    """
    Add Graph features
    :param date_l:
    :param subdf_dic:
    :param adj_df_l:
    :return:
    """
    
    n = len(tickers)
    # 1 obtain graph types
    para_name_l = this_version.split('_')[3].split('+')
        
    new_subdf_l = []
    for dd in dd_l[5:]:
        timestamp = dd_l.index(dd)  ###date_1 is the dd list for all the predicting values
        # split time
        s_p = max(timestamp-1000, 0)  #starting point
        s_date = dd_l[s_p]
        subret = ret_df[ret_df.index < dd]
        subret = subret[subret.index >= s_date]
        subret =subret[tickers]
        
        subdf = subdf_dic[dd]
        # print(subdf)
        tmp_subdf_l = []
    
        #################################################add new content##############
        #####################careful!
        if this_version.split('_')[2]=='GHAR':
            clms = [i for i in subdf.columns if 'lag' in i and 'log' in i]####lag and log should in i, include all the needed feature inside
        if this_version.split('_')[2]=='GHAR+News':####news as the attribute withtout propogation
            clms = [i for i in subdf.columns if 'lag' in i and 'log' in i]
        elif this_version.split('_')[2]=='GHAR+NewsPropogation' or this_version.split('_')[2]=='GHAR+NewsPropogationAttribute': ###using the attribute of news to propogate and attribute
            clms = [i for i in subdf.columns if ('lag' in i and 'log' in i) or i==f'{prefix}Num_News_shock' or i==f'{prefix}num_transfer_market' or i==f'{prefix}Num_scheduled_News']
        else:
            clms = [i for i in subdf.columns if 'lag' in i and 'log' in i]
        print(clms)
        
        adj_df_l = []
        for para_name in para_name_l:
            if para_name == 'iden':
                adj_df = pd.DataFrame(np.identity(n), index=tickers, columns=tickers)
            elif para_name == 'global':
                adj_df = pd.DataFrame(np.ones((n, n)), index=tickers, columns=tickers)
                adj_df -= np.identity(n)
                adj_df /= adj_df.sum()
            elif para_name == 'corret':
                adj_df = Corr_adj(subret)
            elif para_name == 'glasso':
                #adj_df = GLASSO_Precision(subret)
                adj_df = pd.read_csv(join('./backup/stock_100_glasso_graph/%d.csv' % dd), index_col=0)
            elif para_name == 'news':
                adj_df = load_adjacency_news(dd)
            else:
                adj_df = pd.DataFrame(np.zeros((n, n)), index=tickers, columns=tickers)

            adj_df_l.append(adj_df)
            
            
        for k, adj_df in enumerate(adj_df_l):
            print(adj_df)
            # add graph feature
            tmp_subdf = pd.DataFrame(np.dot(adj_df, subdf[clms]), columns=['sec'+str(k)+i for i in clms], index=subdf.index)
            tmp_subdf_l.append(tmp_subdf)
        
        # if the model is 'GHAR+NewsPropogationAttribute', we still need to add raw news feature
        new_subdf = pd.concat([subdf[[f'{prefix}Target_log', 'dd', 'Ticker',f'{prefix}Num_News_shock',f'{prefix}num_transfer_market',f'{prefix}Num_scheduled_News']]]+tmp_subdf_l, axis=1)
        new_subdf_l.append(new_subdf)        ###########################include mor econtents in the 

    df = pd.concat(new_subdf_l)
    df.reset_index(drop=True, inplace=True)
    print('Finish transformation!')
    return df


def load_adjacency_news(dd):#to predict dd
    """
    TODO: 1. binary graph via threshold
    :param date:
    :return:
    """
    #file_df = pd.read_csv(join('E:\filename_match.csv'), index_col=0)
    #dd_idx = file_df[file_df['et_date'] == date]['dd'].values[0]
    adj_df = pd.read_csv(join('E:/stock_100_news_cocoverage_graph/%d.csv' % dd), index_col=0)
    adj_df = adj_df.sort_index(axis=1)
    d_sqrt_inv = np.diag(np.sqrt(1/(adj_df.sum(1)+1e-8)))
    adj_df = pd.DataFrame(np.dot(np.dot(d_sqrt_inv, adj_df), d_sqrt_inv), index=adj_df.index, columns=adj_df.columns)
    return adj_df



def Corr_adj(subret):
    n = subret.shape[1]
    corr = subret.corr()
    if perc == 0:
        corr_adj = corr - np.identity(n)
    else:
        corr -= np.identity(n)
        thr = np.percentile(corr.values, perc)
        corr_adj = (corr > thr).astype(float)
    # corr_adj /= (corr_adj.sum(1) + 1e-8)
    d_sqrt_inv = np.diag(np.sqrt(1/(corr_adj.sum()+1e-8)))
    adj_df = pd.DataFrame(np.dot(np.dot(d_sqrt_inv, corr_adj), d_sqrt_inv), columns=corr.columns, index=corr.index)
    return adj_df


def GLASSO_Precision(subret):
    from sklearn.covariance import GraphicalLassoCV
    n = subret.shape[1]
    tickers = subret.columns
    cov = GraphicalLassoCV().fit(subret)
    print('Alpha in GLASSO: %.3f' % cov.alpha_)
    inv_cov = cov.precision_ != 0
    print('Sparsity of Adj: %.3f' % inv_cov.mean())
    corr_adj = inv_cov - np.identity(n)
    # adj_df = pd.DataFrame(corr_adj / (corr_adj.sum(1)[:, np.newaxis] + 1e-8), columns=tickers, index=tickers)
    d_sqrt_inv = np.diag(np.sqrt(1/(corr_adj.sum(1)+1e-8)))
    adj_df = pd.DataFrame(np.dot(np.dot(d_sqrt_inv, corr_adj), d_sqrt_inv), columns=tickers, index=tickers)
    return adj_df


def df2arr(df, vars_l):
    all_inputs = df[vars_l].values
    all_targets = df['Target_log'].values
    return all_inputs, all_targets


def Train(ret_df, var_df,df,tickers, subdf_dic, dd, dd_l):####all the information has been included in the subdf_dic
    timestamp = dd_l.index(dd)  ###date_1 is the dd list for all the predicting values
    # split time
    s_p = max(timestamp-1000, 0)  #starting point
    f_p = min(timestamp + window, len(dd_l)-1)
    s_date = dd_l[s_p]
    f_date = dd_l[f_p]

    #subret = ret_df[ret_df['dd'] < dd]
    #subret = subret[subret['dd'] >= s_date]

    #subdata = df[df['dd'] < dd]
    #subdata = df[df['dd'] >= s_date]  ###df is the tranformed log

    ##########################based on the news graph##################
    
    tickers.sort()

    #df = preprocess_adj_l(dd_l[s_p:f_p+1],ret_df, subdf_dic,tickers)#############change
    # df = np.log(df)
    if this_version.split('_')[2]=='GHAR+News' or this_version.split('_')[2]=='GHAR+NewsPropogationAttribute':
        vars_l = [i for i in df.columns if 'lag' in i or i=='Num_News_shock' or i=='num_transfer_market' or i=='Num_scheduled_News']
    else:
        vars_l = [i for i in df.columns if 'lag' in i]
    # split data
    train_df = df[df['dd'] >= s_date]
    train_df = train_df[train_df['dd'] < dd]
    #print(train_df)

    test_df = df[df['dd'] >= dd]
    test_df = test_df[test_df['dd'] < f_date]
    #print(test_df)

    train_x, train_y = df2arr(train_df, vars_l)
    test_x, test_y = df2arr(test_df, vars_l)

    best_model = LinearRegression()
    best_model.fit(train_x, train_y)
    print(best_model.coef_)

    test_pred_df = test_df[['Ticker', 'dd']]
    test_pred_df['Pred_VHAR'] = best_model.predict(test_x)
    test_pred_df = test_pred_df.pivot(index='dd', columns='Ticker', values='Pred_VHAR')

    test_pred_df.columns = list(test_pred_df.columns)
    test_pred_df.index = list(test_pred_df.index)

    save_path = join('./backup/Results_Backup/News_Var_20250729/News_Var_Results', this_version)
    os.makedirs(save_path, exist_ok=True)

    test_pred_df.to_csv(join(save_path, 'Pred_%s.csv' % dd))


def connect_pred():
    save_path = join('./backup/Results_Backup/News_Var_20250729/News_Var_Results', this_version)
    files_l = os.listdir(save_path)
    pred_files = [i for i in files_l if 'Pred_' in i]
    pred_files.sort()
    test_pred_df_l = []
    for i in pred_files:
        test_pred_df = pd.read_csv(join(save_path, i), index_col=0)
        test_pred_df_l.append(test_pred_df)

    test_pred_df = pd.concat(test_pred_df_l)
    test_pred_df=test_pred_df.rename(columns={'Unnamed: 0':'pred_dd'})
    #print(test_pred_df)

    sum_path = join('./backup/Results_Backup/News_Var_20250729/Var_Results_Sum')
    os.makedirs(sum_path, exist_ok=True)
    test_pred_df.to_csv(join(sum_path, this_version + '_pred.csv'))
    
    

if __name__ == '__main__':
    para_name = this_version.split('_')[3]
    model_name = this_version.split('_')[2]
    save_path = join('./backup/Results_Backup/News_Var_20250729/News_Var_Results', this_version)
    hf_data_stock_65min=pd.read_csv('E:/Data_100stock_rv/hf_data_stock_65min.csv')
    news_df=pd.read_csv('E:/stock_100_news_attributes/NEWS_attribute_100stocks.csv',index_col=None) 
    stock_df=hf_data_stock_65min.copy()
    
    filer_news = ['Num_scheduled_News', 'Num_unscheduled_News',
        'CSS_abs_mean', 'abs_EVENT_SENTIMENT_SCORE', 'css_std', 'SENTIMENT_std',
        'SENTIMENT_mean', 'SENTIMENT_mean_absolute', 'css_tone_absolute',
        'media_coverage', 'SENTIMENT_offmarket', 'SENTIMENT_onmarket',
        'Num_onmarket_News', 'Num_offmarket_News', 'sentiment_transfer_market',
        'sentiment_transfer_market_abs', 'num_transfer_market']
    
    subdf_dic,dd_l,tickers,prefix = preprocess_HAR(filer_news)

    var_df, ret_df = load_data(tickers,stock_df)
    df=preprocess_adj_l(dd_l,ret_df,subdf_dic,tickers)
    print(df.columns)
    print('Training Starts Now ...')
    start_date = 5266
    idx = dd_l.index(start_date)
    window=1
    for dd in dd_l[idx:-1:window]:
        print(' * ' * 20 + str(dd) + ' * ' * 20)
        Train(model_name, ret_df, var_df,df,tickers, subdf_dic, dd, dd_l)
    
    connect_pred()