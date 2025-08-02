"""
This module applies HAR, Lasso, Ridge to  predict RV at a longer future horizon, by leveraging news attributes 
More details see the methodology in  Graph-based Methods for Forecasting Realized Covariances
"""

import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=1, help="forward-looking period")
parser.add_argument("--horizon", type=int, default=21, help="forecasting horizon", choices=[1, 4, 21])
parser.add_argument("--model_name", type=str, default='HAR', help="model name", choices=['HAR', 'RIDGE', 'LASSO'])
parser.add_argument("--version", type=str, default='Forecast_Var', help="version name")
parser.add_argument("--news", type=str, default='FullNews', help="news", choices=['Nonews', 'FullNews', 'SelectedNews'])


opt = parser.parse_args()
print(opt)

# Specific version
this_version = '_'.join(
    [opt.version,
     opt.model_name,
     opt.news, 
     'raw', 
     'W' + str(opt.window), # length of testset
     'F' + str(opt.horizon)]) # 1-day / 1-week / 1-month ahead

def load_data(news):
    # 1 match the date index with real date
    pp = pd.read_csv('./backup/stock_100_news_attributes/predict_panel.csv')
    dd2date = pd.read_csv('./backup/stock_100_news_cocoverage_graph/filename_match.csv').iloc[:,1:]
    dd2datedic = dd2date.set_index('dd')['et_date'].to_dict()
    pp['date'] = pp['dd'].map(dd2datedic)
    print(pp)

    # 2 generate horizon feature(weekly monthly) following the longer future horizons in Graph-based Methods for Forecasting Realized Covariances
    pp, prefix = add_rolling_horizon_features(pp, news, method='mean', window=opt.horizon)
    if opt.news == 'Nonews':
        cols_1m = ['date', 'var+lag1', 'var+lag5', 'var+lag22', 'OFTIC', f'{prefix}Target']

    elif opt.news == 'SelectedNews':
        selectnews = ['Num_News_shock', 'num_transfer_market', 'Num_scheduled_News', 'Num_unscheduled_News', 'abs_EVENT_SENTIMENT_SCORE', 'SENTIMENT_std',
                      'media_coverage', 'sentiment_transfer_market_abs']
        selectnews = [prefix+i for i in selectnews]
        print(selectnews)
        cols_1m = ['date', 'var+lag1', 'var+lag5', 'var+lag22', 'OFTIC', f'{prefix}Target']

    elif opt.news == 'FullNews':
        cols_1m = [col for col in pp.columns if col.startswith(prefix)] + ['date', 'var+lag1', 'var+lag5', 'var+lag22', 'OFTIC']

    pp_1m = pp[cols_1m]
    # 因为引入 lasso 对 feature 的数量级有要求 这里对 var 进行 log 转换
    pp_1m[f'{prefix}Target_log']=np.log(pp_1m[f'{prefix}Target'])/2
    pp_1m[f'{prefix}var+lag1_log']=np.log(pp_1m['var+lag1'])/2
    pp_1m[f'{prefix}var+lag5_log']=np.log(pp_1m['var+lag5'])/2
    pp_1m[f'{prefix}var+lag22_log']=np.log(pp_1m['var+lag22'])/2
    pp_1m = pp_1m.dropna()

    pp_1m.drop(columns=['var+lag1', 'var+lag5', 'var+lag22',f'{prefix}Target'], inplace=True)
    print('Finish Transformation!')
    print(pp_1m)
    print(pp_1m.columns)
    return pp_1m


def add_rolling_horizon_features(df, columns, method='mean', window=opt.horizon):
    horizon = [1, 4, 21, 63]
    nxt_horizon = horizon[horizon.index(window) + 1]
    print(nxt_horizon)

    df = df.copy()
    df = df.sort_values(['Ticker', 'date'])
    if window == 21:
        prefix='1m_'
    elif window == 4:
        prefix = '1w_'
    # if tries to predict 1-day ahead, no need to conduct feature transformation
    elif window == 1:
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



def df2arr(df, vars_l, prefix = '1m_'):
    all_inputs = df[vars_l].values
    all_targets = df[f'{prefix}Target_log'].values
    return all_inputs, all_targets


def Train(model_name, df, date, date_l):
    timestamp = date_l.index(date)
    # split time
    s_p = max(timestamp-1000, 0)
    f_p = min(timestamp + opt.window, len(date_l)-1)

    s_date = date_l[s_p]
    f_date = date_l[f_p]

    # split data
    train_df = df[df['date'] >= s_date]
    train_df = train_df[train_df['date'] < date]
    # print(train_df)

    test_df = df[df['date'] >= date]
    test_df = test_df[test_df['date'] < f_date]
    if test_df.empty:                 
        return pd.DataFrame()

    vars_l = [i for i in df.columns if i not in ['1m_Target_log', 'date', 'OFTIC']]

    train_x, train_y = df2arr(train_df, vars_l)
    test_x, test_y = df2arr(test_df, vars_l)

    if model_name == 'RIDGE':
        best_model = RidgeCV()
    elif model_name == 'LASSO':
        best_model = LassoCV(max_iter=20000)
    else:
        best_model = LinearRegression()

    best_model.fit(train_x, train_y)
    # print(best_model.coef_)

    test_pred_df = test_df[['OFTIC', 'date']]
    test_pred_df['Pred_VHAR'] = best_model.predict(test_x)

    test_pred_df = test_pred_df.pivot(index='date', columns='OFTIC', values='Pred_VHAR')

    test_pred_df.columns = list(test_pred_df.columns)
    test_pred_df.index = list(test_pred_df.index)
    # print('Before: %.3f' % test_pred_df.min().min())

    # for clm in test_pred_df.columns:
    #     clm_pred_df = test_pred_df[clm]
    #     clm_train_df = train_df[train_df['OFTIC'] == clm]['1m_Target_log']
    #     clm_pred_df[clm_pred_df <= 0] = clm_train_df.min()
    #     test_pred_df[clm] = clm_pred_df

    # print('After: %.3f' % test_pred_df.min().min())

    save_path = join('./backup/Results_Backup/News_Var_20250729/Var_Pred_Results', this_version)
    os.makedirs(save_path, exist_ok=True)

    test_pred_df.to_csv(join(save_path, 'Pred_%s.csv' % date))


def connect_pred():
    save_path = join('./backup/Results_Backup/News_Var_20250729/Var_Pred_Results', this_version)
    files_l = os.listdir(save_path)
    pred_files = [i for i in files_l if 'Pred_' in i]
    pred_files.sort()
    test_pred_df_l = []
    for i in pred_files:
        test_pred_df = pd.read_csv(join(save_path, i), index_col=0)
        test_pred_df_l.append(test_pred_df)

    test_pred_df = pd.concat(test_pred_df_l)
    print(test_pred_df)

    sum_path = join('./backup/Results_Backup/News_Var_20250729/Var_Results_Sum')
    os.makedirs(sum_path, exist_ok=True)
    test_pred_df.to_csv(join(sum_path, this_version + '_pred.csv'))


if __name__ == '__main__':
    save_path = join('./backup/Results_Backup/News_Var_20250729/Var_Pred_Results', this_version)
    
    filer_news = ['Num_scheduled_News', 'Num_unscheduled_News',
       'CSS_abs_mean', 'abs_EVENT_SENTIMENT_SCORE', 'css_std', 'SENTIMENT_std',
       'SENTIMENT_mean', 'SENTIMENT_mean_absolute', 'css_tone_absolute',
       'media_coverage', 'SENTIMENT_offmarket', 'SENTIMENT_onmarket',
       'Num_onmarket_News', 'Num_offmarket_News', 'sentiment_transfer_market',
       'sentiment_transfer_market_abs', 'num_transfer_market']

    df = load_data(filer_news)

    date_l = list(set(df['date'].tolist()))
    date_l.sort()

    print('Training Starts Now ...')
    year_date_l = ['2013-01-03', '2015-01-06', '2017-01-09', '2019-01-04']
    idx = date_l.index(year_date_l[0])
    e_idx = date_l.index(year_date_l[1])

    for date in date_l[idx::opt.window]:
        print(' * ' * 20 + date + ' * ' * 20)
        Train(opt.model_name, df, date, date_l)

    connect_pred()