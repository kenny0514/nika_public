import pandas as pd
import numpy as np
import statsmodels.api as sm

def getBollinger (close, lookback, min_periods = None):
    # Rolling
    if min_periods == None: min_periods = lookback
    return (np.log(close) - np.log(close).rolling(lookback, min_periods = min_periods).mean())/np.log(close).rolling(lookback, min_periods = min_periods).std()

def blgr (series):
    return (np.log(series.iloc[-1]) - np.log(series).mean())/np.log(series).std()

def macd_norm(series, span_short, span_long):
    macd = np.log(series)[-span_short:].mean() - np.log(series)[-span_long:].mean()
    std = np.log(series)[-span_long:].std()
    return macd/std

def williamsR (close, lookback, nFrac = 10):
    pivotHigh = (close.rolling(nFrac).max().shift() <= close) & (close.rolling(nFrac).max().shift(-nFrac) <= close)
    pivotHigh = pd.Series(np.where(pivotHigh, close, np.nan), index = pivotHigh.index)
    rollingHigh = pivotHigh.rolling(window = lookback-nFrac, min_periods = 1).max().shift(nFrac).ffill()

    pivotLow = (close.rolling(nFrac).min().shift() >= close) & (close.rolling(nFrac).min().shift(-nFrac) >= close)
    pivotLow = pd.Series(np.where(pivotLow, close, np.nan), index = pivotLow.index)
    rollingLow = pivotLow.rolling(window = lookback - nFrac, min_periods=1).min().shift(nFrac).ffill()

    r = (close - rollingLow )/ (rollingHigh- rollingLow)
    return r

def trend(series):
    time = np.arange(len(series)) + 1  # Adding 1 to start the time index at 1 instead of 0
    X = sm.add_constant(time)  # Adds a column of ones to include an intercept in the model
    y = series
    model = sm.OLS(y, X).fit()
    return model.tvalues[1] 

def trendblgr(series):
    series = series.dropna()
    if len(series) < 10:
        return np.nan
    time = np.arange(len(series)) + 1
    X = sm.add_constant(time)
    y = series
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    normalized_resid = residuals.iloc[-1] / np.std(residuals)
    return normalized_resid

def ar1(series):
    X = sm.add_constant(series.shift(1)).dropna()  # Adds a column of ones to include an intercept in the model
    y = series.iloc[-len(X):]
    try:
        model = sm.OLS(y.iloc[1:], X.iloc[1:]).fit()
        return np.sign(model.params[1]) * model.tvalues[1]
    except:
        return np.nan

def calcADF(series, constant, lags):
    y,x= getYX(np.log(series),constant=constant,lags=lags)
    bMean_,bStd_= getBetas(y,x) 
    bMean_,bStd_=bMean_[0,0],bStd_[0,0]**.5
    return bMean_/bStd_


def _lagDF(df0,lags):
    df1=pd.DataFrame()
    if isinstance(lags,int):
        lags=range(lags+1) 
    else:
        lags=[int(lag) for lag in lags]
    for lag in lags:
        df_=df0.shift(lag).copy(deep=True) 
        df_.columns=[str(i)+'_'+str(lag) for i in df_.columns] 
        df1=df1.join(df_,how='outer')
    return df1


def getYX(series,constant,lags, uselaggedLevel = True):
    if isinstance(series, pd.Series):
        series = pd.DataFrame(series)    

    # Differencing
    series_=series.diff().dropna() 

    # Lagging
    x=_lagDF(series_,lags).dropna()

    # Reintroduce lagged level: Used for ADF
    if uselaggedLevel:
        x.iloc[:,0]=series.values[-x.shape[0]-1:-1,0] # lagged level 
    else:
        x = x.drop(x.columns[0], axis=1)  # dropping zero lag
    y=series_.iloc[-x.shape[0]:].values

    # Adding Terms
    if constant!='nc': 
        # Add constant
        x=np.append(x,np.ones((x.shape[0],1)),axis=1) 
    if constant[:2]=='ct':
        # Add linear coefficient
        trend=np.arange(x.shape[0]).reshape(-1,1)
        x=np.append(x,trend,axis=1) 
    if constant=='ctt':
        # Add quadratic coefficient
        x=np.append(x,trend**2,axis=1) 
    return y,x

def getBetas(y,x):
    # Linear regression; outputs coefficient and variance
    xy=np.dot(x.T,y)
    xx=np.dot(x.T,x)
    xxinv=np.linalg.inv(xx)
    bMean=np.dot(xxinv,xy)
    err=y-np.dot(x,bMean) 
    bVar=np.dot(err.T,err)/(x.shape[0]-x.shape[1])*xxinv 
    return bMean,bVar

# Based on xrp_model_config_v240524.ipynb:

# 'sharpe1_1', 'sharpe2_1', 'returns_1', 'std_1', 'vol_1', 'blgr_1',
#        'trend_1', 'trendblgr_1', 'willR_1', 'ar1_1', 'adf_1', 'macd_norm1_1',
#        'macd_norm2_1', 

# 'sharpe1_4', 'sharpe2_4', 'returns_4', 'std_4', 'vol_4',
#        'blgr_4', 'trend_4', 'trendblgr_4', 'willR_4', 'ar1_4', 'adf_4',
#        'macd_norm1_4', 'macd_norm2_4', 

# 'sharpe1_10', 'sharpe2_10',
#        'returns_10', 'std_10', 'vol_10', 'blgr_10', 'trend_10', 'trendblgr_10',
#        'willR_10', 'ar1_10'

def transformer(input_data):

    X = {}
    lookback = 100

    for scale in [1, 4, 10]:
        indices = np.full(lookback, -1) + np.arange(-(lookback-1),1)*scale
        closes_path = input_data.Close[indices]
        lrets_path = np.log(input_data.Close[indices]).diff()

        X[f'sharpe1_{scale}'] = (lrets_path.sum()/np.sqrt((lrets_path**2).sum()))
        X[f'sharpe2_{scale}'] = (lrets_path.sum()/lrets_path.std())
        X[f'returns_{scale}'] = lrets_path.sum()
        X[f'std_{scale}'] = lrets_path.std()
        X[f'vol_{scale}'] = np.sqrt((lrets_path**2).sum())

        X[f'blgr_{scale}'] = blgr(input_data.Close.iloc[-lookback*scale:])
        X[f'trend_{scale}'] = trend(np.log(closes_path)) # use_log=True
        X[f'trendblgr_{scale}'] = trendblgr(np.log(closes_path))
        X[f'willR_{scale}'] = williamsR(input_data.Close.iloc[-int(lookback*scale*1.5):], scale*lookback, nFrac=10)[-1] 

        X[f'ar1_{scale}'] = ar1(np.log(closes_path))

        if scale < 10:
            X[f'adf_{scale}'] = calcADF(closes_path, constant='c', lags = int(lookback*0.1))

            X[f'macd_norm1_{scale}'] = macd_norm(input_data.Close, span_short = scale*lookback//4, span_long = scale*lookback//2)
            X[f'macd_norm2_{scale}'] = macd_norm(input_data.Close, span_short = scale*lookback//2, span_long = scale*lookback)

    return pd.DataFrame([X])

