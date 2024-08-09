import pandas as pd
import numpy as np
from Utilities.backtest_tools import *
import statsmodels.api as sm


#———————————————————————————————————————
from scipy.signal import find_peaks

def getBollinger (close, lookback, min_periods = None, use_log = True):
    # Rolling
    if min_periods == None: min_periods = lookback
    if use_log:
        close = np.log(close)
    return (close - close.rolling(lookback, min_periods = min_periods).mean())/close.rolling(lookback, min_periods = min_periods).std()

def getMACD(close, span_short, span_long):
    macd = np.log(close).ewm(span = span_short, min_periods = span_short//2).mean() - np.log(close).ewm(span=span_long, min_periods = span_long//2).mean()
    return macd

def getMACD_hist(close, span_short, span_long):
    macd = np.log(close).ewm(span = span_short, min_periods = span_short//2).mean() - np.log(close).ewm(span=span_long, min_periods = span_long//2).mean()
    macd_hist = macd - macd.ewm(span=(span_short+span_long)//2, min_periods = span_short).mean()
    return macd_hist

def getMACD_norm(close, span_short, span_long, use_ewm = True, use_log = True):
    if use_log:
        close = np.log(close)
        
    if use_ewm:
        macd = close.ewm(span = span_short, min_periods = span_short//2).mean() - close.ewm(span=span_long, min_periods = span_long//2).mean()
        std = close.ewm(span=span_long, min_periods = span_long//2).std()
    else:
        macd = close.rolling(window = span_short, min_periods = span_short//2).mean() - close.rolling(window=span_long, min_periods = span_long//2).mean()
        std = close.rolling(window=span_long, min_periods = span_long//2).std()        
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

def rsi (close, lookback):
    diff = np.log(close).diff()
    gain = (diff*(diff>0)).rolling(lookback).sum()
    loss = (-diff*(diff<0)).rolling(lookback).sum()
    return gain/(gain+loss)

def getROC (close, lookback):
    ret = np.log(close).diff()
    return ret.rolling(lookback).sum()

def getROC_norm (close, lookback):
    ret = np.log(close).diff()
    return ret.rolling(lookback).sum()/ret.rolling(lookback).std()

def is_nyse_trading_hour(datetime_index):
    datetime_index = datetime_index.tz_localize('America/Chicago')

    # Define NYSE trading hours
    nyse_start = pd.Timestamp('09:30:00').time()
    nyse_end = pd.Timestamp('16:00:00').time()

    # Ensure the datetime series is timezone-aware and convert to Eastern Time if necessary
    if datetime_index.tz is None:
        raise ValueError("Datetime objects must be timezone-aware.")
    datetime_index = datetime_index.tz_convert('America/New_York')

    # Check if each datetime is within trading hours
    within_trading_hours = (datetime_index.time >= nyse_start) & (datetime_index.time <= nyse_end)

    # Exclude weekends (Saturday=5, Sunday=6)
    not_weekend = ~datetime_index.weekday.isin([5, 6])

    # Combine checks: within trading hours and not on a weekend
    return within_trading_hours & not_weekend

def get_correlation(series1, series2, tEvent, lookback, scale, grouping1='last', grouping2='last', use_log1 = False, use_log2=False):
    path1 = getPathMatrix(series1, tEvent, lookback, scale = scale, grouping = grouping1)
    path2 = getPathMatrix(series2, tEvent, lookback, scale = scale, grouping = grouping2)
    if use_log1: path1 = np.log(path1)
    if use_log2: path2 = np.log(path2)
    corr = matrix_correlations(path1, path2)
    return corr

def matrix_correlations(A_, B_):
    # Ensure A and B are numpy arrays with float data type to avoid integer division issues
    A = A_.astype(np.float64).values
    B = B_.astype(np.float64).values

    # Step 1: Standardize each row
    A_mean = A.mean(axis=1, keepdims=True)
    A_std = A.std(axis=1, keepdims=True)
    B_mean = B.mean(axis=1, keepdims=True)
    B_std = B.std(axis=1, keepdims=True)

    A_standardized = (A - A_mean) / A_std
    B_standardized = (B - B_mean) / B_std

    # Step 2: Compute the dot products of corresponding rows
    # This results in the sum of products of standardized scores
    products = np.sum(A_standardized * B_standardized, axis=1)

    # Step 3: Divide by (n-1) to get the correlation coefficients
    n = A.shape[1]
    correlations = products / (n - 1)

    return pd.Series(data=correlations,index=A_.index)

def calcADF(series, constant, lags):
    y,x= getYX(np.log(series),constant=constant,lags=lags)
    bMean_,bStd_= getBetas(y,x) 
    bMean_,bStd_=bMean_[0,0],bStd_[0,0]**.5
    return bMean_/bStd_

def ar1(series):
    X = sm.add_constant(series.shift(1)).dropna()  # Adds a column of ones to include an intercept in the model
    y = series.iloc[-len(X):]
    try:
        model = sm.OLS(y.iloc[1:], X.iloc[1:]).fit()
        return np.sign(model.params[1]) * model.tvalues[1]
    except:
        return np.nan
    
def blgr (series):
    return (np.log(series.iloc[-1]) - np.log(series).mean())/np.log(series).std()

def macd_norm(series, span_short, span_long):
    macd = np.log(series)[-span_short:].mean() - np.log(series)[-span_long:].mean()
    std = np.log(series)[-span_long:].std()
    return macd/std

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
    # Multivariate linear regression; outputs coefficient and variance
    xy=np.dot(x.T,y)
    xx=np.dot(x.T,x)
    xxinv=np.linalg.inv(xx)
    bMean=np.dot(xxinv,xy)
    err=y-np.dot(x,bMean) 
    bVar=np.dot(err.T,err)/(x.shape[0]-x.shape[1])*xxinv 
    return bMean,bVar

def rowwise_regression(X, Y):
    
    # Calculate means of each row
    mean_X = X.mean(axis=1)
    mean_Y = Y.mean(axis=1)
    
    # Calculate differences from the mean for each element
    diff_X = X.sub(mean_X, axis=0)
    diff_Y = Y.sub(mean_Y, axis=0)
    
    # Calculate numerator: sum of products of differences
    numerator = (diff_X * diff_Y).sum(axis=1)
    
    # Calculate denominator: sum of squared differences of X
    denominator = (diff_X ** 2).sum(axis=1)
    
    # Calculate slopes
    slopes = numerator / denominator
    
    # Calculate intercepts using the means
    intercepts = mean_Y - slopes * mean_X
    
    # Calculate residuals for each row
    predicted_Y = X.mul(slopes, axis=0).add(intercepts, axis=0)
    residuals = Y - predicted_Y
    
    # Calculate standard deviation of residuals
    residuals_std = residuals.std(axis=1)

    return slopes, intercepts, residuals_std

def _get_bsadf(logP,minSL,constant,lags, quantile = None):
    y,x=getYX(logP,constant=constant,lags=lags)
    startPoints=range(0,y.shape[0]+lags-minSL+1)
    allADF = []
    for start in startPoints:
        y_,x_=y[start:],x[start:] 
        if len(y_) == 0 or x_.shape[0]<=x_.shape[1]: continue
        bMean_,bStd_=getBetas(y_,x_) 
        bMean_,bStd_=bMean_[0,0],bStd_[0,0]**.5 
        allADF.append(bMean_/bStd_)    
    if quantile is not None:
        bsadf = np.nanquantile(allADF, quantile)
    else:
        bsadf = np.nanmax(allADF)

    return bsadf

def _get_smt(series, type, minSL, quantile = None):
    startPoints=range(0,len(series)-minSL+1)
    allSMT = []
    trend=np.arange(len(series)).reshape(-1,1)
    const = np.ones((series.shape[0],1)).reshape(-1,1)
    if type == 'poly1':
        x = np.column_stack([trend**2, trend, const])
        y = series.values.reshape(-1,1)

    elif type == 'poly2':
        x = np.column_stack([trend**2, trend, const])
        y = np.log(series).values.reshape(-1,1)

    elif type == 'exp':
        x = np.column_stack([trend, const])
        y = np.log(series).values.reshape(-1,1)

    elif type == 'power':
        x = np.column_stack([np.log(trend), const])
        y = np.log(series).values.reshape(-1,1)

    else:
        raise ValueError("Invalid type")

    for start in startPoints:
        y_,x_=y[start:],x[start:] 
        if len(y_) == 0 or x_.shape[0]<=x_.shape[1]: continue
        bMean_,bStd_=getBetas(y_,x_) 
        bMean_,bStd_=bMean_[0,0],bStd_[0,0]**.5 
        allSMT.append(abs(bMean_)/bStd_)    
        if quantile is not None:
            smt = np.nanquantile(allSMT, quantile)
        else:
            smt = np.nanmax(allSMT)
    return smt

def get_cusumBDE (events, close, window, lag):
    ''' Recursive version based on AR model with lagged value'''
    import statsmodels.api as sm
    # Run on mpPandasObj
    pathMatrix = getPathMatrix(close, events.index, nbars = (window+lag))
    out = pd.Series(index = events.index, dtype = float)
    for _, series in pathMatrix.iterrows():
        y,x=getYX(np.log(series),constant='c',lags=lag)
        model = sm.RecursiveLS(y, x)
        results = model.fit()
        cumError = results.resid_recursive.sum() / events.trgt[series.name]
        if np.isnan(cumError):
            pausehere=True
        out[series.name] = cumError
    return out

def get_cusumCSW (series, vol, tIn, window):
    ''' Based on simple rolling cusum.'''
    diff = np.log(series).diff()
    diff_pos = diff*(diff>0)/vol
    diff_neg = diff*(diff<0)/vol

    roll_cusum_pos = diff_pos.rolling(window).sum()/np.sqrt(window)
    roll_cusum_neg = diff_neg.rolling(window).sum()/np.sqrt(window)
    # df = pd.DataFrame({'cusum_pos': roll_cusum_pos[tIn], 'cusum_neg':roll_cusum_neg[tIn]}, index = tIn)
    return roll_cusum_pos[tIn], roll_cusum_neg[tIn]

def get_bsadf (events, series, maxL, minL, constant, lags, quantile = None):
    ''' 
    What this feature can tell you very well:
    "between maxL and minL, did (explosive/linear trend/u turn) occur?"
    I used lag 3, minL30, maxL 60. Need sufficient amount of samples.
    Use quantile 0.95 to avoid overfit. 
    
    Instructions:
    1) use lagged level. It forces the model to look at long-term (= the entirety of the window)
    2) If you want to capture explosive growth, set constant to "c"
    3) If you want to capture linear growth, set it to "nc". 
    4) ct captures 2차 parabola - so it captures u turns. 
    5) ctt captures 3차 parabola - a bit messy now. 

    When making features out of this, need to vary 
    1) maxL & minL to focus on different horizon
    2) of course, constant: nc for trend, c for explosion    


    '''
    pathMatrix = np.log(getPathMatrix(series, events.index, maxL + lags))
    out = []
    for _, series in pathMatrix.iterrows():
        bsadf = _get_bsadf(series, minL, constant, lags, quantile)
        out.append(bsadf)
    out = pd.Series(out, index = pathMatrix.index)
    return out

def get_smt (events, type, close, maxL, minL, quantile= None):
    '''This one doesn't work as well as bsadf. 
    It can't differentiate linear vs exponential even if it's set to exponential or power
    types: 'poly1', 'poly2', 'power', 'exp
    '''
    pathMatrix = np.log(getPathMatrix(close, events.index, maxL))
    out = []
    for _, series in pathMatrix.iterrows():
        bsadf = _get_smt(series, type, minL, quantile)
        out.append(bsadf)
    out = pd.Series(out, index = pathMatrix.index)
    return out

def bEncoding (series, events, window, consec = False):
    df = getPathMatrix(series, events.index, window + consec)
    if consec:
        # 1 when two consecutive returns have the same sign, 0 when different
        df = ((df*df.shift(axis=1))>0).iloc[:,1:]*1
    else:
        # 1 when return is positive
        df = (df>0)*1
    df = df.astype(str).apply(lambda x: ''.join(x), axis=1)
    return df

def qEncoding (events, series, window, n_bins):
    from sklearn.preprocessing import KBinsDiscretizer
    encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)

    out = pd.Series(index = events.index, dtype = str)
    for tIn in events.index:
        historical = series[:tIn] # in-sample lrets
        sample = historical.iloc[-window:]

        encoder.fit(historical.values.reshape(-1,1))
        msg = encoder.transform(sample.values.reshape(-1,1)).flatten().astype(int)
        out[tIn] = ''.join(map(str,msg))
    return out

def sEncoding (events, series, window):
    from sklearn.preprocessing import KBinsDiscretizer

    out = pd.Series(index = events.index, dtype = str)
    for tIn in events.index:
        # prep data
        historical = series[:tIn] # in-sample rets
        sample = historical.iloc[-window:]
        
        # build encoder
        n_bins = int((historical.max()-historical.min())//historical.std())
        encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
        encoder.fit(historical.values.reshape(-1,1))

        # transform 
        msg = encoder.transform(sample.values.reshape(-1,1)).flatten().astype(int)
        out[tIn] = ''.join(map(str,msg))
    return out

def _pmf1(msg,w):
    # Compute the prob mass function for a one-dim discrete rv 
    # len(msg)-w occurrences
    lib={}
    if not isinstance(msg,str):msg=''.join(map(str,msg))
    for i in range(w,len(msg)): 
        msg_=msg[i-w:i]
        if msg_ not in lib:lib[msg_]=[i-w]
        else:lib[msg_]=lib[msg_]+[i-w] 
    pmf=float(len(msg)-w) 
    pmf={i:len(lib[i])/pmf for i in lib} 
    return pmf

def _matchLength(msg,i,n):
    # Maximum matched length+1, with overlap. 
    # i>=n & len(msg)>=i+n
    subS=''
    for l in range(n):
        msg1=msg[i:i+l+1]
        for j in range(i-n,i):
            msg0=msg[j:j+l+1] 
            if msg1==msg0:
                subS=msg1
                break # search for higher l. 
    return len(subS)+1,subS # matched length + 1

def WWRunsTest_(msg):
    if isinstance(msg, str):
        sequence = np.array([int(x) for x in msg if x in '01'])
    else:
        sequence = np.array(msg)
        
    sequence = np.array(sequence)
    
    # Count the number of 0s and 1s
    n1 = np.sum(sequence == 0)
    n2 = np.sum(sequence == 1)
    
    # Calculate runs
    R = np.sum(sequence[:-1] != sequence[1:]) + 1
    
    # Calculate expected runs
    E_R = (2 * n1 * n2) / (n1 + n2) + 1
    
    # Calculate standard deviation of runs
    SD_R = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1)))
    
    # Calculate Z-score
    Z = (R - E_R) / SD_R
    
    return Z

def konto_(msg,window=None,reverse=True): 
    # Kenny's comment: This basically iterates through the parts of a message and asks
    # "have we seen this before?". If the answer is yes multiple times, it's lower entropy. 
    # Window limits the historical dictionary as well as the size of the objects we check.
    # Why would I want that? I think default should be window=none, and reversed msg. 
    # So reversing it makes sense because it quantifies "is what happened lately redundant in the full path, or new?"
    # So konto is more versatile than plugin, because the length is not fixed. It's kind of like sadf vs adf. 
    ''' Marcos' comment.
    * Kontoyiannis’ LZ entropy estimate, 2013 version (centered window). 
    * Inverse of the avg length of the shortest non-redundant substring.
    * If non-redundant substrings are short, the text is highly entropic. 
    * window==None for expanding window, in which case len(msg)%2==0
    * If the end of msg is more relevant, try konto(msg[::-1]) 
    '''
    out={'num':0,'sum':0,'subS':[]}
    if not isinstance(msg,str): msg=''.join(map(str,msg))
    if window is None: 
        points=range(1,int(len(msg)/2+1))
    else: 
        window=min(window,len(msg)/2) 
        points=range(window,len(msg)-window+1)
    if reverse:
        msg = msg[::-1]
    for i in points:
        if window is None:
            l,msg_=_matchLength(msg,i,i)
            out['sum']+=np.log2(i+1)/l # to avoid Doeblin condition 
        else:
            l,msg_=_matchLength(msg,i,window)
            out['sum']+=np.log2(window+1)/l # to avoid Doeblin condition 
        out['subS'].append(msg_)
        out['num']+=1
    out['h']=out['sum']/out['num'] 
    out['r']=1-out['h']/np.log2(len(msg)) # redundancy, 0<=r<=1 
    return out


def plugIn_(msg,w,reverse=True):
    # Kenny's comment: Given a message, it checks the level of redundancy of substrings with length w. 
    # So it doesn't have access to global dictionary like I initially thought - could be too computationally intensive.
    # Compute plug-in (ML) entropy rate
    if reverse:
        msg = msg[::-1]
    pmf=_pmf1(msg,w)
    out=-sum([pmf[i]*np.log2(pmf[i]) for i in pmf])/w 
    return out,pmf


def konto(msg_series, window = None):
    out = []
    for msg in msg_series:
        h = konto_(msg, window)['h']
        out.append(h)
    return pd.Series(out, index=msg_series.index)

def plugIn(msg_series, w):
    out = []
    for msg in msg_series:
        h, pmf = plugIn_(msg, w)
        out.append(h)
    return pd.Series(out, index=msg_series.index)

def WWRunsTest(msg_series):
    out = []
    for msg in msg_series:
        z = WWRunsTest_(msg)
        out.append(z)
    return pd.Series(out, index = msg_series.index)

def _getWeights_FFD(d,thres):
    w=[1.]
    k = 1
    while True:
        w_=-w[-1]/k*(d-k+1)
        if abs(w_) > thres:
            w.append(w_)
            k += 1
        else:
            break
    w=np.array(w[::-1]).reshape(-1,1) 
    return w

def plotWeights(dRange,nPlots,size):
    w=pd.DataFrame()
    for d in np.linspace(dRange[0],dRange[1],nPlots):
        w_=_getWeights_FFD(d,size=size) 
        w_=pd.DataFrame(w_,index=range(w_.shape[0])[::-1],columns=[d]) 
        w=w.join(w_,how='outer')
    ax=w.plot()
    ax.legend(loc='upper left'); plt.show() 
    return

def fracDiff_FFD(series,d,thres=1e-5): 
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1]. 
    '''
    #1) Compute weights for the longest series
    if isinstance(series, pd.Series):
        series = pd.DataFrame(series)
    w=_getWeights_FFD(d, thres)
    width=len(w)-1

    #2) Apply weights to values
    df={}
    for name in series.columns: 
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series(dtype=float) 
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs 
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True) 
    df=pd.concat(df,axis=1)
    return df

def getMinFFD(df0, target_t = -4, thres = 1e-4):
    # Modification of plotMinFFD
    from statsmodels.tsa.stattools import adfuller 
    out = []
    d_space=np.linspace(0,1,11)
    for d in d_space:
        df1=fracDiff_FFD(df0,d,thres=thres) 
        result=adfuller(df1,maxlag=1,regression='c',autolag=None)
        out.append(result[0])
    iloc = np.searchsorted(-np.array(out), -target_t, side='right')
    return d_space[iloc].round(2)

def _getBeta(series, sl):
    # This is different from _getBetas - the linear regression function
    hl = series[['High', 'Low']].values
    hl = np.log(hl[:, 0] / hl[:, 1]) ** 2
    hl = pd.Series(hl, index=series.index)
    beta = hl.rolling(window=2).sum()
    beta = beta.rolling(window=sl).mean()
    return beta.dropna()

def _getGamma(series):
    h2 = series['High'].rolling(window=2).max()
    l2 = series['Low'].rolling(window=2).min()
    gamma = np.log(h2.values / l2.values) ** 2
    gamma = pd.Series(gamma, index=h2.index)
    return gamma.dropna()

def _getAlpha(beta, gamma):
    den = 3 - 2 * 2**0.5
    alpha = (2**0.5 - 1) * (beta**0.5) / den
    alpha -= (gamma / den)**0.5
    alpha[alpha < 0] = 0  # set negative alphas to 0 (see p.727 of paper)
    return alpha.dropna()

def corwinSchultz(series, sl=1):
    beta = _getBeta(series, sl)
    gamma = _getGamma(series)
    alpha = _getAlpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    startTime = pd.Series(series.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, startTime], axis=1)
    spread.columns = ['Spread', 'Start_Time']  # 1st loc used to compute beta
    return spread

def kylesLambda(tEvent, df_volbar, lookback, scale = 1):
    # Return sensitivty to net volume
    netVol = 2*df_volbar.Volume_buy-df_volbar.Volume
    netVolPath = getPathMatrix(netVol, tEvent, nbars = lookback, scale = scale)
    retsPath = getPathMatrix(df_volbar.Close.pct_change().dropna(), tEvent, nbars = lookback, scale = scale)
    out = []
    for i in range(len(tEvent)):
        x,y = netVolPath.iloc[i,:], retsPath.iloc[i,:]
        x_,y_ = x.values.reshape(-1,1), y.values.reshape(-1,1)
        bMean, bVar = getBetas(y_,x_,)
        t = bMean[0][0]/(bVar[0][0]**.5)
        out.append(t)
    out = pd.Series(out, index = tEvent)
    return out

def amihudsLambda (tEvent, df_volbar, lookback, scale = 1):
    # Absolute log returns sensitivity to dollar volume
    netDollar = 2*df_volbar.Dollar_buy-df_volbar.Dollar
    netDollarPath = getPathMatrix(netDollar, tEvent, nbars = lookback, scale = scale)
    absLogRet = np.log(df_volbar.Close).diff().abs().dropna()
    absLogRetPath = getPathMatrix(absLogRet, tEvent, nbars = lookback, scale = scale)
    out = []
    for i in range(len(tEvent)):
        x,y = netDollarPath.iloc[i,:], absLogRetPath.iloc[i,:]
        x_,y_ = x.values.reshape(-1,1), y.values.reshape(-1,1)
        bMean, bVar = getBetas(y_,x_,)
        t = bMean[0][0]/(bVar[0][0]**.5)
        out.append(t)
    out = pd.Series(out, index = tEvent)
    return out

def hasbroucksLambda (tEvent, df_volbar, lookback, scale = 1):
    # Log return sensitivity to square root of net dollar volume
    netDollar = 2*df_volbar.Dollar_buy-df_volbar.Dollar
    netDollarSqrt = np.sign(netDollar)*np.sqrt(netDollar.abs())
    netDollarPath = getPathMatrix(netDollarSqrt,tEvent , nbars = lookback, scale = scale)
    logRet = np.log(df_volbar.Close).diff().dropna()
    logRetPath = getPathMatrix(logRet, tEvent, nbars = lookback, scale = scale)
    out = []
    for i in range(len(tEvent)):
        x = netDollarPath.iloc[i,:]
        y = logRetPath.iloc[i,:]
        x_,y_ = x.values.reshape(-1,1), y.values.reshape(-1,1)
        bMean, bVar = getBetas(y_,x_,)
        t = bMean[0][0]/(bVar[0][0]**.5)
        out.append(t)
    out = pd.Series(out, index = tEvent)
    return out    

def vpin(tEvent, df_volbar, lookback):
    netVol = 2*df_volbar.Volume_buy-df_volbar.Volume
    return netVol.rolling(lookback).mean().loc[tEvent]

def HLVol (tEvent,df_volbar,  window):
    HLVol = np.log(df_volbar.High/df_volbar.Low).rolling(window).mean()
    return HLVol.loc[tEvent]

def rollModel (tEvent, close, lookback):
    """
    Estimate the effective bid-ask spread using Roll's measure.
    Returns:
    c: Estimated half bid-ask spread
    sigma_u_squared: Estimated variance of the efficient price process
    """
    # Calculate the variance of price changes
    logRet = np.log(close).diff().dropna()
    logRetPath = getPathMatrix(logRet, tEvent, lookback)
    out = []
    for i in range(len(tEvent)):
        delta_p = logRetPath.iloc[i,:]
        var_delta_p = np.var(delta_p, ddof=1)

        # Calculate the serial covariance of price changes
        serial_cov = np.cov(delta_p[:-1], delta_p[1:], ddof=1)[0, 1]

        # Estimate half the bid-ask spread (c) using Roll's formula
        c = np.sqrt(-serial_cov)

        # Estimate the variance of the efficient price process (σ_u^2) using Roll's formula
        sigma_u_sq = var_delta_p - 2 * serial_cov
        out.append(sigma_u_sq)
    out = pd.Series(out, index = tEvent)
    
    return out

def scaled_rolling (tEvent, series, lookback, scale, grouping, rolling):
    pathMatrix = getPathMatrix(series, tEvent, lookback, scale, grouping = grouping)
    if rolling == 'sum':
        out = np.sum(pathMatrix, axis=1)
    elif rolling == 'avg':
        out = np.mean(pathMatrix, axis=1)
    else:
        raise ValueError("Invalid rolling method")
    return pd.Series(data=out, index= tEvent)

def getAR1(tEvent, close, lookback, scale, use_log, diff=False):
    def ar1(series):
        X = sm.add_constant(series.shift(1)).dropna()  # Adds a column of ones to include an intercept in the model
        y = series.iloc[-len(X):]
        try:
            model = sm.OLS(y.iloc[1:], X.iloc[1:]).fit()
            return np.sign(model.params[1]) * model.tvalues[1]
        except:
            return np.nan
          # Index 1 for the phi coefficient's t-statisti
    pathMatrix = getPathMatrix(close, tEvent, lookback, scale)
    if use_log: pathMatrix = np.log(pathMatrix)
    if diff: pathMatrix = pathMatrix.diff(axis=1).dropna(axis=1)
    t = pathMatrix.apply(ar1, axis=1)
    return t

def getATR(events, df_ohlcv, lookback, scale):
    highPathMatrix = getPathMatrix(np.log(df_ohlcv.High), events.index, lookback, scale, grouping = 'max')
    lowPathMatrix = getPathMatrix(np.log(df_ohlcv.Low), events.index, lookback, scale, grouping = 'min')
    atr = np.mean(highPathMatrix-lowPathMatrix,axis=1)
    return pd.Series(data = atr, index = events.index)

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

def getTrend(tEvent, close, lookback, scale, use_log, look_direction='backward'):
 # Index 1 for the slope coefficient's t-statistic
    pathMatrix = getPathMatrix(close, tEvent, lookback, scale, look_direction = look_direction)
    if use_log:pathMatrix = np.log(pathMatrix)
    t = pathMatrix.apply(trend, axis=1)
    return t

def getTrendBlgr(tEvent, close, lookback, scale, use_log):
    pathMatrix = getPathMatrix(close, tEvent, lookback, scale)
    if use_log:
        pathMatrix = np.log(pathMatrix)
    errors = pathMatrix.apply(trendblgr, axis=1)
    return errors

def getADF (tEvent, close, lookback, lags, constant, scale=1, grouping = 'last', use_log = True):
    pathMatrix = getPathMatrix(close, tEvent, lookback, scale, grouping = grouping)
    if use_log: pathMatrix = np.log(pathMatrix)
    out = []
    for _, close in pathMatrix.iterrows():
        try:
            y,x= getYX(close,constant=constant,lags=lags)
            if  y.shape[0]!= x.shape[0]:
                out.append(np.nan); continue
            bMean_,bStd_= getBetas(y,x) 
            bMean_,bStd_=bMean_[0,0],bStd_[0,0]**.5
            out.append(bMean_/bStd_)
        except:
            out.append(np.nan)
    out = pd.Series(out, index = pathMatrix.index)
    return out

def getHurst (tEvent, close, lookback, minlag, maxlag, scale, grouping ='last'):
    pathMatrix = getPathMatrix(close, tEvent, lookback, scale, grouping = grouping)
    out = []
    for _, series in pathMatrix.iterrows():
        hurst = calc_hurst(series, minlag, maxlag)
        out.append(hurst)
    
    out = pd.Series(out, index = pathMatrix.index)
    return out

def calc_hurst(price_series, minlag, maxlag):

    """Calculate the Hurst exponent of a Pandas Series."""
    # Convert to logarithmic returns
    lrets = np.log(price_series / price_series.shift(1)).dropna()

    # Define the range of lags
    lags = range(minlag, maxlag)

    # Calculate the variance of the lagged differences
    variances = [np.var(lrets.diff(lag)) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(variances), 1)

    # The Hurst exponent is half the slope of the regression line
    hurst = poly[0] / 2

    return hurst

def getSR (tEvent, df_ohlcv, lookback, scale, n_bins = 2, nFrac = 50):
    # Returns df with support & resistance level standardized with respect to latest price & standard deviation
    # Contains SR ratio which represents odds. 
    df_sr = pd.DataFrame(index=tEvent, columns=[f"Resistance_{i+1}" for i in range(n_bins)] + [f"Support_{i+1}" for i in range(n_bins)])

    pivotHigh = (df_ohlcv.Close.rolling(nFrac).max().shift() <= df_ohlcv.Close) & (df_ohlcv.Close.rolling(nFrac).max().shift(-nFrac) <= df_ohlcv.Close)
    pivotLow = (df_ohlcv.Close.rolling(nFrac).min().shift() >= df_ohlcv.Close) & (df_ohlcv.Close.rolling(nFrac).min().shift(-nFrac) >= df_ohlcv.Close)
    pivots = (pivotHigh|pivotLow).sort_index()
    pivots = df_ohlcv.Close[pivots.values]

    Xstd = np.std(getPathMatrix(np.log(df_ohlcv.Close), tEvent, lookback, scale),axis=1)
    minStd = 0.5

    for i in range(len(tEvent)):

        t1 = tEvent[i]
        t1_adj = np.searchsorted(df_ohlcv.index, t1)- (nFrac) # End of search, iloc
        t1_adj = df_ohlcv.index[t1_adj] # to datetime

        last = df_ohlcv.Close.loc[t1]
        std = Xstd.loc[t1]

        # Search resistance
        piv = pivots[:t1_adj].sort_index(ascending=False)
        piv = np.log(piv/last)/std

        res = piv[piv > minStd]
        res = res[res == res.cummax()]
        res = binSupRes (n_bins, res, 0.5, fill_value = 8,  method = 'mean')

        sup = piv[piv < -minStd]
        sup = sup[sup == sup.cummin()]
        sup = binSupRes (n_bins, sup, 0.5, fill_value = 8, method = 'mean')

        for j in range(n_bins):
            df_sr.at[t1, f"Resistance_{j+1}"] = res[j]
            df_sr.at[t1, f"Support_{j+1}"] = sup[j]
            df_sr.at[t1, f"RSratio_{j+1}"] = abs(res[j]/sup[j])
        
        df_sr.at[t1, f"RSratio_avg"] = abs(np.mean(res)/np.mean(sup))
    
    return df_sr['RSratio_avg']

def supportDist (events, close, lookback, scale, nSup = 3, nFrac = 10, include_highs = False, norm = False):
    pivotLow = (close.rolling(nFrac).min().shift() >= close) & (close.rolling(nFrac).min().shift(-nFrac) >= close)
    pivotHigh = (close.rolling(nFrac).max().shift() <= close) & (close.rolling(nFrac).max().shift(-nFrac) <= close)
    if include_highs:
        pivot = pivotLow | pivotHigh
    else:
        pivot = pivotLow
    pivot = pd.Series(np.where(pivot, close, np.nan), index = pivot.index)
    vol = np.log(close).rolling(lookback*scale).std()[events.index]
    out = pd.Series(index = events.index, dtype=float)
    for i in range(len(events)):

        t0 = close.index[events.barID[i] - lookback*scale]
        t1 = close.index[events.barID[i] - nFrac]
        last = close[events.index[i]]

        supports = pivot[t0:t1]
        supports = supports[supports<last]
        supports = supports.sort_values(ascending=False)

        nearest_sups = supports.iloc[:nSup]
        avg_dist = np.log(np.mean(nearest_sups)/last)
        out.iloc[i] = avg_dist
    if norm: out = out/vol
    return abs(out)

def resistDist (events, close, lookback, scale, nRes = 3, nFrac = 10, include_lows = False, norm = False):
    pivotHigh = (close.rolling(nFrac).max().shift() <= close) & (close.rolling(nFrac).max().shift(-nFrac) <= close)
    pivotLow = (close.rolling(nFrac).min().shift() >= close) & (close.rolling(nFrac).min().shift(-nFrac) >= close)
    if include_lows:
        pivot = pivotLow | pivotHigh
    else:
        pivot = pivotHigh
    pivot = pd.Series(np.where(pivot, close, np.nan), index = pivot.index)        
    vol = np.log(close).rolling(lookback*scale).std()[events.index]

    out = pd.Series(index = events.index, dtype=float)
    for i in range(len(events)):

        t0 = close.index[events.barID[i] - lookback*scale]
        t1 = close.index[events.barID[i] - nFrac]
        last = close[events.index[i]]

        resist = pivot[t0:t1]
        resist = resist[resist>last]
        resist = resist.sort_values(ascending=True)

        nearest_res = resist.iloc[:nRes]
        avg_dist = np.log(np.mean(nearest_res)/last)
        out.iloc[i] = avg_dist
    if norm: out = out/vol
    return abs(out)

def findPeakTrough (df):
    # df consists of rows of path prices, with the last column being prominence value
    peaks, troughs = [], []
    for _, row in df.iterrows():
        series = row[:-1]
        series = series[~pd.isna(series)]
        prom = row.values[-1]
        peaks_iloc, _ = find_peaks(series, prominence = prom, height = min(series))
        troughs_iloc, _ = find_peaks(-series,prominence = prom, height = min(-series))    
        try: peak_value = max(series.iloc[peaks_iloc]) or np.nan
        except: peak_value = np.nan
        try: trough_value = min(series.iloc[troughs_iloc])
        except: trough_value = np.nan
        peaks.append(peak_value)
        troughs.append(trough_value)
    df = pd.DataFrame({'peak': peaks, 'trough':troughs}, index = df.index).fillna(method='ffill')
    return df

def peakTroughScaler (fullRangeSeries, integer_indices, lookback, prom_series, num_cpu):
    '''series has to be full range'''
    df_path = getPathMatrix(fullRangeSeries, integer_indices, lookback)
    df_path['prom'] = prom_series
    df_pktr = mpPandasObj(findPeakTrough, df_path, num_cpu=num_cpu)
    peak, trough = df_pktr['peak'], df_pktr['trough']

    series = fullRangeSeries[integer_indices]
    series_scaled = (series-trough)/(peak-trough)
    return series_scaled

