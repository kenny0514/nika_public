from Data.BinanceDownloader import BinanceDownloader
import aiohttp, requests
import mplfinance as mpf
import pandas as pd
import ccxt as ccxt_sync
import creds as creds
import numpy as np
import matplotlib.pyplot as plt
import random, datetime, time


exchange = ccxt_sync.binance(config={'apiKey': creds.api_key, 'secret': creds.api_secret, 'enableRateLimit': False,'options': {'defaultType': 'swap'},})
markets = exchange.fetch_markets()
markets = [market for market in markets if (market['quote']=='USDT')]
spot_markets = [market['baseId'] for market in markets if (market['type'] == 'spot')]
futures_markets = [market['baseId'] for market in markets if (market['type'] == 'swap')]
del markets        

def get_trading_symbol(ticker):
    spot_symbol, futures_symbol = None, None
    if ticker in spot_markets:
        spot_symbol = ticker + "/USDT"
    if ticker in futures_markets:
        futures_symbol = ticker + "/USDT:USDT" 
    if ("1000"+ticker) in futures_markets:
        futures_symbol = "1000" + ticker + "/USDT:USDT"
    return spot_symbol, futures_symbol

def get_trading_symbol_v2(ticker):
    spot_symbol, futures_symbol = None, None
    if ticker in spot_markets:
        spot_symbol = ticker + "USDT"
    if ticker in futures_markets:
        futures_symbol = ticker + "USDT"
    if ("1000"+ticker) in futures_markets:
        futures_symbol = "1000" + ticker + "USDT"
    return spot_symbol, futures_symbol

# CoinMarketCap Related

async def get_cmc_slug(ticker):
    params = {'symbol': ticker}
    headers = {'X-CMC_PRO_API_KEY': 'b34d6ecf-4318-463e-9200-453a03f48d60'}

    async with aiohttp.ClientSession() as session:
        
        async with session.get('https://pro-api.coinmarketcap.com/v1/cryptocurrency/map', params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            data = sorted(data['data'], key=lambda x: x['rank'] if x['rank'] is not None else float('inf'))

        slug = data[0]['slug']    
    return slug

async def get_cmc_id(ticker, session = None):
    params = {'symbol': ticker}
    headers = {'X-CMC_PRO_API_KEY': 'b34d6ecf-4318-463e-9200-453a03f48d60'}

    async with aiohttp.ClientSession() as session:
        
        async with session.get('https://pro-api.coinmarketcap.com/v1/cryptocurrency/map', params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            data = sorted(data['data'], key=lambda x: x['rank'] if x['rank'] is not None else float('inf'))

        slug = data[0]['id']    
    return slug

async def get_cmc_volume_share (ticker, exchange='binance'):
    slug = await get_cmc_slug(ticker)
    url = f'https://api.coinmarketcap.com/data-api/v3/cryptocurrency/market-pairs/latest?slug={slug}&start=1&limit=10&category=spot&centerType=all&sort=cmc_rank_advanced&direction=desc&spotUntracked=true'
    try:
        async with aiohttp.ClientSession() as session:

            async with session.get(url) as response:
                response.raise_for_status()
                markets = await response.json()
                markets = markets['data']['marketPairs']
                for market in markets:
                    if market['exchangeSlug'] == exchange:
                        return round(market['volumePercent'],2)
        print(f"No {exchange} found for {ticker}")
        return 0
    except:
        print(f"Error with with cmc volume share request: {ticker}")
        return 0
                
async def get_cmc_volume_backdated(ticker, date, lookback_days = 30):
    endDate = int(date.timestamp())
    startDate = int(endDate - 60*60*24*lookback_days)
    id = await get_cmc_id(ticker)
    url = f"https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical?id={id}&convertId=2781&timeStart={startDate}&timeEnd={endDate}"
    response = requests.get(url)
    try:
        datas = response.json()['data']['quotes']
        data = pd.DataFrame([data['quote'] for data in datas])
        volume = (data.volume*data.close).mean()
        return volume
    except:
        print(f"No cmc historical quotes fetched for {ticker}")
        return None

async def get_cmc_marketcap(ticker, date):
    # Gets marketcap of the day before
    endDate = int(date.timestamp()- 60*60*24*1)
    startDate = int(endDate - 60*60*24*1)
    id = await get_cmc_id(ticker)
    url = f"https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical?id={id}&convertId=2781&timeStart={startDate}&timeEnd={endDate}"
    response = requests.get(url)
    try: 
        marketcap = response.json()['data']['quotes'][0]['quote']['marketCap']/1e6
        return round(marketcap)
    except:
        print(f"No cmc historical quotes fetched for {ticker}")
        return None        

# Other

def plot_events (event_list, num_days):
    bd = BinanceDownloader()

    for event in event_list:
        time0 = event['date']
        for ticker in event['tickers']:
            asset_type = None
            result = {'ticker':ticker, 'date': time0,}
            
            try:
                # Data Retrieval
                symbol = (ticker+'USDT')
                df_ohlcv = pd.DataFrame()
                for n in np.arange(0,num_days):
                    date = time0 + pd.Timedelta(days = n)
                    df_ohlcv = pd.concat([df_ohlcv,bd.get_ohlcv_daily_(ticker = symbol, date = date.strftime("%Y-%m-%d"), timeframe = '15m', type='futures')], axis=0)
                    asset_type = 'futures'
            except:
                try:
                    # symbol = ("1000"+ticker+'USDT')
                    df_ohlcv = pd.DataFrame()
                    for n in np.arange(0,num_days):
                        date = time0 + pd.Timedelta(days = n)
                        df_ohlcv = pd.concat([df_ohlcv,bd.get_ohlcv_daily_(ticker = symbol, date = date.strftime("%Y-%m-%d"), timeframe = '15m', type='spot')], axis=0)
                        asset_type = 'spot'                
                except:
                    continue

            df_ohlcv.Time = pd.to_datetime(df_ohlcv.Time, unit='ms', utc=True)
            df_ohlcv.set_index('Time', inplace=True)
            result['type'] = asset_type

            t0 = np.searchsorted(df_ohlcv.index, time0)
            # df_ohlcv.Close.plot();plt.show()
            # df_ohlcv['pct_change'] = (df_ohlcv.Close/df_ohlcv.Open.iloc[0] - 1) * 100

            df_ohlcv = np.log(df_ohlcv)

            mpf.plot(df_ohlcv, type='candle', figsize = (16,8), title = f"{ticker} - {time0}");plt.show()
            
def get_event_returns (ticker, time0, timeframe = '5m', hold_periods = [10/60, 6, 24, 48], low = 6, plot=False, use_log = False, show_vertical = False):
    bd = BinanceDownloader()

    #1: Secure OHLCV
    spot_ticker, futures_ticker = get_trading_symbol_v2(ticker)
    if spot_ticker is None and futures_ticker is None:
        return None
    
    df_ohlcv = pd.DataFrame()
    try:
        for n in np.arange(max(hold_periods)//24+1):
            date = time0 + pd.Timedelta(days = n)
            df_ohlcv = pd.concat([df_ohlcv,bd.get_ohlcv_daily_(ticker = futures_ticker, date = date.strftime("%Y-%m-%d"), timeframe = timeframe, type='futures')], axis=0)
        asset_type = 'futures'
    except:
        try:
            df_ohlcv = pd.DataFrame()
            for n in np.arange(max(hold_periods)//24+1):
                date = time0 + pd.Timedelta(days = n)
                df_ohlcv = pd.concat([df_ohlcv,bd.get_ohlcv_daily_(ticker = spot_ticker, date = date.strftime("%Y-%m-%d"), timeframe = timeframe, type='spot')], axis=0)
            asset_type = 'spot'
        except:
            return None

    df_ohlcv.Time = pd.to_datetime(df_ohlcv.Time, unit='ms', utc=True)
    df_ohlcv.set_index('Time', inplace=True)


    result = {'asset': asset_type}
    
    #2: Calculate Returns
    t0 = np.searchsorted(df_ohlcv.index, time0)
    t0c = df_ohlcv.index[t0] # candle after the first one
    t0 = df_ohlcv.index[t0-1]
    p0 = df_ohlcv.Open[t0]
    rets = []
    lows = []
    highs = []
    hps = []
    for hp in hold_periods:
        t1 = t0 + pd.Timedelta(hours=hp)
        if t1 in df_ohlcv.index:
            p1 = df_ohlcv.Close[t1]
            rets.append(round(p1/p0-1,3))
            hps.append(round(hp,2))
            p1 = min(df_ohlcv.Low[t0c:t1])
            lows.append(round(p1/p0-1,3))
            p1 = max(df_ohlcv.High[t0c:t1])
            highs.append(round(p1/p0-1,3))
        else:
            # print(f"t1 not in index for {ticker}. t1 = {t1}")
            pass


    if len(rets) == 0: return None
    result['rets'] = rets
    result['hps'] = hps
    result['lows'] = lows
    result['highs'] = highs

    if plot==True:
        if use_log:
            df_ohlcv = np.log(df_ohlcv/df_ohlcv.Open[t0])
        if show_vertical:
            mpf.plot(df_ohlcv[t0-pd.Timedelta(minutes=30):], type='candle', figsize = (16,8), title = f"{ticker} - {asset_type} - {time0}", vlines=dict(vlines=time0,linewidths=2,alpha=0.4))
            plt.show()
        else:
            mpf.plot(df_ohlcv[t0-pd.Timedelta(minutes=30):], type='candle', figsize = (16,8), title = f"{ticker} - {asset_type} - {time0}")
        plt.show()


    #3: Sign Consistency
    # returns = [result[f'ret_{int(hp)}hr'] for hp in hold_periods]
    # all_positive = all(r > 0 for r in returns)
    # all_negative = all(r < 0 for r in returns)
    # result['sign_consistent'] = all_positive or all_negative        
    return result

class SyntheticNewsGenerator ():

    def __init__ (self):

        import ccxt
        exchange = ccxt.binance(config={
            'apiKey': creds.api_key,
            'secret': creds.api_secret,
            'enableRateLimit': False,
            'portfolioMargin': True,
            })
        markets = exchange.fetch_markets()
        markets = [market for market in markets if (market['quote']=='USDT') & (market['active'] == True)]
        self.markets = [market['baseId'] for market in markets]
        self.spot_markets = [market['baseId'] for market in markets if (market['type'] == 'spot')]
        self.futures_markets = [market['baseId'] for market in markets if (market['type'] == 'swap')]

        # Top 200 Cryptos in CMC
        url = 'https://api.coinmarketcap.com/data-api/v3/exchange/market-pairs/latest?slug=binance&category=spot&start=1&limit=200'
        response = requests.get(url)
        rankers = response.json()['data']['marketPairs']
        rankers = [ranker['baseSymbol'] for ranker in rankers]
        rankers = list(set(rankers))
        self.delist_candidates = [ticker for ticker in self.markets if ticker not in rankers]
        

    def get_new_announcements (self, event_type, min_tickers = 3, max_tickers = 10):
        if event_type == 'delisting':
            tickers = self.delist_candidates
        elif event_type == 'monitoring_tag':
            tickers = self.delist_candidates
        else:
            raise ValueError ("Code not ready for other event types..")
        
        ticker_samples = random.sample(tickers,random.randint(min_tickers,max_tickers))
        date = datetime.today().strftime('%Y-%m-%d')

        if event_type == 'monitoring_tag':
            msg = 'Binance Will Extend the Monitoring Tag to Include '
            for ticker in ticker_samples:
                if ticker == ticker_samples[-1]:
                    msg += 'and '+ticker
                else:
                    msg += ticker +", "
            msg += f' on {date}'
            news = {'exchange':'binance', 'title': msg, 'date': pd.to_datetime(time.time(),unit='s',utc=True) - pd.Timedelta(minutes=2)}
        elif event_type == 'delisting':
            msg = 'Binance Will Delist '
            for ticker in ticker_samples:
                if ticker == ticker_samples[-1]:
                    msg += 'and '+ticker
                else:
                    msg += ticker +", "
            msg += f' on {date}'
            news = {'exchange':'binance', 'title': msg, 'date': pd.to_datetime(time.time(),unit='s',utc=True) - pd.Timedelta(minutes=2)}            
        
        
        return news
    
