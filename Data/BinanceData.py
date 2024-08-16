import pandas as pd
import numpy as np
import os
import io
from concurrent.futures import ThreadPoolExecutor
import zipfile
import requests
import pandas as pd
from Utilities.backtest_tools import *
import gc
import aiohttp, time
import ccxt.async_support as ccxt_async
import asyncio
import creds


''' 
ticker = 'BTC'
symbol = 'BTC/USDT' for spot, 'BTC/USDT:USDT' for futures

'''

BASE_URL = "https://data.binance.vision/data/"

class BinanceData ():
    def __init__ (self, use_api_key = False):
        if use_api_key:
            self.ccxt = ccxt_async.binance(config={'apiKey': creds.api_key, 'secret': creds.api_secret, 'enableRateLimit': False,'options': {'defaultType': 'spot'},})
        else:
            self.ccxt = ccxt_async.binance(config={'enableRateLimit': False,'options': {'defaultType': 'spot'},})
        self.kaiko = None
        self.spot_markets = None
        self.futures_markets = None
    
    async def load_markets(self):
        markets = await self.ccxt.fetch_markets()
        markets = [market for market in markets if (market['quote']=='USDT')]
        self.spot_markets = [market['baseId'] for market in markets if (market['type'] == 'spot')]
        self.futures_markets = [market['baseId'] for market in markets if (market['type'] == 'swap')]
        del markets        

    async def ticker_to_symbol (self, ticker):
        spot_symbol, futures_symbol = None, None
        if self.spot_markets is None:
            await self.load_markets()
        if ticker in self.spot_markets:
            spot_symbol = ticker + "/USDT"
        if ticker in self.futures_markets:
            futures_symbol = ticker + "/USDT:USDT" 
        if ("1000"+ticker) in self.futures_markets:
            futures_symbol = "1000" + ticker + "/USDT:USDT"
        return spot_symbol, futures_symbol

    def symbol_to_ticker (self, symbol, type):
        if type == 'spot':
            ticker = symbol.replace("/USDT", "")
        elif type == 'futures':
            ticker = symbol.replace("/USDT:USDT","")
            if '1000' in ticker:
                ticker = ticker.replace("1000","")
            if ticker in ['DODOX', 'BEAMX', 'LUNA2']:
                ticker = ticker.replace("2","")
                ticker = ticker.replace("X","")
        return ticker



    async def get_tickers (self, include_delisted = False):
        bn_markets = await self.ccxt.fetch_markets()
        if include_delisted:
            bn_spot_markets = [market['base'].replace('USDT', "") for market in bn_markets if market['quote'] == 'USDT' and market['spot']]
            bn_futures_markets = [market['base'].replace('USDT', "") for market in bn_markets if market['quote'] == 'USDT' and market['swap']]
        else:
            bn_spot_markets = [market['base'].replace('USDT', "") for market in bn_markets if market['quote'] == 'USDT' and market['spot'] and market['active']]
            bn_futures_markets = [market['base'].replace('USDT', "") for market in bn_markets if market['quote'] == 'USDT' and market['swap'] and market['active']]

        return bn_spot_markets, bn_futures_markets


    async def get_symbols (self, include_delisted = False):
        bn_markets = await self.ccxt.fetch_markets()
        if include_delisted:
            bn_spot_markets = [market['symbol'] for market in bn_markets if market['quote'] == 'USDT' and market['spot']]
            bn_futures_markets = [market['symbol'] for market in bn_markets if market['quote'] == 'USDT' and market['swap']]
        else:
            bn_spot_markets = [market['symbol'] for market in bn_markets if market['quote'] == 'USDT' and market['spot'] and market['active']]
            bn_futures_markets = [market['symbol'] for market in bn_markets if market['quote'] == 'USDT' and market['swap'] and market['active']]

        return bn_spot_markets, bn_futures_markets

    async def get_tickers_with_both_spotNfutures (self):
        spot, futures = await self.get_tickers(include_delisted = False)
        spot_tickers, fut_tickers = [], []
        for ticker in spot:
            if ticker in futures:
                spot_tickers.append(ticker)
                fut_tickers.append(ticker)
            elif '1000'+ticker  in futures:
                spot_tickers.append(ticker)
                fut_tickers.append('1000'+ticker)
            elif ticker == 'LUNA':
                spot_tickers.append(ticker)
                fut_tickers.append(ticker + "2")
            elif ticker == 'DODO':
                spot_tickers.append(ticker)
                fut_tickers.append(ticker + "X")
        return spot_tickers, fut_tickers
        
# ------------------------------------------------------------           
# Basic
    
    async def get_ohlcv (self, ticker, asset_type, timeframe, numBars, since = None):

        if since is not None and not isinstance(since, (int, float)):
            since = int(pd.to_datetime(since, utc=True).timestamp()) * 1000

        spot_symbol, futures_symbol = await self.ticker_to_symbol(ticker)
        if asset_type == 'spot':
            symbol = spot_symbol
        elif asset_type == 'futures':
            symbol = futures_symbol
        max_retries = 5
        for i in range(max_retries):
            try:
                ohlcv = await self.ccxt.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=numBars,)
                break
            except Exception as e:
                print(f"{str(e)} -- Network error encountered. Retrying in {2*i} seconds...")
                return None
        else:
            print("Max retries reached. Unable to fetch data.")
            return None

        # 데이터프레임으로 변환
        if ohlcv is not None:
            df = pd.DataFrame(data=ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Time'] = pd.to_datetime(df['Time'], unit='ms', utc=True)
            df.set_index('Time', inplace=True)          
            return df        

    async def get_multi_ohlcv (self, ticker_list, asset_type, timeframe, numBars, since = None, price_type = 'Open'):
        tasks = []
        for ticker in ticker_list:
            tasks.append(self.get_ohlcv(ticker, asset_type, timeframe, numBars, since))
        dfs = await asyncio.gather(*tasks)
        df = pd.concat([df[price_type] for df in dfs if df is not None], axis =1) ## Note: Using Open price!
        df.columns = ticker_list
        df = df.iloc[-numBars:,:]
        return df

    async def get_multi_premium_index (self, symbol_list, timeframe, numBars, since = None, price_type = 'Open'):
        tasks = []
        for symbol in symbol_list:
            tasks.append(self.get_premium_index(symbol, timeframe, numBars, since))
        dfs = await asyncio.gather(*tasks)
        df = pd.concat([df[price_type] for df in dfs if df is not None], axis =1) ## Note: Using Open price!
        df.columns = symbol_list
        df = df.iloc[-numBars:,:]
        return df


# ------------------------------------------------------------           
# Future - Spot Spread
    async def get_spot_futures_spread (self, ticker):
        spot_symbol, futures_symbol = await self.ticker_to_symbol(ticker)
        if spot_symbol is not None and futures_symbol is not None:
            tasks = [self.ccxt.fetch_ticker(spot_symbol), self.ccxt.fetch_ticker(futures_symbol)]
            spot_price, futures_price = await asyncio.gather(*tasks)
            spot_price, futures_price = spot_price['last'], futures_price['last']
            spread_pct = (futures_price - spot_price)/spot_price
            return spread_pct


    async def get_premium_index (self, symbol, timeframe, numBars, since = None):

        if since is not None and not isinstance(since, (int, float)):
            since = int(pd.to_datetime(since, utc=True).timestamp()) * 1000

        max_retries = 5
        for i in range(max_retries):
            try:
                ohlcv = await self.ccxt.fetch_premium_index_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=numBars,)
                if ohlcv is not None: 
                    break
            except Exception as e:
                print(f"{str(e)} -- Network error encountered. Retrying in {i} seconds...")
                await asyncio.sleep(i)
        else:
            print("Max retries reached. Unable to fetch data.")
            return None

        # 데이터프레임으로 변환
        if ohlcv is not None:
            df = pd.DataFrame(data=ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Time'] = pd.to_datetime(df['Time'], unit='ms', utc=True)
            df.set_index('Time', inplace=True)          
            return df        

# ------------------------------------------------------------           
# 입출금 관련
    async def get_coin_status (self, ticker):
        coin_status = await self.ccxt.sapi_get_capital_config_getall()
        coin_status = {coin['coin']:coin for coin in coin_status}
        return coin_status[ticker]

    async def can_deposit (self,ticker):
        coin_status = await self.get_coin_status(ticker)
        return coin_status['depositAllEnable']
    
    async def can_withdraw (self,ticker):
        coin_status = await self.get_coin_status(ticker)
        return coin_status['withdrawAllEnable']


# ------------------------------------------------------------           
# For Full Historical Files

    async def get_futures_depth_full (self, ticker):

        expected_cols = ['timestamp', 'percentage','depth','notional']
        target_cols = ['timestamp', 'percentage','depth','notional']

        daily_range = self._get_date_range('daily', ticker)
        
        df_concat = pd.DataFrame()
        daily_range_ticker = daily_range.copy()

        for date in daily_range:
            path = BASE_URL + f"futures/um/daily/bookDepth/{ticker}/{ticker}-bookDepth-{date}.zip"
            result = await self.download_df(path, expected_cols, target_cols)
            if result is not None:
                df_concat = pd.concat([df_concat, result], ignore_index=True)
            else:
                daily_range_ticker.remove(date)

        if len(df_concat):
            filename = f"{ticker}_{self.type}_{self.timeframe}_{daily_range_ticker[0].replace('-', '')}_{daily_range_ticker[-1].replace('-', '')}.csv"
            filepath = self.output_dir + filename
            df_concat.to_csv(filepath)
            print(f"{filename} done")
        else:
            print(f"Empty df for {ticker}. Look into this.")

    async def get_ohlcv_full(self, ticker, timeframe, type, start_date, num_threads = 1, output_dir = None, freq='monthly'):
        # This function now contains the core logic that was previously in the loop

        # Setting
        base = BASE_URL + f"spot/{freq}/klines/" if type == 'spot' else BASE_URL + f"futures/um/{freq}/klines/"

        expected_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        target_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'taker_buy_volume']
        rename_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'TakerBuyVolume']

        # Prepare Monthly Range
        date_range = self._get_date_range(freq, ticker, start_date)

        # Gather files in the directory to check if it already exists
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)       
            else:
                files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
                if any(f"{ticker}_{type}" in file for file in files):
                    print(f'{ticker} already exists in the directory. Skipping...')
                    return      

        df_concat = pd.DataFrame()
        date_range = list(reversed(date_range))
        while date_range:
            tasks = []
            dates_to_remove = []

            for _ in range(num_threads):
                if not date_range:
                    break
                date = date_range.pop()  # Pop from the end of the reversed list
                path = f"{base}{ticker}/{timeframe}/{ticker}-{timeframe}-{date}.zip"
                tasks.append(self.download_df(path, expected_cols, target_cols, rename_cols))
                dates_to_remove.append(date)                
            
            results = await asyncio.gather(*tasks)
            for date, result in zip(dates_to_remove, results):
                if result is None: continue
                df_concat = pd.concat([df_concat, result], ignore_index=True)
        
        if len(df_concat)>0:
            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)       
                # filename = f"{ticker}_{type}_{timeframe}_{monthly_range_ticker[0].replace('-', '')}_{monthly_range_ticker[-1].replace('-', '')}.csv"
                filename = f"{ticker}_{type}_{timeframe}.csv"
                filepath = output_dir + filename
                df_concat.to_csv(filepath)
                print(f"{filename} done")
            return df_concat
        else: 
            print("ohlcv - empty result for ",ticker)
            return None

    async def get_trades_full(self, aggTrades = True, freq = 'monthly', by_date = True, get_zip= False, num_threads = 4):
        assert freq in ['monthly', 'daily'], "freq must be either 'monthly' or 'daily'"

        # Setting
        if self.type == 'spot': base = BASE_URL + f"spot/{freq}/"
        elif self.type == 'futures': base = BASE_URL + f"futures/um/{freq}/"

        if aggTrades: 
            base += "aggTrades/"
            expected_cols = ['agg_trade_id','price','quantity','first_trade_id','last_trade_id','transact_time','is_buyer_maker']
            target_cols = ['transact_time','price','quantity','is_buyer_maker']
        else:
            base += "trades/"
            expected_cols = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']
            target_cols = ['id', 'price', 'qty','time', 'is_buyer_maker']


        # Run
        for ticker in self.tickers:
            try:
                date_range = self._get_date_range(freq, ticker)
            except:
                continue
            ticker_base = base + f"{ticker}/{ticker}-aggTrades-" if aggTrades else base + f"{ticker}/{ticker}-trades-"
            
            # Check if data is available in Binance Vision
            if not self._ticker_has_data(ticker, base, freq, aggTrades): 
                continue

            # Iterate through range and fetch data
            if by_date:
                with ThreadPoolExecutor(max_workers = num_threads) as executor:
                    futures = [executor.submit(self._process_single_date, ticker, ticker_base, date, expected_cols, target_cols, get_zip = get_zip) for date in date_range]            
            # This hasn't been updated. Check before using. 
            else:
                self._process_ticker_data(ticker, ticker_base, date_range, expected_cols, target_cols)

    async def get_premium_full (self, ticker, timeframe, start_date, num_threads = 4, output_dir = None, freq='monthly'):

        base = BASE_URL + f"futures/um/{freq}/premiumIndexKlines/"
        expected_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        target_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        rename_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Prepare Monthly Range
        date_range = self._get_date_range(freq, ticker, start_date)

        # Gather files in the directory to check if it already exists
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)       
            else:
                files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
                if any(f"{ticker}_{type}" in file for file in files):
                    print(f'{ticker} already exists in the directory. Skipping...')
                    # return      

        df_concat = pd.DataFrame()
        date_range = list(reversed(date_range))
        while date_range:
            tasks = []
            dates_to_remove = []

            for _ in range(num_threads):
                if not date_range:
                    break
                date = date_range.pop()  # Pop from the end of the reversed list
                path = f"{base}{ticker}/{timeframe}/{ticker}-{timeframe}-{date}.zip"
                tasks.append(self.download_df(path, expected_cols, target_cols, rename_cols))
                dates_to_remove.append(date)                
            
            results = await asyncio.gather(*tasks)
            for date, result in zip(dates_to_remove, results):
                if result is None: continue
                df_concat = pd.concat([df_concat, result], ignore_index=True)

        if len(df_concat)>0:
            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)       
                filename = f"{ticker}_premium_{timeframe}.csv"
                filepath = output_dir + filename
                df_concat.to_csv(filepath)
                print(f"{filename} done")
            return df_concat
        else: 
            return None        

    def _get_date_range(self, freq, ticker, start_time):

        if self.kaiko is not None:
            start_time = self.kaiko.loc[ticker,'firstTradeDate']
        else:
            start_time = start_time
        
        end_time = pd.to_datetime(time.time(),unit='s')

        if freq == 'monthly':
            date_range = pd.date_range(start=start_time, end=end_time, freq='MS')
            date_range = [date.strftime('%Y-%m') for date in date_range]
        elif freq == 'daily':
            date_range = pd.date_range(start=start_time, end=end_time, freq='D')
            date_range = [date.strftime('%Y-%m-%d') for date in date_range]
        else:
            raise ValueError("Unsupported frequency. Please choose 'monthly' or 'daily'.")            
        return date_range


# ------------------------------------------------------------           
# For a certain date              

    async def get_trades_single(self, ticker, date, aggTrades, freq, type):
        assert freq in ['monthly', 'daily'], "freq must be either 'monthly' or 'daily'"

        # Setting
        if type == 'spot': 
            base = BASE_URL + f"spot/{freq}/"
        elif type == 'futures': 
            base = BASE_URL + f"futures/um/{freq}/"    

        if aggTrades: 
            base += "aggTrades/"
            expected_cols = ['agg_trade_id','price','quantity','first_trade_id','last_trade_id','transact_time','is_buyer_maker']
            target_cols = ['transact_time','price','quantity','is_buyer_maker']
        else:
            base += "trades/"
            expected_cols = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']
            target_cols = ['id', 'price', 'qty','time', 'is_buyer_maker']

        if type == 'spot': 
            expected_cols.append('someboolean')        

        # Run
        ticker_base = base + f"{ticker}/{ticker}-aggTrades-" if aggTrades else base + f"{ticker}/{ticker}-trades-"

        # Iterate through range and fetch data
        res = self._process_single_date(ticker, ticker_base, date, expected_cols, target_cols, get_zip = False)
        return res

    async def get_futures_depth_single (self, date, ticker, type):

        expected_cols = ['timestamp', 'percentage','depth','notional']
        target_cols = ['timestamp', 'percentage','depth','notional']
        path = BASE_URL + f"futures/um/daily/bookDepth/{ticker}/{ticker}-bookDepth-{date}.zip"
        result = await self.download_df(path, expected_cols, target_cols)
        return result

    async def get_ohlcv_daily_single (self, ticker, date, timeframe, type, num_days = 1):
        base = BASE_URL + "spot/daily/klines/" if type == 'spot' else BASE_URL + "futures/um/daily/klines/"

        expected_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        target_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'taker_buy_volume']
        rename_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'TakerBuyVolume']

        df_ohlcv = pd.DataFrame()
        for day in range(num_days):
            date += pd.Timedelta(days=day)
            path = base + ticker + f"/{timeframe}/{ticker}-{timeframe}-{date.strftime('%Y-%m-%d')}.zip"
            df = await self.download_df(path, expected_cols, target_cols, rename_cols)
            df_ohlcv = pd.concat([df_ohlcv, df], axis = 0)
        if len(df_ohlcv):
            df_ohlcv.set_index('Time', inplace=True, drop=True)
            return df_ohlcv
        else:
            return None
    
    async def get_ohlcv_monthly_single(self, ticker, date, timeframe, type):
        base = BASE_URL + "spot/monthly/klines/" if type == 'spot' else BASE_URL + "futures/um/monthly/klines/"

        expected_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        target_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'taker_buy_volume']
        rename_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'TakerBuyVolume']

        path = base + ticker + f"/{timeframe}/{ticker}-{timeframe}-{date}.zip"
        result = await self.download_df(path, expected_cols, target_cols, rename_cols)
        return result

# ------------------------------------------------------------                             
# Util

    async def _process_single_date(self, ticker, ticker_base, date, expected_cols, target_cols, get_zip):
        # Make Directory
        ticker_directory = os.path.join(self.output_dir, ticker)
        if not os.path.exists(ticker_directory):
            os.makedirs(ticker_directory)      

        # Check if it exists
        files = [f for f in os.listdir(ticker_directory)]
        if any(date.replace('-', '') in file for file in files):
            print(f'{ticker}-{date} already exists in the directory. ...')
            return
        
        path = ticker_base + f"{date}.zip"
        if get_zip:
            result = self._download_zip(path)

        else:
            result = await self.download_df(path, expected_cols, target_cols)

        return result

    def _save_ticker_dataframe(self, ticker, dataframe, date):
        filename = f"{ticker}/{ticker}_{self.type}_{self.timeframe}_{date.replace('-', '')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        dataframe.to_csv(filepath, index=False)
        print(f"{ticker} data for {date} saved at {filepath}")
        del dataframe

    def _save_ticker_zip (self, ticker, response, date):
        filename = f"{ticker}_{self.type}_{self.timeframe}_{date.replace('-', '')}.zip"
        ticker_directory = os.path.join(self.output_dir, ticker)
        if not os.path.exists(ticker_directory):
            os.makedirs(ticker_directory)      
        zip_path = os.path.join(ticker_directory, filename)
        with open(zip_path, 'wb') as zip_file:
            for chunk in response.iter_content(chunk_size=128):
                zip_file.write(chunk)
            print(f"{ticker} zip data for {date} saved at {zip_path}")
       
    async def _process_ticker_data(self, ticker, ticker_base, date_range, expected_cols, target_cols):
            # Check if the ticker file already exists
            files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]
            if any(ticker in file for file in files):
                print(f'{ticker} already exists in the directory. ...')
                return
            
            df_concat = pd.DataFrame()
            date_range_ = date_range.copy()
            for date in date_range:
                path = ticker_base + f"{date}.zip"
                result = await self.download_df(path, expected_cols, target_cols)
                if result is not None:
                    df_concat = pd.concat([df_concat, result], ignore_index=True)
                    print(f'{date} done')              
                else:
                    date_range_.remove(date)

            if len(df_concat):
                self._save_ticker_dataframe(ticker, df_concat, date_range_)
                del df_concat
                gc.collect()
            else:
                print(f"Empty df for {ticker}. Look into this.")

    def _save_ticker_dataframe(self, ticker, df_concat, monthly_range_ticker):
        filename = f"{ticker}_{self.type}_{self.timeframe}_{monthly_range_ticker.replace('-', '')}.csv"
        filepath = os.path.join(self.output_dir, filename)                       
        df_concat.to_csv(filepath, index=False)
        print(f"{ticker} done")                

    def _ticker_has_data(self, ticker, base, freq, aggTrades):
        if self.latest_start_time is not None:
            if freq == 'daily':
                latest_start_time = self.latest_start_time+'-01'    
            else:
                latest_start_time = self.latest_start_time
            if aggTrades:
                path = base + ticker + f"/{ticker}-aggTrades-{latest_start_time}.zip"
            else:
                path = base + ticker + f"/{ticker}-trades-{latest_start_time}.zip"         
            test = self._download_zip(path)
            if test is None:
                print(f'{ticker} data begins after {latest_start_time}. Skipping this.')
                return False
        return True

    def _download_zip(self, path):
        max_retries = 5
        attempt = 0
        for i in range(max_retries):
            try:
                response = requests.get(path, stream=True)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    return None
                else:
                    attempt += 1  # Increment the attempt counter if an error occurs
                    if attempt == max_retries:  # Check if this is the last attempt
                        raise ValueError(f"All {max_retries} attempts failed. Last error was: {e}")

    def _download_sync (self, path, max_retries =5):
        attempt = 0
        for i in range(max_retries):
            try: 
                response = requests.get(path, stream=True)
                response.raise_for_status()
                return response.content
            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    return None
                else:
                    attempt += 1  # Increment the attempt counter if an error occurs
                    if attempt == max_retries:  # Check if this is the last attempt
                        raise ValueError(f"All {max_retries} attempts failed. Last error was: {e}")
                    
    async def _download_async (self, path, max_retries = 5):
        attempt = 0
        async with aiohttp.ClientSession() as session:
            for i in range(max_retries):
                try:
                    async with session.get(path) as response:
                        response.raise_for_status()
                        return await response.read()
                except aiohttp.ClientResponseError as e:
                    if e.status == 404:
                        return None
                    else:
                        attempt += 1
                        if attempt == max_retries:
                            raise ValueError(f"All {max_retries} attempts failed. Last error was: {e}")

    async def download_df (self,path, expected_cols, target_cols, rename_cols = None):
        try:
            content = await self._download_async(path)

            zip_buffer = io.BytesIO(content)        
            
            # Open the zip archive from the buffer
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                # Assume the first file inside the zip is the CSV
                csv_file_name = zip_ref.namelist()[0]

                # Read this file directly into pandas
                with zip_ref.open(csv_file_name) as file:
                    df = pd.read_csv(file)        

                # 가끔 헤더없이 바로 첫줄로 데이터가 들어오는 경우
                if list(df.columns) != expected_cols:
                    list_df = pd.DataFrame([list(df.columns)], columns=expected_cols)      
                    # 가끔 2 decimal points가 있음      
                    list_df.iloc[0]=list_df.iloc[0].apply(correct_first_line_errors)
                    df.columns = expected_cols
                    df = pd.concat([list_df, df],ignore_index=True)        
        except Exception as e:
            return None

        # Cut off unnecessary columns
        df = df.loc[:,target_cols]

        # Rename
        if rename_cols is not None:
            rename_dict = dict(zip(target_cols, rename_cols))            
            df.rename(columns=rename_dict, inplace=True)

        # Formatting
        df['Time'] = pd.to_datetime(df['Time'], unit='ms', utc=True)
        df.loc[:, df.columns != 'Time'] = df.loc[:, df.columns != 'Time'].astype(float)
        df.index = df.Time
        return df
    
    







# ############################################
# def get_historical_futures_klines(symbol, interval, start_str=None, end_str=None, limit=1000):
#     """Get Historical Futures Klines from Binance."""
#     # Binance Client Set up
#     api_key = "vDy6zTzLhFSDN4O5PyI6o2MaG3pxeQ54nWaiRbh3SJb8qUOp8d6D2XPw3iq3H9ZR"
#     api_secret = "chlh1RuyzdkreREDBf95Q9x274UXJcSM4MfNVPD9Edm1plxlqbOZjQJEcRU8FnWL"
#     client = Client(api_key, api_secret)

#     if not isinstance(start_str, str):
#         start_str = str(int(start_str))
#     if not isinstance(end_str, str):
#         end_str = str(int(end_str))        

#     klines_args = {
#         "symbol": symbol, 
#         "interval": interval,
#         "limit": limit,
#         "klines_type": HistoricalKlinesType.FUTURES
#     }

#     if start_str is not None:
#         klines_args["start_str"] = start_str

#     if end_str is not None:
#         klines_args["end_str"] = end_str

#     df = _klines_to_df(client.get_historical_klines(**klines_args))
#     return df

# def _klines_to_df(klines):
#     """Convert klines to DataFrame."""
#     df = pd.DataFrame(klines, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
#     df.set_index('Time', inplace=True)
#     df.index = pd.to_datetime(df.index, unit="ms")
#     return df

