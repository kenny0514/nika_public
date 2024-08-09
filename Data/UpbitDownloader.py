import requests
import creds as creds
import pandas as pd
import ccxt.async_support as ccxt_async
from Utilities.ip_rotator import ApiGateway
import aiohttp, asyncio, time, os


class UpbitDownloader ():
    def __init__ (self, rotate_ip=False):
        self.rotate_ip = rotate_ip
        self.ccxt = ccxt_async.upbit(config={'apiKey': creds.upbit_api_key, 'secret': creds.upbit_api_secret, 'enableRateLimit': False})
        if rotate_ip:
            self.gateway = ApiGateway(
                                'https://api.upbit.com',
                                regions=['us-east-1'],
                                access_key_id=creds.aws_key,
                                access_key_secret=creds.aws_secret)
            self.gateway.start()
            session = requests.session()
            session.mount('https://', self.gateway)
            self.session = session
            time.sleep(1)            
        pass

    async def get_ohlcv_full (self, ticker, minutes,  start_datetime, end_datetime=None, num_threads=1, rotate_ip = False, block_size = 200, output_dir = None, sleep = 0):

        if isinstance(start_datetime, str):
            start_datetime = pd.to_datetime(start_datetime)
        if end_datetime is None:
            end_datetime= pd.to_datetime(int(time.time()),unit='s')
        if isinstance(end_datetime, str):
            end_datetime = pd.to_datetime(end_datetime)


        # See if the file already exists
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)       
            else:
                files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
                if any(ticker in file for file in files):
                    print(f'{ticker} already exists in the directory. Skipping...')
                    return
                
        
        df_result = pd.DataFrame()
        done=False
                                    
        while end_datetime > start_datetime:
            start = time.time()
            tasks = []
            for _ in range(num_threads):
                if end_datetime <= start_datetime:
                    break
                tasks.append(self.get_ohlcv(ticker, minutes, end_datetime, block_size, rotate_ip))
                end_datetime -= pd.Timedelta(minutes=block_size * minutes)
            
            results = await asyncio.gather(*tasks)
            
            for ohlcv in results:
                if ohlcv is None:
                    done = True
                    break
                else:
                    df_result = pd.concat([df_result, ohlcv], axis=0)
            if done:
                break

            duration = time.time()-start
            if duration < sleep:
                await asyncio.sleep(sleep-duration)
                


        if len(df_result)>0:
            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)       
                filename = f"{ticker}_spot_{minutes}m.csv"
                filepath = output_dir + filename
                df_result.to_csv(filepath)
                print(f"{filename} done")
                return df_result
        else: 
            return None
        
    async def get_ohlcv (self, ticker, minutes, num_bars, end_datetime = None, rotate_ip = False):

        # Date Formatting
        if end_datetime is not None:
            if isinstance(end_datetime,str):
                end_datetime = pd.to_datetime(end_datetime, utc=True)
            end_datetime = (end_datetime).isoformat()
            if "+00:00" in end_datetime:
                end_datetime = end_datetime.replace("+00:00","")

        if end_datetime is not None:
            url = f"https://api.upbit.com/v1/candles/minutes/{minutes}?market=KRW-{ticker}&to={end_datetime}&count={num_bars}"
        else:
            url = f"https://api.upbit.com/v1/candles/minutes/{minutes}?market=KRW-{ticker}&count={num_bars}"
        
        headers = {"accept": "application/json"}
        if rotate_ip:
            url, host = self.gateway.get_new_url_n_host(url)
            headers['Host'] = host
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                response = await response.read()

        df = pd.DataFrame(eval(response))

        target_cols = ['candle_date_time_utc', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume','candle_acc_trade_price']
        rename_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'PriceVolume']        
        

        if len(df):
            df = df.loc[:,target_cols]
            rename_dict = dict(zip(target_cols, rename_cols))            
            df.rename(columns=rename_dict, inplace=True)
            df['Time'] = pd.to_datetime(df['Time'], utc=True)
            df = df.sort_values(by='Time',ascending=True).reset_index(drop=True)
            df.index = df['Time']
            return df
        else:
            return None
    
    async def get_multi_ohlcv (self, ticker_list, minutes, num_bars, price_type = 'Open', end_datetime = None, rotate_ip = False):
        tasks = []
        for ticker in ticker_list:
            tasks.append(self.get_ohlcv(ticker, minutes = minutes, num_bars = num_bars, end_datetime = end_datetime, rotate_ip = rotate_ip))
        dfs = await asyncio.gather(*tasks)
        df = pd.concat([df[price_type] for df in dfs], axis =1) ## Note: Using Open price!
        df.columns = ticker_list
        df = df.iloc[-num_bars:,:]
        return df

    async def get_spot_tickers (self):
        upbit_markets = await self.ccxt.fetch_markets()
        upbit_markets = [market['base'] for market in upbit_markets if market['quote']=='KRW']
        return upbit_markets
    
    async def get_current_prices(self, tickers):
        # Only for KRW base
        tickers_krw = ["KRW-"+ticker for ticker in tickers]
        url = 'https://api.upbit.com/v1/ticker'
        params = ",".join(tickers_krw)
        params = {'markets': params}
        response = eval(requests.get(url, params = params).text)
        df = pd.DataFrame(response)
        df.index = df['market']
        df.index = df.index.str.replace("KRW-","")
        return df