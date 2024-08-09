from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from tqdm import tqdm
from Utilities.backtest_tools import correct_first_line_errors
import os,gc, time
from BinanceVision import get_historical_futures_klines
import warnings
import zipfile

class BarBuilder:
    def __init__(self, inDir, outDir = None, max_slots=5, filerange = (None, None)):
        self.inDir, self.outDir = inDir, outDir
        self.max_slots = max_slots
        self.dict_ready = {}  # Dictionary to store ready DataFrames by index
        self.full_dict = {i: file for i, file in enumerate(sorted([f for f in os.listdir(self.inDir)[filerange[0]:filerange[1]] 
                                                                   if (f.endswith('.csv') or f.endswith('.zip')) and not f.startswith('._')]))}
        self.unread_indices = [i for i in range(len(self.full_dict))]
        self.futures = []  # Keep track of pending futures
        self.target_index = 0  # Index of the next file to process


    def read_file_to_dict(self, index):
        try:
            filepath = os.path.join(self.inDir, self.full_dict[index])
        except:
            pausehere=True
        if filepath:
            if 'zip' in filepath:
                try:
                    df = zip_to_dataframe(filepath)
                    self.dict_ready[index] = df
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(filepath)
                    
            else:
                self.dict_ready[index] = pd.read_csv(filepath, index_col = 0, on_bad_lines='warn')

    def createBars(self, type, barSize):

        df_out = pd.DataFrame(columns=['Time','Open','High','Low','Close','Volume'])
        df_remainder = pd.DataFrame()
        self.executor = ThreadPoolExecutor(max_workers=self.max_slots)
        progress_bar = tqdm(total=len(self.full_dict)) 

        while self.target_index < len(self.full_dict):
            free_slots = self.max_slots - (len(self.dict_ready) + len(self.futures))
            # Submit new read tasks if there are free slots and unread files
            for _ in range(free_slots):
                if self.unread_indices:
                    next_index = self.unread_indices.pop(0)
                    future = self.executor.submit(self.read_file_to_dict, next_index)
                    future.add_done_callback(future_callback)
                    self.futures.append(future)

            # Check and process ready files
            if self.target_index in self.dict_ready:
                df_tick = self.dict_ready[self.target_index]
                if not df_remainder.empty:
                    df_tick = pd.concat([df_remainder, df_tick]).reset_index(drop=True)    
                
                df_main, df_remainder = tickToBar(df_tick, barSize, type)        
                if not df_main.empty:
                    df_out = pd.concat([df_out, df_main], axis = 0)               

                self.dict_ready.pop(self.target_index, None)
                self.target_index += 1 
                progress_bar.update(1)
                del df_tick  # Explicitly delete the variable holding the DataFrame
                gc.collect()
                

            # Clean up finished futures
            self.futures = [f for f in self.futures if not f.done()]

        self.executor.shutdown(); progress_bar.close() 
        df_out = df_out.reset_index(drop=True)
        if self.outDir is not None:
            df_out.to_csv(self.outDir)
        return df_out

def future_callback(future):
    try:
        result = future.result()  # This will re-raise any exception that occurred
    except Exception as e:
        raise ValueError (f"An error occurred: {e}")
    finally:
        gc.collect()

def createBars_ (type, inDir, barSize, filerange = (None,None)):
    # For debugging, with no threading
    filepaths = sorted([os.path.join(inDir, f) for f in os.listdir(inDir) if f.endswith('.csv')][filerange[0]:filerange[1]])
    df_out = pd.DataFrame(columns=['Time','Open','High','Low','Close','Volume'])
    df_remainder = pd.DataFrame()
    progress_bar = tqdm(total=len(filepaths)) 
    for filepath in filepaths:
        df_tick = pd.read_csv(filepath, index_col=0, on_bad_lines='warn')

        if any(df_remainder):
            df_tick = pd.concat([df_remainder, df_tick]).reset_index(drop=True)

        # Tick 데이터를 볼륨바로 변환하고 마지막에 남는 볼륨은 다음 iteration을 위해 저장해두기
        df_main, df_remainder = tickToBar(df_tick, barSize, type)
        progress_bar.update(1)

        # 아웃풋 DF에 저장
        if not df_main.empty:
            df_out = pd.concat([df_out, df_main], axis = 0)

    df_out = df_out.reset_index(drop=True)  
    progress_bar.close()
    return df_out

def tickToBar (df_tick, barSize, type, agg = False, has_maker_info = False, remainder_rollover = True):
    # agg_trades have different column names than trades
    if not has_maker_info:
        df_tick['is_buyer_maker'] = df_tick['price'] > df_tick['price'].shift()
        df_tick['is_buyer_maker'] = np.where(df_tick['price'] == df_tick['price'].shift(), np.nan, df_tick['is_buyer_maker'])
        df_tick['is_buyer_maker'].fillna(method='ffill', inplace=True)
        df_tick['is_buyer_maker'] = df_tick['is_buyer_maker'].astype(bool)
    if agg:
        df_tick = df_tick.rename(columns={'quantity': 'qty', 'transact_time':'time'})
    df_tick['price_qty'] = df_tick['qty']*df_tick['price']        
    df_tick['is_round'] = np.where((df_tick['qty']%0.01==0) ^ (df_tick['price_qty']%10==0),1, 0)
    
    if type == 'volume': 
        df_tick['target'] = df_tick['qty']
    elif type == 'dollar': 
        df_tick['target'] = df_tick['price_qty']
    elif type =='tick':
        df_tick['target'] = len(df_tick)

    #1) 뭉쳐있는 Trade 나눠주기
    df_tick['cumsum'] = df_tick['target'].cumsum()
    df_tick['quotient'] = df_tick['cumsum']//barSize
    df_tick['q_chg'] = 1*(df_tick['quotient']>(df_tick['quotient'].shift(1)))
    df_tick['remainder'] = np.where(df_tick['q_chg'], df_tick['cumsum']%barSize,0)

    # Rows to be split
    rows_to_copy = df_tick[df_tick['q_chg'] == 1].copy()
    rows_to_copy.index += 0.5
    rows_to_copy['target'] = rows_to_copy['remainder']

    # Edit existing row
    df_tick.loc[df_tick['q_chg'] == 1, 'target'] = df_tick['target'] - df_tick['remainder']
    df_tick['q_chg'] = 0

    # Combine splitted row
    df_tick = pd.concat([df_tick, rows_to_copy]).sort_index()

    #2) 깔끔하게 row 쪼갰으니 이제 다시 Bar Grouping
    df_tick['cumsum'] = (df_tick['target']).cumsum()
    df_tick['quotient'] = df_tick['cumsum']//barSize
    df_tick['group'] = df_tick['q_chg'].cumsum()
    
    # Split to two groups: aggregate & remainder
    last_group = df_tick['group'].max()
    if remainder_rollover:
        if df_tick[df_tick['group']==last_group]['target'].sum() < barSize*0.999:
            df_main = df_tick[df_tick['group']<last_group].copy()
            df_remainder = df_tick[df_tick['group']==last_group]
            df_remainder = df_remainder.drop(columns=['cumsum','quotient','q_chg','remainder','group','is_round'])
        else:
            df_main = df_tick
            df_remainder = pd.DataFrame()
    else:
        df_main = df_tick

    # Reflect the adjusted values of target to original columns
    if type == 'volume':
        df_main.loc[:,'qty'] = df_main['target']
    elif type == 'dollar':
        df_main.loc[:,'price_qty']=df_main['target']
        df_main.loc[:,'qty'] = df_main['price_qty'] / df_main['price']

    ### Add features ###
    # Uptick
    df_main['price_chg'] = df_main['price'].diff()
    df_main['uptick'] = np.select(
        [df_main['price_chg'] > 0, df_main['price_chg'] < 0,],
        [1,-1],
        default=np.nan)
    df_main['uptick'].ffill(inplace=True)

    # Taker Buy
    df_main['taker_buy_vol'] = (~df_main['is_buyer_maker'])*df_main['qty']
    df_main['taker_buy_dollar'] = (~df_main['is_buyer_maker'])*df_main['price_qty']

    # Round Orders
    # df_main['round_vol_total'] = df_main['qty'] * df_main['is_round']
    # df_main['round_vol_buy'] = df_main['taker_buy_vol'] * df_main['is_round']
    # df_main['round_tick_total'] = df_main['is_round']
    # df_main['round_tick_buy'] = df_main['is_round']*(~df_main['is_buyer_maker'])


    # Aggregate
    df_main = df_main.groupby('group').agg(
        Time0=('time','first'),
        Time1=('time','last'),
        Open=('price', 'first'),
        High=('price', 'max'),
        Low=('price', 'min'),
        Close=('price', 'last'),
        Volume=('qty', 'sum'),
        Volume_buy = ('taker_buy_vol', 'sum'),
        Dollar = ('price_qty', 'sum'),
        Dollar_buy = ('taker_buy_dollar', 'sum'),
        Tick = ('uptick', 'count'),
        NetTick = ('uptick','sum'),
        # RO_vol = ('round_vol_total','sum'),
        # RO_vol_buy = ('round_vol_buy','sum'),
        # RO_tick = ('round_tick_total', 'sum'),
        # RO_tick_buy = ('round_tick_buy','sum'),
        # medianOS = ('price_qty', 'median'),
        # avgOS = ('price_qty', 'mean'),
    )
    df_main['VWAP'] = df_main['Dollar']/df_main['Volume']
    if not df_main.empty:
        df_main['Time0'] = pd.to_datetime(df_main['Time0'], unit='ms')
        df_main['Time1'] = pd.to_datetime(df_main['Time1'], unit='ms')

    if remainder_rollover:
        return df_main, df_remainder
    else:
        return df_main

def zip_to_dataframe(zip_file_path):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expected_cols = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']
    target_cols = ['id', 'price', 'qty','time', 'is_buyer_maker']
    target_cols_dtypes = [float, float, float, int, bool]
    rename_cols = None

    try:
        with open(zip_file_path, 'rb') as f:
            df = pd.read_csv(f, on_bad_lines='warn')
    except:
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            csv_file_name = z.namelist()[0]
            with z.open(csv_file_name) as csv_file:
                df = pd.read_csv(csv_file, on_bad_lines='warn')

    
    # Occasionally, the data starts directly without a header
    if list(df.columns) != expected_cols:
        list_df = pd.DataFrame([list(df.columns)], columns=expected_cols)
        # Occasionally, there are decimal point errors
        list_df.iloc[0] = list_df.iloc[0].apply(correct_first_line_errors)
        df.columns = expected_cols
        df = pd.concat([list_df, df], ignore_index=True, axis=0)
        

    target_cols = ['id', 'price', 'qty','time', 'is_buyer_maker']
    
    # Cut off unnecessary columns
    df = df.loc[:,target_cols]

    # Rename
    if rename_cols is not None:
        rename_dict = dict(zip(target_cols, rename_cols))            
        df.rename(columns=rename_dict, inplace=True)

    df[df.columns[:-1]] = df[df.columns[:-1]].astype(float)
    df[df.columns[-1]] = df[df.columns[-1]].astype(bool)    
    return df

def round_sigfig(x, n):
    if x == 0:
        return 0
    else:
        from math import log10, floor
        # Determine the factor to scale x up or down to its most significant figure
        factor = 10 ** (n - 1 - floor(log10(abs(x))))
        # Scale x, round it, and then scale back
        return round(x * factor) / factor
    
def findBarSize (ticker, inDir, type, target_minute):
    filepaths = sorted([os.path.join(inDir,f) for f in os.listdir(inDir) if not f.startswith('._') ])
    total = 0

    # Find Duration
    if 'zip' in filepaths[0]:
        startTime = str(round(zip_to_dataframe(filepaths[0])['time'].iloc[0]))
        endTime = str(round(zip_to_dataframe(filepaths[-1])['time'].iloc[-1]))
    else:
        startTime = pd.read_csv(filepaths[0], index_col=0)['time'].iloc[0]
        endTime = pd.read_csv(filepaths[-1], index_col=0)['time'].iloc[-1]
    
    numBars = (float(endTime)-float(startTime))/1000/60/target_minute

    df = get_historical_futures_klines(ticker, '1d', startTime, endTime)
    if type =='volume':
        target = df['Volume'].astype(float).sum()
    elif type =='dollar':
        target = (df['Volume'].astype(float)*df['Close'].astype(float)).sum()

    barSize = round_sigfig(target/numBars, 4)

    return barSize

if __name__ == "__main__":
    tickers = ['XRP']
    type = 'dollar'
    for ticker in tickers:
        inDir = f'/Volumes/T7 Shield/data/{ticker}/'
        outDir = f'data/alt_bars/{ticker}_{type}.csv'
        bb = BarBuilder(inDir, outDir, max_slots = 5, filerange = (None, None)) # multi threading
        # barSize = findBarSize(ticker, inDir, type = type, target_minute = 5)
        out = bb.createBars(type, 2941000)
        del bb


