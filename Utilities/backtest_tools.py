import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import t
from datetime import datetime
import matplotlib.pyplot as plt
import os, traceback
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy import stats
import datetime


### Ctrl/Cmd+K & Ctrl/Cmd+0 to collapse and have bird's eye view ###


# Sampling
def getCrossingEvents (feat, df_ohlcv,  thres, crossing, mid=0, minVol = None, vol = None):
    if crossing == 'outward':
        if mid is None: mid = feat.quantile(0.50)        
        if thres >= mid:
            events = (feat >= thres) & (feat.shift() < thres)
        else:
            events = (feat <= thres) & (feat.shift() > thres)
    elif crossing == 'upward':
        events = (feat >= thres) & (feat.shift() < thres)
    elif crossing == 'downward':
        events = (feat <= thres) & (feat.shift() > thres)
    else:
        raise ValueError("Invalid crossing method")
    events = pd.DataFrame(index = events.index[events.values])
    events['barID'] = df_ohlcv.barID[events.index]

    if minVol is not None:
        if vol is None: raise ValueError("vol must be provided if minVol is applied")
        below_minVol = vol[events.index] < minVol
        events = events[~below_minVol]
    return events

def getCrossingEvents_dynamicExit(feat, df_ohlcv, entry_thres, exit_thres1, entry_cross, exit_cross1, exit_thres2= None,exit_cross2 = None,  mid = 0, minVol = None, vol = None):
    entry = getCrossingEvents(feat, df_ohlcv, entry_thres, entry_cross, mid, minVol, vol)
    entry['type'] = 'entry'
    exit = getCrossingEvents(feat, df_ohlcv, exit_thres1, exit_cross1, mid, None, None)
    exit['type'] = 'exit'
    if exit_thres2 is not None:
        exit2 = getCrossingEvents(feat, df_ohlcv, exit_thres2, exit_cross2, mid, None, None)
        exit2['type'] = 'exit'
        exit = pd.concat([exit, exit2],axis =0).sort_index()
    trades = pd.concat([entry, exit], axis = 0).sort_index()
    trades = trades[trades.type != trades.type.shift()]
    if len(trades):
        if trades.type.iloc[0]=='exit': trades = trades.iloc[1:]
    if len(trades):
        if trades.type.iloc[-1]=='entry': trades = trades.iloc[:-1]

    events = trades[trades.type=='entry'].copy()
    events['t1'] = trades[trades.type=='exit'].index
    events.drop(columns='type',inplace=True)
    return events

def cusumSampling (df_ohlcv, numBars, use_neg, use_pos, h_factor,  span = 1e4, min_periods = 5e3, verbose = False):
    cusumVol = getVol(df_ohlcv.Close, span = span, min_periods = min_periods, numBars = numBars, use_log = True)
    h = cusumVol/h_factor
    tEvents, _ = get_cusum_events(np.log(df_ohlcv.Close)[-len(h):], h, use_neg=use_neg, use_pos=use_pos)
    if verbose:
        cusumVol.plot(); plt.show()
        print("Opps per day:", pd.Series(index=tEvents, data=1).resample("D").sum().mean())
    return tEvents, cusumVol


# Labeling
def getLabel (tEvents, df_ohlcv, cusumVol, numBars, side, commission, verbose = False):
    t1 = getVertBarrier(tEvents, df_ohlcv.Close, numBars = numBars)
    events = getEvents(df_ohlcv.Close, tEvents, [0,0], cusumVol, minVol = 0, num_cpu = 1, t1= t1)
    Y = getBins(events,df_ohlcv.Close, use_log = True, commission = commission, side = side) # gets 'bins' and 'ret' 
    Y['t1'] = events.t1
    Y['entryP'], Y['exitP'], Y['barID'] = df_ohlcv.Close[Y.index], df_ohlcv.Close[Y.t1].values, df_ohlcv.barID[Y.index]
    events['barID'] = df_ohlcv.barID[events.index]
    if verbose:
        print("Average Duration:", (t1-t1.index).mean())
    return Y, events

def get_returns (ohlcv, time_indices, hold_periods, side, commission, use_log):
    result  = []
    if isinstance(hold_periods,int):
        hold_periods = [hold_periods]
    for hp in hold_periods:
        rets = ohlcv.Close.shift(-hp)[time_indices]/ohlcv.Close[time_indices] - 1
        rets = rets*np.where(side=='long',1,-1) - 2*commission
        result.append(rets)
    returns_matrix = np.column_stack(result)
    if use_log:
        returns_matrix = np.log(1+returns_matrix)
    return returns_matrix

def get_lrets (events, df_ohlcv, commission = 0, betSize = 1, side = 'long'):
    p1 = df_ohlcv.Close[events.t1.values].values
    p0 = df_ohlcv.Close[events.index]
    if isinstance(betSize, pd.Series):
        betSize = betSize[events.index]
    if side =='long':
        rets = (p1/p0 - commission - 1)*betSize
    else:
        rets = (1-p1/p0-commission)*betSize
    lrets = np.log(1+rets)
    return lrets

def triple_barrier (tEvent, df_ohlcv, hpbars, vol, tp, sl, side):
    close_path = getPathMatrix(df_ohlcv.Close, tEvent, nbars = hpbars+1, scale = 1, grouping = 'last', look_direction = 'forward')

    # Process TP & SL
    if sl > 0:
        if side == 'long':
            sl_price = (np.exp(-sl*vol)*df_ohlcv.Close)[tEvent]
            sl_bool = close_path.le(sl_price, axis = 0)
        else:
            sl_price = (np.exp(+sl*vol)*df_ohlcv.Close)[tEvent]
            sl_bool = close_path.ge(sl_price, axis = 0)

        sl_idx = np.argmax(sl_bool.values,axis = 1)            
        sl_idx[sl_idx==0] = hpbars        
    else:
        sl_idx = np.full(len(tEvent),hpbars)

    if tp > 0:
        if side == 'long':
            tp_price = (np.exp(tp*vol)*df_ohlcv.Close)[tEvent]
            tp_bool = close_path.ge(tp_price, axis = 0)
        else:
            tp_price = (np.exp(-tp*vol)*df_ohlcv.Close)[tEvent]
            tp_bool = close_path.le(tp_price, axis = 0)
    
        tp_idx = np.argmax(tp_bool.values,axis = 1)
        tp_idx[tp_idx==0] = hpbars
    else:
        tp_idx = np.full(len(tEvent),hpbars)
    
    exit_idx = np.minimum(tp_idx, sl_idx)
    t1 = df_ohlcv.index.get_indexer(tEvent) + exit_idx
    t1[t1>=len(df_ohlcv.index)] = len(df_ohlcv)-1
    t1 = df_ohlcv.index[t1]
    t1 = pd.Series(index = tEvent, data = t1)
    return t1    

def triple_barrier_ (df_ohlcv, integer_index,  barrier, side, commission, width_unit = 'atr', use_log = False):
    if not df_ohlcv.index.is_integer():
        raise ValueError("DataFrame index must be an integer index")
    

    num_entries = len(integer_index)
    entry_prices = np.array(df_ohlcv.Close.loc[integer_index])
    if width_unit == 'atr':
        width_unit = (calc_atr(df_ohlcv, lookback=30).loc[integer_index] / df_ohlcv.Close.loc[integer_index]).values
    elif width_unit == 'pct':
        width_unit = 0.01
    else:
        raise ValueError("Unknown type for width_unit")
    
    sl = barrier[1] if side == 'long' else barrier[0]

    ############### Process Vertical Barrier ###############
    # Calculate expiry index by adding vertical barrier to entry index
    index_expiry = (integer_index + np.full(num_entries,barrier[2])).astype(int)

    # Set out-of-bound index to -1 for now. (because Numpy can't process NAs)
    last_index = df_ohlcv.index.values[-1]
    index_expiry[index_expiry>last_index] = -1

    # Get Expiry price and Expiry Time
    expiry_price = df_ohlcv.Close.values[index_expiry]
    expiry_time = df_ohlcv.Time.values[index_expiry]

    # Set out-of-bound index, price, time to back to NAs.
    expiry_price[index_expiry<0] =np.nan
    expiry_time[index_expiry<0] = np.datetime64('NaT')
    index_expiry = index_expiry.astype(float)
    index_expiry[index_expiry<0] = np.nan

    ############### Process Horizontal Barriers ###############
    # Broadcast a 2D of m*n (m = trades in df_entry, n = price paths for each trade, value = index in ohlcv)
    # This creates index of price paths for each signal in one matrix for vectorized calculation. 
    index_matrix = (integer_index.reshape(-1,1) + np.arange(1, barrier[2]+1)).astype(int)
    '''Note: To transform from any timeframe to 1m timeframe, we add 1, multiply by tf_in_mins, and subtract 1'''
    index_matrix[index_matrix>last_index] = last_index

    # Processing upper barrier. 
    if barrier[0] > 0:
    
        # Calculate price for upper barrier
        upper_price = entry_prices*(1+width_unit*barrier[0]) 
    
        # Map high pricess using the index matrix
        high_matrix = df_ohlcv.High.values[index_matrix]

        # Create a boolean matrix marking if top barrier is crossed
        cross_upper = high_matrix > upper_price.reshape(-1,1)

        # Use the boolean matrix to mask the integer matrix and replace 'False' values with 'np.nan'
        masked_index_matrix = np.where(cross_upper, index_matrix, np.nan)

        # Get the minimum value for each row where the condition is True
        with np.errstate(invalid='ignore'):
            index_upper = np.nanmin(masked_index_matrix, axis=1).reshape(-1)

    else:
        index_upper = np.full(num_entries, np.nan)
        upper_price = np.full(num_entries, np.nan)

    # Processing lower barrier
    if barrier[1] > 0:

        # Calculate price of lower barrier
        lower_price = entry_prices*(1-width_unit*barrier[1])

        # Map high pricess using the index matrix
        low_matrix = df_ohlcv.Low.values[index_matrix]

        # Create a boolean matrix marking if top barrier is crossed
        cross_lower = low_matrix < lower_price.reshape(-1,1)

        # Use the boolean matrix to mask the integer matrix and replace 'False' values with 'np.inf'
        masked_index_matrix = np.where(cross_lower, index_matrix, np.nan)

        # Get the minimum value for each row where the condition is True
        with np.errstate(invalid='ignore'):
            index_lower = np.nanmin(masked_index_matrix, axis=1).reshape(-1)

    else:
        index_lower = np.full(num_entries, np.nan)
        lower_price = np.full(num_entries, np.nan)


    # Find which of upper, lower, expiry has the earliest index and set it as exit.
    stacked_arrays = np.vstack((index_upper.ravel(), index_lower.ravel(), index_expiry.ravel())).T
    exit_index = np.nanmin(stacked_arrays, axis=1)

    # Set np.nan to -1 for now.
    exit_index[np.isnan(exit_index)] = -1
    exit_index = exit_index.astype(int)

    # Calculate Exit Time and Duration
    # exit_time = df_ohlcv.Time.values[exit_index]
    # duration = exit_time - entry_times.values

    # Set -1 back to np.nans
    # duration[exit_index<0] = np.timedelta64('NaT')
    # exit_time[exit_index<0] = np.datetime64('NaT')
    exit_index = exit_index.astype(float)
    exit_index[exit_index<0] = np.nan

    # Calculate Exit Price
    conditions = [
        exit_index == index_upper,
        exit_index == index_lower,
        exit_index == index_expiry,
        ]
    choices = [upper_price, lower_price, expiry_price]

    exit_prices = np.select(conditions, choices, default=np.nan)

    # Calculate Returns
    rets = (exit_prices/entry_prices-1)*np.where(side=="long",1,-1) - 2*commission

    # Store necessary information to df_entry
    # df_entry['duration'] = duration
    # df_entry['rets'] = rets
    # df_entry['lrets'] = lrets
    # df_entry['exit_time'] = exit_time
    if use_log:
        rets = np.log(1+rets)
    return rets

def triple_barrier_trailing (df_ohlcv_1m, df_entry, barrier, side, tf_in_mins, use_trailing=False, activation_dist=None, trailing_dist=None):
    ##### New Version #####
    num_entries = len(df_entry)

    ############### Process Vertical Barrier ###############
    # Calculate expiry index by adding vertical barrier to entry index
    index_expiry = (df_entry.index.values+1)*tf_in_mins-1 + np.full(len(df_entry),barrier[2]*tf_in_mins)
    '''Transforming any timeframe to 1m timeframe: [Add 1, multiply by tf_in_mins, subtract 1]'''

    # Set out-of-bound index to -1 for now. (because Numpy can't process NAs)
    last_index = df_ohlcv_1m.index.values[-1]
    index_expiry[index_expiry>last_index] = -1

    # Get Expiry price and Expiry Time
    expiry_price = df_ohlcv_1m.Close.values[index_expiry]
    expiry_time = df_ohlcv_1m.Time.values[index_expiry]

    # Set out-of-bound index, price, time to back to NAs.
    expiry_price[index_expiry<0] =np.nan
    expiry_time[index_expiry<0] = np.datetime64('NaT')
    index_expiry = index_expiry.astype(float)
    index_expiry[index_expiry<0] = np.nan

    ############### Process Horizontal Barriers ###############
    # Broadcast a 2D of m*n (m = trades in df_entry, n = price paths for each trade, value = index in ohlcv)
    # This creates index of price paths for each signal in one matrix for vectorized calculation. 
    index_matrix = (df_entry.index.values.reshape(-1,1)+1)*tf_in_mins-1 + np.arange(1, barrier[2]*tf_in_mins+1)
    '''Note: To transform from any timeframe to 1m timeframe, we add 1, multiply by tf_in_mins, and subtract 1'''
    index_matrix[index_matrix>last_index] = last_index

    # Processing upper barrier. 
    if barrier[0] > 0:
    
        # Calculate price for upper barrier
        upper_price = df_entry.entry_price.values + df_entry.atr.values*barrier[0]
    
        # Map high pricess using the index matrix
        high_matrix = df_ohlcv_1m.High.values[index_matrix]

        # Create a boolean matrix marking if top barrier is crossed
        cross_upper = high_matrix > upper_price.reshape(-1,1)

        # Use the boolean matrix to mask the integer matrix and replace 'False' values with 'np.nan'
        masked_index_matrix = np.where(cross_upper, index_matrix, np.nan)

        # Get the minimum value for each row where the condition is True
        with np.errstate(invalid='ignore'):
            index_upper = np.nanmin(masked_index_matrix, axis=1).reshape(-1)

    else:
        index_upper = np.full(num_entries, np.nan)
        upper_price = np.full(num_entries, np.nan)

    # Processing lower barrier
    if barrier[1] > 0:

        # Calculate price of lower barrier
        lower_price = df_entry.entry_price.values - df_entry.atr.values*barrier[1]

        # Map high pricess using the index matrix
        low_matrix = df_ohlcv_1m.Low.values[index_matrix]

        # Create a boolean matrix marking if top barrier is crossed
        cross_lower = low_matrix < lower_price.reshape(-1,1)

        # Use the boolean matrix to mask the integer matrix and replace 'False' values with 'np.inf'
        masked_index_matrix = np.where(cross_lower, index_matrix, np.nan)

        # Get the minimum value for each row where the condition is True
        with np.errstate(invalid='ignore'):
            index_lower = np.nanmin(masked_index_matrix, axis=1).reshape(-1)

    else:
        index_lower = np.full(num_entries, np.nan)
        lower_price = np.full(num_entries, np.nan)

        ############### Process Trailing Stop ###############

    if use_trailing:
        if side == "long":

            # Check if trailing is activated
            highs = df_ohlcv_1m.High.values[index_matrix]
            rolling_max = np.maximum.accumulate(highs,axis=1)
            activation_price = (df_entry.entry_price.values+df_entry.atr.values*activation_dist).reshape(-1,1)
            is_activated = rolling_max >= activation_price

            # Check if trailing stop is crossed
            trail_distance = (df_entry.atr.values*trailing_dist).reshape(-1,1)
            trail_price = rolling_max - trail_distance
            lows = df_ohlcv_1m.Low.values[index_matrix]
            crossed_trail_stop = lows <= trail_price

            # Get earliest position (trigger_positions) within each path satisfyingfor both conditions
            combined_conditions = is_activated & crossed_trail_stop
            trigger_positions = np.argmax(combined_conditions, axis=1)
            untriggered = ~combined_conditions.any(axis=1)
            
            # Get the trigger price and the index of each path
            index_trail = index_matrix[np.arange(index_matrix.shape[0]), trigger_positions]
            trail_price = trail_price[np.arange(trail_price.shape[0]),trigger_positions]

            # Assign NA value to untriggered paths
            index_trail = np.where(untriggered, np.nan, index_trail)
            trail_price = np.where(untriggered, np.nan, trail_price)

        else: #if short
            # Check if trailing is activated
            lows = df_ohlcv_1m.Low.values[index_matrix]
            rolling_min = np.minimum.accumulate(lows,axis=1)
            activation_price = (df_entry.entry_price.values-df_entry.atr.values*activation_dist).reshape(-1,1)
            is_activated = rolling_min <= activation_price

            # Check if trailing stop is crossed
            trail_distance = (df_entry.atr.values*trailing_dist).reshape(-1,1)
            trail_price = rolling_min + trail_distance
            highs = df_ohlcv_1m.High.values[index_matrix]
            crossed_trail_stop = highs >= trail_price

            # Get earliest position (trigger_positions) within each path satisfyingfor both conditions
            combined_conditions = is_activated & crossed_trail_stop
            trigger_positions = np.argmax(combined_conditions, axis=1)
            untriggered = ~combined_conditions.any(axis=1)
            
            # Get the trigger price and the index of each path
            index_trail = index_matrix[np.arange(index_matrix.shape[0]), trigger_positions]
            trail_price = trail_price[np.arange(trail_price.shape[0]),trigger_positions]

            # Assign NA value to untriggered paths
            index_trail = np.where(untriggered, np.nan, index_trail)
            trail_price = np.where(untriggered, np.nan, trail_price)            

    else:
        index_trail = np.full(num_entries, np.nan)
        trail_price = np.full(num_entries, np.nan)
            

    # Find which of upper, lower, expiry has the earliest index and set it as exit.
    stacked_arrays = np.vstack((index_upper.ravel(), index_lower.ravel(), index_expiry.ravel(), index_trail.ravel())).T
    exit_index = np.nanmin(stacked_arrays, axis=1)

    # Set np.nan to -1 for now.
    exit_index[np.isnan(exit_index)] = -1
    exit_index = exit_index.astype(int)

    # Calculate Exit Time and Duration
    exit_time = df_ohlcv_1m.Time.values[exit_index]
    duration = exit_time - df_entry.entry_time.values

    # Set -1 back to np.nans
    duration[exit_index<0] = np.timedelta64('NaT')
    exit_time[exit_index<0] = np.datetime64('NaT')
    exit_index = exit_index.astype(float)
    exit_index[exit_index<0] = np.nan

    # Calculate Exit Price
    conditions = [
        exit_index == index_upper,
        exit_index == index_lower,
        exit_index == index_expiry,
        exit_index == index_trail,]
    choices = [upper_price, lower_price, expiry_price, trail_price]

    exit_price = np.select(conditions, choices, default=np.nan)

    # Calculate Returns
    rets = (exit_price/df_entry.entry_price.values-1)*np.where(side=="long",1,-1)
    lrets = np.log(1+rets)

    # Store necessary information to df_entry
    df_entry['duration'] = duration
    df_entry['rets'] = rets
    df_entry['lrets'] = lrets
    df_entry['exit_time'] = exit_time

    # df_entry['index_upper'] = index_upper
    # df_entry['index_lower'] = index_lower
    # df_entry['index_trail'] = index_trail
    # df_entry['index_expiry'] = index_expiry
    # df_entry['index_exit'] = exit_index

    # df_entry['lower_price'] = lower_price
    # df_entry['upper_price'] = upper_price
    # df_entry['trail_price'] = trail_price
    # df_entry['expiry_price'] = expiry_price
    # df_entry['exit_price'] = exit_price    
    return df_entry

def triple_barrier_dfstyle (df_ohlcv, df_entry_, barrier, side):

    df_entry = df_entry_.copy()
    num_entries = len(df_entry)

    # Calculate expiry index by adding vertical barrier to entry index
    expiry_index = df_entry.index.values + np.full(len(df_entry),barrier[2])

    # Set out of bound index to -1 for now.
    last_index = df_ohlcv.index.values[-1]
    expiry_index[expiry_index>last_index] = -1

    # Calculate Expiry price and Expiry Time
    expiry_price = df_ohlcv.Close.values[expiry_index]
    expiry_time = df_ohlcv.Time.values[expiry_index]

    # Set out of bound index, price, time to NAs.
    expiry_price[expiry_index<0] =np.nan
    expiry_time[expiry_index<0] = np.datetime64('NaT')
    expiry_index = expiry_index.astype(float)
    expiry_index[expiry_index<0] = np.nan

    # Set upper, lower and vertical
    upper_price = df_entry.entry_price.values + df_entry.atr.values*barrier[0]
    lower_price = df_entry.entry_price.values - df_entry.atr.values*barrier[1]

    # Find time hit for upper and lower for row. 
    if barrier[0] == 0 and barrier[1] == 0:
        index_upper = np.full(num_entries, np.nan)
        index_lower = np.full(num_entries, np.nan)

    else:

        ### This is the main code ###

        # Broadcast a 2D of m*n (m = trades in df_entry, n = price paths for each trade, value = index in ohlcv)
        index_matrix = df_entry.index.values.reshape(-1,1) + np.arange(1, barrier[2]+1)
        index_matrix[index_matrix>last_index] = last_index

        # If we have a top barrier
        if barrier[0] > 0:

            # Map high pricess using the index matrix
            high_matrix = df_ohlcv.High.values[index_matrix]

            # Create a boolean matrix marking if top barrier is crossed
            cross_upper = high_matrix > upper_price.reshape(-1,1)

            # Use the boolean matrix to mask the integer matrix and replace 'False' values with 'np.nan'
            masked_index_matrix = np.where(cross_upper, index_matrix, np.nan)

            # Get the minimum value for each row where the condition is True
            with np.errstate(invalid='ignore'):
                index_upper = np.nanmin(masked_index_matrix, axis=1).reshape(-1)

        else:
            index_upper = np.full(num_entries, np.nan)

        # If we have a bottom barrier
        if barrier[1] > 0:

            # Map high pricess using the index matrix
            low_matrix = df_ohlcv.Low.values[index_matrix]

            # Create a boolean matrix marking if top barrier is crossed
            cross_lower = low_matrix < lower_price.reshape(-1,1)

            # Use the boolean matrix to mask the integer matrix and replace 'False' values with 'np.inf'
            masked_index_matrix = np.where(cross_lower, index_matrix, np.nan)

            # Get the minimum value for each row where the condition is True
            with np.errstate(invalid='ignore'):
                index_lower = np.nanmin(masked_index_matrix, axis=1).reshape(-1)

        else:
            index_lower = np.full(num_entries, np.nan)

    # Find which of upper, lower, expiry has the earliest index and set it as exit.
    stacked_arrays = np.vstack((index_upper.ravel(), index_lower.ravel(), expiry_index.ravel())).T
    exit_index = np.nanmin(stacked_arrays, axis=1)


    # Set np.nan to -1 for now.
    exit_index[np.isnan(exit_index)] = -1
    exit_index = exit_index.astype(int)

    # Calculate Exit Time and Duration
    exit_time = df_ohlcv.Time.values[exit_index]
    duration = exit_time - df_entry.entry_time.values

    # Set -1 back to np.nans
    duration[exit_index<0] = np.timedelta64('NaT')
    exit_time[exit_index<0] = np.datetime64('NaT')
    exit_index = exit_index.astype(float)
    exit_index[exit_index<0] = np.nan

    # Calculate Exit Price
    conditions = [
        exit_index == index_upper,
        exit_index == index_lower,
        exit_index == expiry_index]
    choices = [upper_price, lower_price, expiry_price]

    exit_price = np.select(conditions, choices, default=np.nan)

    # Calculate Returns
    rets = (exit_price/df_entry.entry_price.values-1)*np.where(side=="long",1,-1)
    lrets = np.log(exit_price/df_entry.entry_price.values)*np.where(side=="long",1,-1)

    # Store necessary information to df_entry
    df_entry['duration'] = duration
    df_entry['rets'] = rets
    df_entry['lrets'] = lrets
    df_entry['exit_time'] = pd.to_datetime(exit_time,utc=True)

    return df_entry

def trailing_exits(tEvent, df_ohlcv,hpbars, vol, sl, trail_trigger, trail_dist, side):
    # Returns trailing exit time

    # std is logprice standard deviation

    close_path = getPathMatrix(df_ohlcv.Close, tEvent, nbars = hpbars, scale = 1, grouping = 'last', look_direction = 'forward')

    # Process SL
    if side == 'long':
        sl_price = (np.exp(-sl*vol)*df_ohlcv.Close)[tEvent]
        sl_bool = close_path.le(sl_price, axis = 0)
    else:
        sl_price = (np.exp(+sl*vol)*df_ohlcv.Close)[tEvent]
        sl_bool = close_path.ge(sl_price, axis = 0)

    # Process Trail
    if side == 'long':
        trail_trigger_price = (np.exp(trail_trigger*vol)*df_ohlcv.Close)[tEvent]
        cummax_path = close_path.cummax(axis=1)
        trigger_bool = cummax_path.ge(trail_trigger_price, axis = 0)
        trail_distance = (   abs(np.exp(-vol*trail_dist)-1)*df_ohlcv.Close  )[tEvent]
        diff_path = cummax_path - close_path
    else:
        trail_trigger_price = (np.exp(-trail_trigger*vol)*df_ohlcv.Close)[tEvent]
        cummin_path = close_path.cummin(axis=1)
        trigger_bool = cummin_path.le(trail_trigger_price, axis = 0)
        trail_distance = (   abs(np.exp(-vol*trail_dist)-1)*df_ohlcv.Close  )[tEvent]
        diff_path = close_path - cummin_path
    trail_bool = diff_path.ge(trail_distance, axis=0)
    trail_bool &= trigger_bool

    # Combine Result
    trail_idx = np.argmax(trail_bool.values,axis = 1)
    sl_idx = np.argmax(sl_bool.values,axis = 1)
    trail_idx[trail_idx==0] = hpbars-1
    sl_idx[sl_idx==0] = hpbars-1
    exit_idx = np.minimum(trail_idx, sl_idx)

    t1 = df_ohlcv.index.get_indexer(tEvent) + exit_idx
    t1[t1>=len(df_ohlcv.index)] = len(df_ohlcv)-1
    t1 = df_ohlcv.index[t1]
    t1 = pd.Series(data = t1, index = tEvent)
    return t1

def trailing_exits_iterative (tEvent, df_ohlcv, expiryTime, vols, sl, trail_trigger, trail_dist, side):

    out = pd.Series(index = tEvent)

    for i in range(len(tEvent)):
        t0 = tEvent[i]
        t1 = expiryTime[i]
        vol = vols[i]

        close_path = df_ohlcv.Close[t0:t1]

        # Process SL
        if side == 'long':
            sl_price = (np.exp(-sl*vol)*df_ohlcv.Close)[t0]
            sl_bool = close_path.le(sl_price, axis = 0)
        else:
            sl_price = (np.exp(+sl*vol)*df_ohlcv.Close)[t0]
            sl_bool = close_path.ge(sl_price, axis = 0)

        # Process Trail
        if side == 'long':
            trail_trigger_price = (np.exp(trail_trigger*vol)*df_ohlcv.Close)[t0]
            cummax_path = close_path.cummax()
            trigger_bool = cummax_path.ge(trail_trigger_price)
            trail_distance = (   abs(np.exp(-vol*trail_dist)-1)*df_ohlcv.Close  )[t0]
            diff_path = cummax_path - close_path
        else:
            trail_trigger_price = (np.exp(-trail_trigger*vol)*df_ohlcv.Close)[t0]
            cummin_path = close_path.cummin()
            trigger_bool = cummin_path.le(trail_trigger_price)
            trail_distance = (   abs(np.exp(-vol*trail_dist)-1)*df_ohlcv.Close  )[t0]
            diff_path = close_path - cummin_path
        trail_bool = diff_path.ge(trail_distance)
        trail_bool &= trigger_bool

        # Combine Result
        trail_idx = np.argmax(trail_bool.values)
        sl_idx = np.argmax(sl_bool.values)
        trail_idx = len(close_path)-1 if trail_idx==0 else trail_idx
        sl_idx = len(close_path)-1 if sl_idx==0 else sl_idx
        exit_idx = np.minimum(trail_idx, sl_idx)

        t1 = close_path.index[exit_idx]
        out.loc[t0] = t1
    return out


# Evaluation
def calc_zscore(daily_lrets, pop_u = 0):
    daily_lrets = daily_lrets[daily_lrets != 0]
    u = daily_lrets.mean()
    s = daily_lrets.std()
    z = (u - pop_u)/s*np.sqrt(len(daily_lrets))
    return z

def calc_zscore_2d(daily_lrets_2d):   
    mean_1d = np.nanmean(daily_lrets_2d, axis=0)
    std_1d = np.nanstd(daily_lrets_2d, axis=0, ddof=1)/np.sqrt(daily_lrets_2d.shape[0])

    if daily_lrets_2d.shape[0] <= 1:
        print("Degrees of freedom must be positive.")
    
    # Check if any standard deviations are zero (or effectively zero)
    if np.any(std_1d == 0):
        print("Zero standard deviation encountered.")

    std_1d[std_1d == 0] = np.nan
    zscore_1d = mean_1d/std_1d

    return zscore_1d

def calc_sharpe_on_2d(daily_lrets_2d, total_days):
    # Total days refer to the days of the entire period, not the number of days traded.

    # Mask out days with zero returns
    masked_lrets_2d = np.ma.masked_array(daily_lrets_2d, daily_lrets_2d == 0)
    
    # Count number of days traded
    num_days_1d = np.ma.count(masked_lrets_2d, axis=0)
    
    # If there is less than 1, can't calculate sharpe.
    if len(masked_lrets_2d) <= 1:
        return None
    
    # This is the annualizing factor.
    sample_size_per_year = num_days_1d/(total_days/365)
    
    # Calculate
    mean_1d = np.mean(masked_lrets_2d, axis=0)
    std_1d = np.std(masked_lrets_2d, axis=0, ddof=1)/np.sqrt(sample_size_per_year)
    sharpe_1d = mean_1d/std_1d

    return sharpe_1d

def calc_subperiod_metric (daily_lrets, num_periods, function, use_equal_sample_size = True):
    if use_equal_sample_size:
        sub_period_length = len(daily_lrets) // num_periods
        sub_metrics = []
        for i in range(num_periods):
            start_idx = i * sub_period_length
            # For the last sub-period, take all remaining data points
            end_idx = (i + 1) * sub_period_length if i < (num_periods-1) else len(daily_lrets)

            # Extract the sub-period
            sub_period_lrets = daily_lrets.iloc[start_idx:end_idx]
            sub_period_lrets = sub_period_lrets[sub_period_lrets!=0]

            # Calculate the submetric value
            sub_metric = function(sub_period_lrets)
            sub_metrics.append(sub_metric)
        return sub_metrics

    else:
        raise ValueError("아직 equal sample size 버전만 코드짜놓음. Equal time period 못만듬. ")

def calc_subsharpes_on_2d(daily_lrets_2d, start_indices, days_per_period):
    sub_sharpes = []
    for i in range(len(start_indices)):
        start = start_indices[i]
        end = start_indices[i+1] if i < len(start_indices)-1 else None
        sub_daily_lrets = daily_lrets_2d[start:end,:]
        sub_sharpe = calc_sharpe_on_2d(sub_daily_lrets, total_days = days_per_period)
        if sub_sharpe is None: # Skip if no return data for that period
            continue
        sub_sharpes.append(sub_sharpe)
    sub_sharpes = np.vstack(sub_sharpes)
    return sub_sharpes

def calc_sharpe(daily_lrets, total_days = None):
    # Calculate the number of days between the first and last date
    if total_days is None:
        total_days = (daily_lrets.index[-1] - daily_lrets.index[0]).days       

    # Calculate the multiplier as the proportion of the number of days over 365
    multiplier = 365 / total_days

    # Calculate the average of excess returns
    avg_excess_return = daily_lrets.mean()

    # Calculate the standard deviation of the excess returns
    std_dev_excess_return = daily_lrets.std()/np.sqrt(len(daily_lrets)*multiplier)

    # Calculate and return the Sharpe Ratio
    sharpe_ratio = avg_excess_return / std_dev_excess_return
    return sharpe_ratio

def calc_sharpe_zero(daily_lrets):
    # Calculate the number of days between the first and last date
    num_days = (daily_lrets.index[-1] - daily_lrets.index[0]).days

    if num_days == 0:
        return 0
 
    numerator = daily_lrets.sum()
    denominator = np.sqrt((daily_lrets**2).sum())
    annualizer = np.sqrt(len(daily_lrets)*365 / num_days)

    sharpe_ratio = numerator/denominator/annualizer
    return sharpe_ratio

def calc_calmar (daily_lrets):
    mdd = calc_mdd(daily_lrets)
    mean = daily_lrets[daily_lrets!=0].mean()
    return mean/mdd

def calc_HHI(series, n):
    # Split the series into n parts
    parts = np.array_split(series, n)
    
    # Calculate the sum of returns in each part
    sums = [part.sum() for part in parts]
    
    # Calculate the total sum of returns
    total_sum = series.sum()
    
    # Calculate the squared proportion for each part and sum them
    part_prop = sums / total_sum
    hhi = sum((part_sum / total_sum) ** 2 for part_sum in sums if total_sum != 0)

    return part_prop, hhi*n

def calc_mdd(daily_lrets):
    """
    Calculate the maximum drawdown from daily log returns.
    
    Args:
    daily_lrets (pandas.Series): A series of daily log returns.
    
    Returns:
    float: The maximum drawdown as a percentage.
    """
    # Convert log returns to cumulative returns
    cumulative_returns = np.exp(daily_lrets.cumsum())

    # Track the running maximum
    running_max = cumulative_returns.cummax()

    # Calculate drawdowns
    drawdowns = (cumulative_returns - running_max) / running_max

    # Find the maximum drawdown
    max_drawdown = abs(drawdowns.min())
    
    return max_drawdown

def get_kelly(lrets):
    rets = np.exp(lrets)-1
    winProb = np.sum(rets>0)/len(rets)
    avgGain = np.mean(rets[rets>0])
    avgLoss = -np.mean(rets[rets<0])
    GLratio = avgGain/avgLoss
    kelly = winProb-(1-winProb)/GLratio
    return kelly

def get_kelly_blocks (lrets_adj, block_size = 10):
    num_blocks = len(lrets_adj)//block_size
    kelly_list = [] 
    for block in range(num_blocks):
        start_idx = block * block_size
        end_idx = start_idx + block_size
        block_lrets = lrets_adj[start_idx:end_idx]
        winProb = (block_lrets > 0).sum() / len(block_lrets)

        positive_returns = block_lrets[block_lrets > 0]
        negative_returns = block_lrets[block_lrets < 0]

        if len(positive_returns) == 0:
            kelly_list.append(-1)
        elif len(negative_returns) == 0:
            kelly_list.append(1)
        else:
            avgGain = positive_returns.mean()
            avgLoss = -negative_returns.mean()
            GLratio = avgGain / avgLoss
            kelly = winProb - (1 - winProb) / GLratio
            kelly_list.append(kelly)
    return kelly_list

def get_kelly_chunks (lrets_adj, num_chunks):
    n = len(lrets_adj)
    chunk_size = n//num_chunks
    kelly_list = [] 
    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = start_idx + chunk_size
        chunk_lrets = lrets_adj[start_idx:end_idx]
        winProb = (chunk_lrets > 0).sum() / chunk_size

        positive_returns = chunk_lrets[chunk_lrets > 0]
        negative_returns = chunk_lrets[chunk_lrets < 0]

        if len(positive_returns) == 0:
            kelly_list.append(-1)
        elif len(negative_returns) == 0:
            kelly_list.append(1)
        else:
            avgGain = positive_returns.mean()
            avgLoss = -negative_returns.mean()
            GLratio = avgGain / avgLoss
            kelly = winProb - (1 - winProb) / GLratio
            kelly_list.append(kelly)
    return kelly_list

def get_lrets_chunks (lrets_adj, block_size = 10):
    num_blocks = len(lrets_adj)//block_size
    lret_list = [] 
    for block in range(num_blocks):
        start_idx = block * block_size
        end_idx = start_idx + block_size
        cum_lrets = lrets_adj[start_idx:end_idx].sum()
        lret_list.append(cum_lrets)
    return lret_list

def get_rolling_kelly(_returns, window_size):
    returns = np.array(_returns)

    n = len(returns)
    # Step 1: Create a 2D matrix for rolling window indices
    indices = np.arange(window_size)[None, :] + np.arange(n - window_size + 1)[:, None]

    # Step 2: Extract returns for each window
    windowed_returns = returns[indices]

    # Step 3: Calculate Kelly criterion for each window
    positive_returns = windowed_returns > 0
    win_prob = np.mean(positive_returns, axis=1)
    gains = np.where(positive_returns, windowed_returns, np.nan)
    losses = np.where(~positive_returns, windowed_returns, np.nan)
        
    average_gain = np.nanmean(gains, axis=1)
    average_loss = -np.nanmean(losses, axis=1)

    gl_ratio = average_gain / average_loss
    kelly_values = win_prob - (1 - win_prob) / gl_ratio
    
    # Adjust for all nans
    kelly_values[np.isnan(average_gain)] = -1
    kelly_values[np.isnan(average_loss)] = 1
    return kelly_values, average_loss

def t_test (daily_lrets):
    # Mask out days with zero returns
    daily_lrets = daily_lrets[daily_lrets!=0]

    # Count number of days traded
    num_days = len(daily_lrets)
    
    # If there is less than 1, can't calculate sharpe.
    if num_days <= 1:
        return None
    
    # Calculate
    mean = np.mean(daily_lrets)
    std = np.std(daily_lrets, ddof=1)/np.sqrt(num_days)
    t_stat = mean/std
    degrees_of_freedom = num_days - 1
    p_value = stats.t.sf(t_stat, degrees_of_freedom)
    
    return p_value

def t_test_2d (daily_lrets_2d):
    # Mask out days with zero returns
    masked_lrets_2d = np.ma.masked_array(daily_lrets_2d, daily_lrets_2d == 0)

    # Count number of days traded
    num_days_1d = np.ma.count(masked_lrets_2d, axis=0)
    
    # If there is less than 1, can't calculate sharpe.
    if len(masked_lrets_2d) <= 1:
        return None
    
    # Calculate
    mean_1d = np.mean(masked_lrets_2d, axis=0)
    std_1d = np.std(masked_lrets_2d, axis=0, ddof=1)/np.sqrt(num_days_1d)
    t_stats_1d = mean_1d/std_1d
    degrees_of_freedom_1d = num_days_1d - 1
    p_values_1d = stats.t.sf(t_stats_1d, degrees_of_freedom_1d)
    
    return p_values_1d

def contribution_from_n(series, n):
    """
    Calculate the total contribution of the top n-th percentile in a series.
    
    :param series: Numpy array or list of numeric values.
    :param n: Percentile to calculate (0-100).
    :return: Total contribution of the top n-th percentile.
    """

    if not 0 <= n <= 100:
        raise ValueError("Percentile n must be between 0 and 100")

    # Determine the (100 - n)th percentile value
    percentile_value = np.percentile(series, 100 - n)

    # Sum values that are greater than or equal to the percentile value
    contribution = np.sum(series[series >= percentile_value])

    return contribution

def contribution_from_max_group(series, n_group):
    """
    Calculate the contribution proportion of the highest return group among n groups in a series.

    :param series: Numpy array or list of numeric values.
    :param n: Number of groups to divide the series into.
    :return: Proportion of the highest return group's contribution.
    """
    # Calculate the size of each group
    total_elements = len(series)
    group_size = math.ceil(total_elements / n_group)

    # Split the series into n groups
    groups = [series[i:i + group_size] for i in range(0, total_elements, group_size)]

    # Calculate total return for each group
    total_returns = [np.sum(group) for group in groups]

    # Identify the highest total return
    highest_return = max(total_returns)

    # Calculate the total return of all groups
    total_return_all_groups = sum(total_returns)

    # Calculate the proportion
    proportion = highest_return / total_return_all_groups if total_return_all_groups != 0 else 0

    return proportion

def getFeatUtility (feat, side, quantiles, df_ohlcv, dataDays, crossing, leverage, commission,  mid, visual, tradingDays,hpBar = None, hpDay = None,):
    '''This outputs the utility of a feature. Utility is defined as annual total return times annual sharpe'''
    '''Num Days required for annualization'''

    if (hpBar == None) & (hpDay == None):
        raise ValueError ("Either hp Bar of hp Day must be provided")

    tot_rets, sharpes, utilities, zscores, freq, avgHP, sharpe_freqs = pd.Series(index = quantiles, dtype=float), pd.Series(index= quantiles, dtype=float), pd.Series(index= quantiles, dtype=float), pd.Series(index= quantiles, dtype=float), pd.Series(index= quantiles, dtype=float), pd.Series(index= quantiles, dtype=float),pd.Series(index= quantiles, dtype=float)

    for thres in quantiles:

        # Prepare Events
        events = getCrossingEvents(feat = feat, df_ohlcv= df_ohlcv, thres = thres, crossing = crossing, mid= mid)
        events = addT1toEvents (events, df_ohlcv, hpDay, hpBar)
        pre_len = len(events)
        events = removeTimeOverlap(events)
        post_len = len(events)
        try:
            reduced = 1-post_len/pre_len
#            if reduced>0.3: print("Thresh:",thres," Overlap accounted for:", reduced)
        except: pass

        # Prepare Returns
        p0 = df_ohlcv.Close[events.index]
        p1 = df_ohlcv.Close[events.t1.values].values
        if side == 'long': ret = (p1/p0 -1- commission)*leverage 
        else: ret = (1-p1/p0 - commission)*leverage
        ret = np.log(1+ret)
        daily_ret = ret.resample('1D').sum()
        daily_ret = daily_ret[daily_ret!=0].dropna()     

        # Compute Metrics
        if len(daily_ret) > 5:
            tot_ret = daily_ret.sum()
            std = np.sqrt((daily_ret**2).sum()) 
            sharpe = tot_ret / std * np.sqrt(tradingDays/dataDays)
            utility = tot_ret*max(0,min(6,sharpe))*tradingDays/dataDays
            sharpe_freq = len(ret)*tradingDays/dataDays * max(0,min(6,sharpe))
            if ret.std() == 0:
                raise ValueError ("SDJF:LSDKJF")
            zscore = ret.mean()/ret.std()*np.sqrt(len(ret))
        else:
            continue

        # Save
        tot_rets[thres]= tot_ret*tradingDays/dataDays
        sharpes[thres] = sharpe
        utilities[thres] = utility
        zscores[thres]= zscore
        freq[thres] = len(ret)/dataDays*tradingDays
        avgHP[thres] = (events.t1 - events.index).median().seconds/3600
        sharpe_freqs[thres] = sharpe_freq


    if visual:
        utilities.plot(label='utility');plt.legend();plt.show()
        sharpes.plot(label='sharpe'); plt.axhline(y=0); plt.legend();plt.show()
        tot_rets.plot(label='annual return'); plt.axhline(y=0); plt.legend();plt.show()
        zscores.plot(label='zscore'); plt.axhline(y=0); plt.legend();plt.show()
        freq.plot(label='annual freq'); plt.legend();plt.show()
        avgHP.plot(label='holding - hours'); plt.legend();plt.show()
        sharpe_freqs.plot(label='sharpe_freqs'); plt.legend();plt.show()
        
    return utilities, sharpes, tot_rets, zscores, sharpe_freqs

def getFeatUtility_tb (feat, side, quantiles, df_ohlcv, dataDays, crossing, vol, leverage, commission, tp = 0, sl =0, hpBar = None, hpDay = None, mid = None, trail_dist = None, trail_trigger = None, tradingDays = 250,visual = False, ):

    if (hpBar == None) & (hpDay == None):
        raise ValueError ("Either hp Bar of hp Day must be provided")

    tot_rets, sharpes, utilities, zscores, freq, avgHP,sharpe_freqs = pd.Series(index = quantiles, dtype=float), pd.Series(index= quantiles, dtype=float), pd.Series(index= quantiles, dtype=float), pd.Series(index= quantiles, dtype=float), pd.Series(index= quantiles, dtype=float), pd.Series(index= quantiles, dtype=float),pd.Series(index= quantiles, dtype=float)

    for thres in quantiles:

        # Prepare Events
        events = getCrossingEvents(feat = feat, df_ohlcv= df_ohlcv, thres = thres, crossing = crossing, mid= mid)
        if trail_dist is not None:
            t1 = trailing_exits(events.index, df_ohlcv, hpBar, vol, sl, trail_trigger, trail_dist, side)
        else:
            t1 = triple_barrier(events.index, df_ohlcv, hpBar, vol, tp, sl, side)
        events['t1'] = t1
        events = removeTimeOverlap(events)

        # Prepare Returns
        p0 = df_ohlcv.Close[events.index]
        p1 = df_ohlcv.Close[events.t1.values].values
        if side == 'long': ret = (p1/p0 -1 - commission)*leverage - commission
        else: ret = (1-p1/p0 - commission)*leverage - commission
        ret = np.log(1+ret)
        daily_ret = ret.resample('1D').sum()
        daily_ret = daily_ret[daily_ret!=0].dropna()     

        # Compute Metrics
        if len(daily_ret) > 5:
            tot_ret = daily_ret.sum()
            std = np.sqrt((daily_ret**2).sum()) 
            sharpe = tot_ret / std * np.sqrt(tradingDays/dataDays)
            utility = tot_ret*max(0,min(6,sharpe))*tradingDays/dataDays
            sharpe_freq = len(ret)*tradingDays/dataDays * max(0,min(6,sharpe))
            if ret.std() == 0:
                raise ValueError ("SDJF:LSDKJF")
            zscore = ret.mean()/ret.std()*np.sqrt(len(ret))
        else:
            continue

        # Save
        tot_rets[thres]= tot_ret*tradingDays/dataDays
        sharpes[thres] = sharpe
        utilities[thres] = utility
        zscores[thres]= zscore
        freq[thres] = len(ret)/dataDays*tradingDays
        avgHP[thres] = (events.t1 - events.index).median().seconds/3600
        sharpe_freqs[thres] = sharpe_freq


    if visual:
        utilities.plot(label='utility');plt.legend();plt.show()
        sharpes.plot(label='sharpe'); plt.axhline(y=0); plt.legend();plt.show()
        tot_rets.plot(label='annual return'); plt.axhline(y=0); plt.legend();plt.show()
        zscores.plot(label='zscore'); plt.axhline(y=0); plt.legend();plt.show()
        freq.plot(label='annual freq'); plt.legend();plt.show()
        avgHP.plot(label='holding - hours'); plt.legend();plt.show()
        sharpe_freqs.plot(label='sharpe_freqs'); plt.legend();plt.show()
        
    return utilities, sharpes, tot_rets, zscores, sharpe_freqs

def getFeatUtility_pairDE (feat, beta, pair_df, df1, df2, entry_quantiles, entry_crossing, exit_thres1, exit_cross1, exit_thres2, exit_cross2, dataDays, tradingDays, commission, mid = 0, minVol = None, maxVol = None, targetVol = None, vol = None, visual = False):

    tot_rets, sharpes, utilities, zscores, freq, avgHP = pd.Series(index = entry_quantiles, dtype=float),pd.Series(index = entry_quantiles, dtype=float),pd.Series(index = entry_quantiles, dtype=float),pd.Series(index = entry_quantiles, dtype=float),pd.Series(index = entry_quantiles, dtype=float),pd.Series(index = entry_quantiles, dtype=float)
    for entry_thres in entry_quantiles:

        events = getCrossingEvents_dynamicExit(feat, pair_df, entry_thres, exit_thres1, entry_crossing, exit_cross1,exit_thres2, exit_cross2, mid, minVol, vol)
        ret1 = get_lrets(events, df1, commission = commission, side='long', betSize = 1)
        ret2 = get_lrets(events, df2, commission = commission, side='short', betSize = beta)
        ret = ret1 + ret2
        if targetVol is not None:
            if vol is None: raise ValueError("vol is needed if maxVol is applied")
            lev_adj = vol[ret.index]/targetVol
            ret = np.log((np.exp(ret)-1)/lev_adj + 1)
        elif maxVol is not None:
            if vol is None: raise ValueError("vol is needed if maxVol is applied")
            lev_adj = vol[ret.index]/maxVol
            lev_adj[lev_adj<=1] = 1
            ret = np.log((np.exp(ret)-1)/lev_adj + 1)
        daily_ret = ret.resample('1D').sum()
        daily_ret = daily_ret[daily_ret!=0]

        # Compute Metrics
        if len(daily_ret) > 5:
            tot_ret = daily_ret.sum()
            std = np.sqrt((daily_ret**2).sum()) 
            sharpe = tot_ret / std * np.sqrt(tradingDays/dataDays)
            utility = tot_ret*max(0,min(6,sharpe))*tradingDays/dataDays
            if ret.std() == 0:
                raise ValueError ("SDJF:LSDKJF")
            zscore = ret.mean()/ret.std()*np.sqrt(len(ret))
        else:
            continue

        # Save
        tot_rets[entry_thres]= tot_ret*tradingDays/dataDays
        sharpes[entry_thres] = sharpe
        utilities[entry_thres] = utility
        zscores[entry_thres]= zscore
        freq[entry_thres] = len(ret)/dataDays*tradingDays
        avgHP[entry_thres] = (events.t1 - events.index).median().seconds/3600

    if visual:
        utilities.plot(label='utility');plt.legend();plt.show()
        sharpes.plot(label='sharpe'); plt.axhline(y=0); plt.legend();plt.show()
        tot_rets.plot(label='annual return'); plt.axhline(y=0); plt.legend();plt.show()
        zscores.plot(label='zscore'); plt.axhline(y=0); plt.legend();plt.show()
        freq.plot(label='annual freq'); plt.legend();plt.show()
        avgHP.plot(label='holding - hours'); plt.legend();plt.show()    

    return utilities, sharpes, tot_rets
        
def getFeatUtility_dynamicExit (feat, side, entry_quantiles, entry_crossing, exit_thres, exit_crossing, df_ohlcv, dataDays, tradingDays, leverage, commission, mid = 0, visual = False):
    tot_rets, sharpes, utilities, zscores, freq, avgHP = pd.Series(index = entry_quantiles, dtype=float),pd.Series(index = entry_quantiles, dtype=float),pd.Series(index = entry_quantiles, dtype=float),pd.Series(index = entry_quantiles, dtype=float),pd.Series(index = entry_quantiles, dtype=float),pd.Series(index = entry_quantiles, dtype=float)
    for entry_thres in entry_quantiles:
        events = getCrossingEvents_dynamicExit(feat, df_ohlcv, entry_thres, exit_thres, entry_crossing, exit_crossing, None, None, 0)
        ret = get_lrets(events, df_ohlcv, commission = commission, side=side, betSize = leverage)
        daily_ret = ret.resample('1D').sum()
        # Compute Metrics
        if len(daily_ret) > 5:
            tot_ret = daily_ret.sum()
            std = np.sqrt((daily_ret**2).sum()) 
            sharpe = tot_ret / std * np.sqrt(tradingDays/dataDays)
            utility = tot_ret*max(0,min(6,sharpe))*tradingDays/dataDays
            if ret.std() == 0:
                raise ValueError ("SDJF:LSDKJF")
            zscore = ret.mean()/ret.std()*np.sqrt(len(ret))
        else:
            continue

        # Save
        tot_rets[entry_thres]= tot_ret*tradingDays/dataDays
        sharpes[entry_thres] = sharpe
        utilities[entry_thres] = utility
        zscores[entry_thres]= zscore
        freq[entry_thres] = len(ret)/dataDays*tradingDays
        avgHP[entry_thres] = (events.t1 - events.index).median().seconds/3600
    
    if visual:
        utilities.plot(label='utility');plt.legend();plt.show()
        sharpes.plot(label='sharpe'); plt.axhline(y=0); plt.legend();plt.show()
        tot_rets.plot(label='annual return'); plt.axhline(y=0); plt.legend();plt.show()
        zscores.plot(label='zscore'); plt.axhline(y=0); plt.legend();plt.show()
        freq.plot(label='annual freq'); plt.legend();plt.show()
        avgHP.plot(label='holding - hours'); plt.legend();plt.show()        

def profitfactor(ret):
    ret = ret.resample('1D').sum()
    return ret[ret>0].sum()/-ret[ret<0].sum()

def get_cont(df_5m, events, method ='lret'):
    if method =='lret':
        p1 = df_5m.Close[events.t1.values].values
        p0 = df_5m.Close[events.index.values]
        cont = np.log(p1/p0)

    elif method =='sharpe':
        cont = pd.Series(index = events.index)
        for i, t0 in enumerate(events.index):
            t1 = events.t1.values[i]
            path = df_5m[t0:t1].Close
            lrets = np.log(path).diff()
            cont.loc[t0] = lrets.sum()/np.sqrt((lrets**2).sum())

    elif method =='slope':
        cont = pd.Series(index = events.index)
        for i, t0 in enumerate(events.index):
            t1 = events.t1.values[i]
            path = df_5m[t0:t1].Close
            if len(path) > 1:
                # This gets average slope of all points. 
                times = (path.index - t0).total_seconds() / 60  
                y0 = path.iloc[0]
                slopes = (path - y0) / times
                best_fit_slope = slopes.mean()
                cont.at[t0] = best_fit_slope            
    else:
        raise ValueError("Invalid Method")

    return cont

def vectorized_linear_regression(Y, X):
    # Compute means of X and Y across rows
    mean_x = np.mean(X, axis=1)
    mean_y = np.mean(Y, axis=1)

    # Compute the components of the formulas
    # Demean X and Y
    X_demeaned = X - mean_x[:, None]
    Y_demeaned = Y - mean_y[:, None]

    # Calculate numerator and denominator for the slope (a)
    numerator = np.sum(X_demeaned * Y_demeaned, axis=1)
    denominator = np.sum(X_demeaned**2, axis=1)

    # Slope (a)
    slope = numerator / denominator

    # Intercept (b)
    intercept = mean_y - slope * mean_x

    return slope, intercept

def scale_volumebar(df, n):
    df = df.reset_index()
    df['barID'] = df.index // n
    ohlcv_resampled = df.groupby('barID').agg({
        'Time0': 'first',
        'Time1': 'last',
        'Open': 'first',  # First entry in the group
        'High': 'max',   # Maximum of the group
        'Low': 'min',    # Minimum of the group
        'Close': 'last', # Last entry in the group
        'Volume': 'sum',  # Sum of the group
    })
    ohlcv_resampled.set_index('Time1', inplace=True, drop=False)
    ohlcv_resampled['barID'] = ohlcv_resampled.reset_index(drop=True).index.values
    return ohlcv_resampled

def visualize_tree(clf, column_names, nth_tree):
    from sklearn.tree import plot_tree
    plt.figure(figsize=(20,10))
    plot_tree(clf.estimators_[nth_tree], filled=True, feature_names=list(column_names), class_names=['Class 0', 'Class 1'], fontsize=10, proportion= True)
    plt.title('Visualization of the first tree in the RandomForest')
    plt.show()
    
def plotTripleBarrier(close, events, i):
    exp= close.index[close.index.searchsorted(events.index[i]+pd.Timedelta(days=0.5))]
    path = close[events.index[i]:exp]
    path.plot()
    plt.axvline(events.index[i])
    plt.axvline(events.t1[i])
    plt.axvline(exp)
    plt.axhline(close[events.index[i]]*(1+events.vol[i]))
    plt.axhline(close[events.index[i]]*(1-events.vol[i]))
    plt.show()


# Cross Validation Related
def time_series_cv_dates(num_splits, start_date, end_date, warm_up_splits=0, gap_days=0):
    # Validate the number of warm-up splits
    if warm_up_splits >= num_splits:
        raise ValueError("The number of warm-up splits must be less than the total number of splits.")
    
    # Convert start and end date to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Calculate the total number of days in the range
    total_days = (end_date - start_date).days + 1
    
    # Calculate the number of days per split
    days_per_split = total_days // num_splits
    
    # Initialize the start date for the first test period
    current_test_start = start_date
    
    for split in range(num_splits):
        # Determine the end date of the current test period
        if split == num_splits - 1:
            # Ensure the last split captures all remaining days
            test_end_date = end_date
        else:
            test_end_date = current_test_start + pd.Timedelta(days=days_per_split - 1)
        
        # Only start yielding splits after the warm-up splits
        if split >= warm_up_splits:
            # The training period ends with a gap before the test period starts
            train_start_date = start_date
            train_end_date = current_test_start - pd.Timedelta(days=gap_days + 1)

            # Yield train and test date ranges as two tuples
            yield ((train_start_date, train_end_date), (current_test_start, test_end_date))

        # Update the start date for the next test period
        # It starts immediately after the end of the previous test period
        current_test_start = test_end_date

def addT1toEvents(events, df_ohlcv, hpDay = None, hpBar = None):
        if hpDay is not None:
            t1 = getVertBarrier(events.index, df_ohlcv.Close, hpDay)
            events['t1'] = t1.values
        elif hpBar is not None: 
            events['t1'] = df_ohlcv.index[np.minimum(events['barID']+hpBar, len(df_ohlcv)-1)]
        return events



# Data Clean Up
def removeBarOverlap (events, df, minBars):
    minBars = round(minBars,0)
    events['i0'] = np.searchsorted(df.index,events.index)
    events['i1'] = np.searchsorted(df.index, events.t1)
    is_overlap = np.full(len(events),False)
    for j in range(0, len(events)-1): # Iterate through each event
        if is_overlap[j]:
            continue
        k = j+1
        overlapTime = (events.i0[k] < events.i0[j] + minBars ) 
        while overlapTime: 
            is_overlap[k] = True
            k += 1
            if k >= len(events): break
            overlapTime = (events.i0[k] < events.i0[j] + minBars ) 
    events = events[~is_overlap]
    return events

def removeTimeOverlap (events, mutliple = 1):
    is_overlap = np.full(len(events),False)
    for j in range(0, len(events)-1): # Iterate through each event
        if is_overlap[j]:
            continue
        k = j+1
        overlapTime = (events.t1[j]  > events.index[k]) 
        while overlapTime: 
            is_overlap[k] = True
            k += 1
            if k >= len(events): break
            overlapTime = (events.t1[j]  > events.index[k]) 
    events = events[~is_overlap]
    return events

def removeOverlap_2factor (Y, events, hpBars, vol, dBars, dPrice):
    is_overlap = np.full(len(events),False)
    for j in range(0, len(events)-1): # Iterate through each event
        if is_overlap[j]:
            continue
        k = j+1
        overlapTime = (events.barID[k] - events.barID[j]) < dBars*hpBars
        while overlapTime: # Now check dX
            overlapPrice = np.log(Y.entryP[k]/Y.entryP[j]) > -dPrice*vol[events.index[j]] # Check if the log price has decreased by more than the threshold in std
            if overlapPrice:
                is_overlap[k] = True
                k += 1
                if k >= len(events): break
                overlapTime = abs(events.barID[j] - events.barID[k]) < dBars*hpBars
            else:
                break
    return Y[~is_overlap], events[~is_overlap]

def correct_duplicate_timeindex (df_ohlcv):
    def find_duplicate_indices(df):
        # Getting the duplicate status for each index
        duplicate_mask = df.index.duplicated(keep=False)
        # Filter to get only duplicates
        duplicates = df[duplicate_mask]
        # Getting positions of duplicates
        positions = duplicates.index.to_series().groupby(duplicates.index).apply(lambda x: list(x.index))
        return positions
    # Usage
    duplicate_positions = find_duplicate_indices(df_ohlcv)
    df_ohlcv['Time0'] = df_ohlcv.index.values

    for dup in duplicate_positions.index:
        idx = np.searchsorted(df_ohlcv.index, dup)
        length = len(duplicate_positions[dup])
        for i in range(length):
            df_ohlcv['Time0'].iloc[idx+i]+= pd.Timedelta(microseconds=(i))

    df_ohlcv.reset_index(drop=True,inplace=True)
    df_ohlcv.set_index('Time0', drop= True, inplace=True)
    return df_ohlcv


# Bootstrapping
def stair_bootstrap (series, timestamps, last_date, block_size, num_blocks, num_iterations, p_group1 = 0.1, p_group2 = 0.3, p_group3=0.6):
    
    # We'll be working with numpy for maximum efficiency. 
    data = np.array(series.copy())
    timestamps = np.array(timestamps.copy())
    
    # Pre-compute
    n = len(data)
    total_num_blocks = n - block_size + 1                      
    block_timestamps = timestamps[:-block_size+1]
                      
    ### Weight Calculation ###
    block_weights = np.full(total_num_blocks, 0.0)
    group3_start_date = last_date - pd.DateOffset(months=6)
    group2_start_date = group3_start_date- pd.DateOffset(months=6)

    is_group1 = block_timestamps < group2_start_date
    is_group2 = (block_timestamps >= group2_start_date) & (block_timestamps < group3_start_date)
    is_group3 = block_timestamps >= group3_start_date

    block_weights[is_group1] = p_group1/sum(is_group1) if sum(is_group1)>0 else 0
    block_weights[is_group2] = p_group1/sum(is_group2) if sum(is_group2)>0 else 0
    block_weights[is_group3] = p_group1/sum(is_group3) if sum(is_group3)>0 else 0
    block_weights /= block_weights.sum()
        
    # Weighted Random Sampling starting index of the blocks.
    start_indices = np.random.choice(total_num_blocks, size=(num_blocks, num_iterations), p=block_weights)

    # Add the rest of the indices to the starting index of each block.
    block_offset = np.arange(block_size) # Creates an array of [0,1,.. block_size]
    all_indices = start_indices[:,:,np.newaxis] + block_offset[np.newaxis,np.newaxis,:]
    
    # Extracting the return figures with our full indices. 
    # This is 3D currently (block_num x iterations x block_size).
    bootstrapped_data = data[all_indices] 
    
    # Connecting different blocks into one array, and hence reducing down to 2d (iterations x block_size)
    concat_data = np.concatenate([bootstrapped_data[i] for i in range(num_blocks)], axis=1)
    
    return concat_data

def get_bootstrapped_indices (data_size, block_size, num_blocks, num_iterations, pct_diff):

    # Calculate what's needed for bootstrap
    max_block_idx = data_size - block_size
    discount_rate = (pct_diff)**(1/max_block_idx)
    weights = np.full(max_block_idx, discount_rate)
    weights = np.cumprod(weights)
    weights /= weights.sum()

    # Sampling starting index of the blocks.
    start_indices = np.random.choice(max_block_idx, size=(num_blocks, num_iterations), p=weights)

    # Add the rest of the indices to the starting index of each block.
    block_offset = np.arange(block_size) # Creates an array of [0,1,.. block_size]
    all_indices = start_indices[:,:,np.newaxis] + block_offset[np.newaxis,np.newaxis,:]

    return all_indices

def block_bootstrap (series, block_size, num_blocks, num_iterations, pct_diff=1):
    # We'll be working with numpy for maximum efficiency. 
    data = np.array(series.copy())

    # Calculate what's needed for bootstrap
    n = len(data)
    max_block_idx = n - block_size
    discount_rate = (pct_diff)**(1/max_block_idx)
    weights = np.full(max_block_idx, discount_rate)
    weights = np.cumprod(weights)
    weights /= weights.sum()

    # Sampling starting index of the blocks.
    start_indices = np.random.choice(max_block_idx, size=(num_blocks, num_iterations), p=weights)

    # Add the rest of the indices to the starting index of each block.
    block_offset = np.arange(block_size) # Creates an array of [0,1,.. block_size]
    all_indices = start_indices[:,:,np.newaxis] + block_offset[np.newaxis,np.newaxis,:]

    # Extracting the return figures with our full indices. 
    # This is 3D currently (block_num x iterations x block_size).
    bootstrapped_data = data[all_indices] 

    # Connecting different blocks into one array, and hence reducing down to 2d (iterations x block_size)
    concat_data = np.concatenate([bootstrapped_data[i] for i in range(num_blocks)], axis=1)

    return concat_data

def quickBootstrap (series, index_matrix, num_blocks):
    bootstrapped_data = np.array(series)[index_matrix]
    concat_data = np.concatenate([bootstrapped_data[i] for i in range(num_blocks)], axis=1)
    return concat_data

    
# 자잘한것들
def read_CSV(filepath, index = 'int'):
    df = pd.read_csv(filepath, index_col=0)
    try: df.Time = pd.to_datetime(df.Time, unit='ms')
    except: df.Time = pd.to_datetime(df.Time)    
    if index == 'Time': df.set_index('Time', inplace=True)
    return df

def roundDown (array, mod, room = 0.005):
    out =(array*(1+room)) - (array*(1+room))%mod
    return out

def binSupRes (n_group, series, binSize, fill_value, method = 'last'):
    # Iterates through each value within the series, and splits them into bins based on binSize.
    # Returns n groups in a list
    count, out, temp = 0, [], []
    for item in series:
        
        if len(temp) == 0:
            temp.append(item)

        else:
            if abs(item - temp[0]) <= binSize:
                temp.append(item)
            
            else:
                if method =='last':
                    out.append(temp[-1])
                elif method =='first':
                    out.append(temp[0])
                elif method == 'mean':
                    out.append(np.mean(temp))
                else:
                    raise ValueError("Invalid Method")
                temp = [item]
                count += 1
        
        if count >= n_group: 
            return out
    while count < n_group:
        out.append(fill_value)
        count += 1
    return out

def calculate_impurity_decrease(tree, node_index=0):
    # Base case: if the node is a leaf
    if tree.children_left[node_index] == tree.children_right[node_index] == -1:
        return 0

    # Recursive case: calculate for left and right children
    left_child_index = tree.children_left[node_index]
    right_child_index = tree.children_right[node_index]

    # Calculate the weighted impurity of current node and its children
    current_impurity = tree.impurity[node_index]
    left_impurity = tree.impurity[left_child_index]
    right_impurity = tree.impurity[right_child_index]

    left_samples = tree.weighted_n_node_samples[left_child_index]
    right_samples = tree.weighted_n_node_samples[right_child_index]
    total_samples = tree.weighted_n_node_samples[node_index]

    weighted_child_impurity = (left_impurity * left_samples + right_impurity * right_samples) / total_samples

    # Impurity decrease of the current node
    impurity_decrease = current_impurity - weighted_child_impurity

    # Recursively calculate impurity decrease for children and add to current node's decrease
    left_decrease = calculate_impurity_decrease(tree, left_child_index)
    right_decrease = calculate_impurity_decrease(tree, right_child_index)

    return impurity_decrease + left_decrease + right_decrease

def featCurveEntropy (x, Y, min_samples_leaf = 0.1, max_depth = 1, min_impurity_decrease = 0.0, visual=False):
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    clf = DecisionTreeClassifier(criterion ='entropy', max_depth = max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease)
    clf = clf.fit(x.values.reshape(-1,1), Y['bin'], sample_weight=Y['ret'].abs())
    if visual:
        plt.figure(figsize=(7,6)) 
        plot_tree(clf, filled=True, feature_names=['Feature1'], class_names=['0', '1'], rounded=False, proportion=True, precision=3); plt.show()
    tree = clf.tree_
    impurity_decrease = calculate_impurity_decrease(tree)
    featCurve =pd.Series(index = x, data = Y['ret'].values).sort_index()
    if visual:
        print("Entropy decrease:", impurity_decrease)
        featCurve.cumsum().plot()
        for i in range(len(tree.threshold)):
            if tree.threshold[i] == -2: continue
            plt.axvline(x=tree.threshold[i])
        plt.show()
        featCurve.reset_index()[0].cumsum().plot(); plt.show()
    return impurity_decrease

def applyPathFunc (events, close, numDays, func):
    '''Get volume weighted average by getting the price path iteratively for each row of event'''
    out = pd.Series(index=events.index, dtype=float)
    t0s = np.searchsorted(close.index, events.index - pd.Timedelta(days=numDays)) # Find the beginning of the 
    numOverflow = len(events) - (t0s>0).sum() # To adjust for data overflow
    t0s = close.index[t0s - 1] # Substract 1 as a way to round up. 
    for i in range(len(events)):
        if i < numOverflow: continue # Adjusting for data overflow
        t1 = events.index[i]
        t0 = t0s[i]
        path = close[t0:t1]
        out.iloc[i] = func(path)
    return out

def probaBinning (x, y, minBin = -10, maxBin = 0, interval = 0.5):
    out = []
    for lower in np.arange(minBin, maxBin+interval, interval):
        upper = lower + interval
        x_ = x[(x<=upper) & (x>lower)]
        if len(x_) == 0 : continue
        y_ = y[x_.index]
        prob = y_[y_>0].sum()/y_.abs().sum().round(3)
        out.append([np.median(x_.values),prob, len(x_.values), y_.sum(), y_.mean(), y_.median()])
    out = pd.DataFrame(out)
    out.columns = ['feature','prob', 'n_samples', 'sum', 'mean', 'median']
    return out

def sampling_hyper_search_(param, volbar_filepath):
    # Load Volume Bar
    df_volbar = pd.read_csv(volbar_filepath, index_col=0)
    df_volbar.Time = pd.to_datetime(df_volbar.Time)
    df_volbar.set_index('Time', inplace=True)
    close = df_volbar.Close

    # Get Volatility
    vol = getVol(df_volbar.Close, span = param['span0'], use_log=True, numDays = 0.5)

    # Cut off overly heteroscedastic parts
    cutoff = pd.to_datetime("2020-01-01")
    df_volbar = df_volbar[cutoff:]
    vol = vol[cutoff:]
    # vol.plot()
    # plt.ylim(0, 0.3)

    # Get Cusum Events
    h = vol*param['h_mul']
    tEvents, iEvents = get_cusum_events(np.log(df_volbar.Close), h, use_neg=True, use_pos=True,)
    # df = get_cusum_events(np.log(df_volbar.Close), h, use_neg=True, use_pos=True, debug_mode=True) # for debugging
    # df = df[cutoff:]
    # print("Opps per day:", pd.Series(index=tEvents, data=1).resample("D").sum().mean())

    # Apply Triple Barrier
    t1 = getVertBarrier(tEvents, df_volbar.Close, numDays = param['hold_per'])
    events = getEvents(df_volbar.Close, tEvents, [param['barrier'],param['barrier']], vol, minVol = 0.005, num_cpu = 1, t1= t1)
    t1 = events.t1
    # print('Pct samples touching horizontals:', ((events.t1-events.index) < pd.Timedelta(hours = 10)).sum()/len(events))

    # Get bins
    bins = getBins(events,close)

    # Get sample weights
    uWght = getUniqueWght(close.index, t1)
    rWght = pd.Series(abs(bins['ret']), index = bins.index)
    tWght = getTimeDecay(uWght, w0 = 0.2)
    wght = uWght * rWght * tWght
    wght = wght/wght.sum()

    # Eval
    score = uWght.mean() * (rWght.mean()-0.001) * len(t1)
    print()
    print("Opps per day:", pd.Series(index=tEvents, data=1).resample("D").sum().mean())
    print('Pct samples touching horizontals:', ((events.t1-events.index) < pd.Timedelta(hours = param['hold_per']*24)).sum()/len(events))
    print('AvgAbsRet:', rWght.mean(), 'Stdev:', rWght.std(), 'Z:',(rWght.mean()-0.001)/(rWght.std()) )
    print('Score:',score)
    print('ScoreBreakDown:',uWght.mean().round(3), (rWght.mean()-0.001).round(3), len(t1))
    print(param)
    return score, param

def sampling_hyper_search(params_grid, volbar_filepath, num_cpu):
    best_score = -np.inf
    best_param = None
    
    with ProcessPoolExecutor(max_workers=num_cpu) as executor:
        futures = [executor.submit(sampling_hyper_search_, param, volbar_filepath) for param in params_grid]
        
        for future in as_completed(futures):
            score, param = future.result()  # Extend this as needed based on what evaluate_params returns
            
            if score > best_score:
                best_score = score
                best_param = param
                print(f"#######New best score: {best_score} with params: {best_param}########")



# Functions from Lopez's Book
def mpPandasObj (func, pdObj, num_cpu=8, **kwargs):
    '''Returns in dataframes or series'''
    # Build job list
    chunks = linParts(len(pdObj), num_cpu)
    jobs = []
    for i in range(1,len(chunks)):
        job = [func, pdObj[chunks[i-1]:chunks[i]], kwargs]
        jobs.append(job)

    # Run Jobs
    if num_cpu > 1: 
        df = processJobs(jobs, num_cpu)
    else: 
        df = processJobs_(jobs) # debug mode
    df = df.sort_index()
    return df

def processJobs (jobs, num_cpu):
    # Parallel processing
    results = []
    with ProcessPoolExecutor(max_workers= num_cpu) as executor:
        futures = [executor.submit(job[0], job[1], **job[2]) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result)
    df = pd.concat(results, axis = 0)
    return df

def processJobs_ (jobs):
    # Sequential processing for debugging
    results = []
    for job in jobs:
        result = job[0](job[1], **job[2])
        results.append(result)
    df = pd.concat(results, axis = 0)
    return df

def linParts(numAtoms,num_cpu):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(num_cpu,numAtoms)+1) 
    parts=np.ceil(parts).astype(int)
    return parts

def getPathMatrix(close, tIn, nbars, scale=1, grouping='last', look_direction='backward'):
    integer_index = close.index.get_indexer(tIn)
    if look_direction == 'backward':
        path_indices = integer_index.reshape(-1, 1) + np.arange(-(nbars * scale - 1), 1)
    elif look_direction == 'forward':
        path_indices = integer_index.reshape(-1, 1) + np.arange(0, nbars * scale)
    else:
        raise ValueError("look_direction must be either 'backward' or 'forward'")
    
    path_indices = path_indices.reshape(path_indices.shape[0], nbars, -1)
    mask = path_indices<0
    mask = (path_indices < 0) | (path_indices >= len(close))
    path_indices[mask] = 0
    price_path = close.values[path_indices]
    price_path[mask] = np.nan
    if grouping == 'last':
        price_path = price_path[:,:,-1]
    elif grouping == 'sum':
        price_path = np.sum(price_path, axis =2)
    elif grouping == 'avg':
        price_path = np.mean(price_path, axis =2)
    elif grouping == 'max':
        price_path = np.max(price_path, axis =2)
    elif grouping == 'min':
        price_path = np.min(price_path, axis =2)
    else:
        raise ValueError("Invalid grouping option")
    
    return pd.DataFrame(data = price_path, index = close.index[integer_index])

def getTimeDecay(uniq_w, w0):
    # apply piecewise-linear decay to observed uniqueness
    # newest observation gets weight=1, oldest observation gets weight = w0
    # w0 could be negative, in which memory is cut off even earlier.
    uniq_cumsum =uniq_w.sort_index().cumsum()
    if w0 >= 0:
        slope=(1.-w0)/uniq_cumsum.iloc[-1] 
    else: 
        slope=1./((w0+1)*uniq_cumsum.iloc[-1])
    const=1.-slope*uniq_cumsum.iloc[-1]

    decay_w = const+slope*uniq_cumsum
    decay_w[decay_w<0]=0
    return decay_w

def getVol(close,span, min_periods = 0,numBars = None, numDays = None, use_log = False, ):
    # daily vol, reindexed to close 
    if numDays is not None:
        df0=close.index.searchsorted(close.index-pd.Timedelta(days=numDays))
    elif numBars is not None:
        df0=close.index.searchsorted(close.index) - numBars
    else:
        raise ValueError("either numBars or numDays must be provided")
    df0 = df0[df0>0]
    df0=pd.Series(data =close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    if use_log: 
        df0=np.log(close.loc[df0.index]/(close.loc[df0.values].values))
    else:
        df0=close.loc[df0.index]/(close.loc[df0.values].values)-1 # daily returns 
    df0=df0.ewm(span=span, min_periods = min_periods).std()
    return df0

def getROC(close, numBars = None, numDays = None, use_log = False, span = None):
    # daily vol, reindexed to close 
    if numDays is not None:
        df0=close.index.searchsorted(close.index-pd.Timedelta(days=numDays))
    elif numBars is not None:
        df0=close.index.searchsorted(close.index) - numBars
    else:
        raise ValueError("either numBars or numDays must be provided")
    df0 = df0[df0>0]
    df0=pd.Series(data =close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    if use_log: 
        df0=np.log(close.loc[df0.index]/(close.loc[df0.values].values))
    else:
        df0=close.loc[df0.index]/(close.loc[df0.values].values)-1 # daily returns 
    if span is not None:
        df0 = df0.ewm(span=span).mean()
    return df0

def getPastValue(series, numBars = None, numDays = None):
    # daily vol, reindexed to close 
    if numDays is not None:
        df0=series.index.searchsorted(series.index-pd.Timedelta(days=numDays))
    elif numBars is not None:
        df0=series.index.searchsorted(series.index) - numBars
    else:
        raise ValueError("either numBars or numDays must be provided")
    df0 = df0[df0>0]
    df0=pd.Series(data =series.values[df0-1], index=series.index[series.shape[0]-df0.shape[0]:])
    return df0

def get_cusum_events (close_raw, h_array, use_neg, use_pos, debug_mode = False):
    if len(close_raw)!= len(h_array):
        raise ValueError("Unequal data length")
    tEvents,sPos,sNeg=[],0,0 
    diff=close_raw.diff()
    p, n = [], []
    for i, time in enumerate(diff.index[1:]):
        h = h_array[1:][i]
        sPos,sNeg=max(0,sPos+diff.loc[time]),min(0,sNeg+diff.loc[time]) 
        p.append(sPos); n.append(sNeg)
        if use_neg & (sNeg<-h):
            sNeg=0;tEvents.append(time);
        if use_pos & (sPos>h):
            sPos=0;tEvents.append(time) 

    iEvents = [close_raw.index.get_loc(timestamp) for timestamp in tEvents]
    if debug_mode:
        return pd.DataFrame({'pos': p, 'neg': n}, index=diff.index[1:])

    else:
        return pd.DatetimeIndex(tEvents), np.array(iEvents)

def getVertBarrier(tEvents, close, numDays = None, numBars = None):
    if numDays is not None:
        t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
        t1=t1[t1<close.shape[0]] 
        t1=pd.Series(close.index[t1],index=tEvents[:t1.shape[0]])
    elif numBars is not None:
        t1=close.index.searchsorted(tEvents) + numBars
        t1=t1[t1<close.shape[0]] 
        t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])

    return t1

def applyPtSlOnT1(events,close, ptSl):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    out=events[['t1']].copy(deep=True)
    if ptSl[0]>0:pt=ptSl[0]*events['vol'] 
    else:pt=pd.Series(index=events.index, dtype = float) # NaNs
    if ptSl[1]>0:sl=-ptSl[1]*events['vol'] 
    else:sl=pd.Series(index=events.index, dtype = float) # NaNs
    for loc,t1 in events['t1'].fillna(close.index[-1]).items():
        df0=close[loc:t1] # path prices 
        df0=(df0/close[loc]-1)*events.at[loc,'side'] # path returns 
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss. 
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking.
    return out

def getEvents(close,tEvents,ptSl,vol,minVol,num_cpu,t1=False):
    #1) get target
    vol=vol.loc[tEvents]
    vol=vol[vol>minVol] # minRet
    #2) get t1 (max holding period)
    if t1 is False: t1=pd.Series(pd.NaT,index=tEvents)
    #3) form events object, apply stop loss on t1 
    side_=pd.Series(1.,index=vol.index) 
    events=pd.concat({'t1':t1,'vol':vol,'side':side_}, axis=1).dropna(subset=['vol']) 
    df0=mpPandasObj(func=applyPtSlOnT1,pdObj= events, num_cpu=num_cpu,close=close,ptSl=ptSl) 
    events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan 
    events=events.drop('side',axis=1)
    events.dropna(inplace=True)
    return events

def getBins(events,close, use_log, commission = 0, side = 'long'):
    #1) prices aligned with events
    events_=events.dropna(subset=['t1']) 
    px=events_.index.union(events_['t1'].values).drop_duplicates() 
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index) 
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if side == 'long':
        out['bin']=(out['ret']>0).astype(int)
    else:
        out['bin']=(out['ret']<0).astype(int)
        out['ret'] = -1*out['ret']
    out['ret'] -= 2*commission
    if use_log: out['ret']=np.log(1+out['ret'])        
    return out

def getUniqueWght(closeIdx, t1):
    ''' I have combined mpNumCoEvents and mpSampleTW from Lopez's code'''
    # Count overlaps for each bar
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()])) 
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.items():
        count.loc[tIn:tOut]+=1. 

    # Get weights
    wght = pd.Series(index = t1.index, dtype = float)
    for tIn, tOut in t1.items():
        wght.loc[tIn] = (1./count.loc[tIn:tOut]).mean()        
    return wght



# Adjusted function to include the entire last day of each period
def convert_fold_to_datetime(folds):
    result = []
    for start, end in folds:
        start_adj = pd.to_datetime(start, utc=True)
        end_adj = pd.to_datetime(end, utc=True) + pd.DateOffset(months=1) - pd.Timedelta(seconds=1)
        result.append((start_adj,end_adj))
    return result

def find_threshold(X, transf, target_per_month, band_side, overlap_dist, increment = 0.05, tolerance=0.1):
    # Determine the direction of monotonicity by comparing the outputs at low and high
    total_months = (transf.index[-1] - transf.index[0]).days/30
    target = total_months * target_per_month

    if band_side == 'lower':
        low = transf.min()
        high = transf.median()
        increasing = True
    else:
        low = transf.median()
        high = transf.max()
        increasing = False
    
    while low <= high:
        mid = (low + high) / 2
        mid_val = X(transf, mid, overlap_dist)
        
        lower_bound = target * (1 - tolerance)
        upper_bound = target * (1 + tolerance)
        
        if lower_bound <= mid_val <= upper_bound:
            return mid  # The mid value's output is within the target range
        
        # Adjust search based on direction of monotonicity and comparison with the target bounds
        if increasing:
            if mid_val < lower_bound:
                low = mid + increment
            else:
                high = mid - increment
        else:
            if mid_val > upper_bound:
                low = mid + increment
            else:
                high = mid - increment
    
    if tolerance <= 0.25:
        find_threshold(X, transf, target_per_month, band_side, overlap_dist, increment, tolerance*1.5)
    if increment >= 0.01:
        find_threshold(X, transf, target_per_month, band_side, overlap_dist, increment/2, tolerance)
    else:
        print("Couldn't find threshold")

def get_sample_index (transf, threshold, overlap_dist, center = 0, crossing = 'exit'):
    nan_mask = np.isnan(transf)
    transf[nan_mask] = 0
    if (threshold > center) == (crossing == 'exit'):
        signal = (transf >= threshold) & (transf.shift() < threshold)
    else:
        signal = (transf <= threshold) & (transf.shift() > threshold)
    signal = signal & (signal.shift().rolling(overlap_dist).sum()==0)
    time_indices = np.array(transf.index[signal])
    integer_indices = np.array(transf.reset_index().index[signal])
    return time_indices, integer_indices

def get_control_index (ohlcv, samples_per_month):
    total_months = (ohlcv.index.max() - ohlcv.index.min()).days/30
    total_samples = int(round(samples_per_month * total_months))
    int_indices = np.floor(np.linspace(0, len(ohlcv.index) - 1, total_samples)).astype(int)
    control_indices = ohlcv.index[int_indices]
    return control_indices

def correct_first_line_errors(value):
    if value.count('.') > 1:  # Check if there are more than one decimal points
        parts = value.split('.')
        # Keep the first part, the first decimal point, and concatenate the rest without additional decimal points
        corrected_value = parts[0] + '.' + ''.join(parts[1:])
    else:
        corrected_value = value

    
    if isinstance(corrected_value, str):
        # Text number to number
        if corrected_value.isdigit():
            corrected_value = float(corrected_value)
        # Text bool to boolean
        elif corrected_value.lower() == "true":
            corrected_value = True
        elif corrected_value.lower() == "false":
            corrected_value = False            
    return corrected_value


# Visualization Prep
def visualize_by_index(plottype,tickers_subset, df, i, size, palette='dark'):
    
    col_name = df.columns[i]
    # Filter dataframe for tickers in the current subset
    df_subset = df[df['ticker'].isin(tickers_subset)]
    
    plt.figure(figsize=(15, 6))

    if plottype == 'swarm':
        sns.swarmplot(x='ticker', y=col_name, data=df_subset, size=size, palette=palette)
    elif plottype == 'strip':
        sns.stripplot(x='ticker', y=col_name, data=df_subset, size=size, palette=palette)
    elif plottype == 'hist':
        sns.histplot(x='ticker', y=col_name, data=df_subset, palette=palette)                     
    elif plottype == 'violin':
        sns.violinplot(x='ticker', y=col_name, data=df_subset, size=size, palette=palette)                                
    elif plottype == 'box':
        sns.boxplot(x='ticker', y=col_name, data=df_subset, palette=palette)                                                
    
    # Highlighting the min and max values for each ticker
    for ticker in tickers_subset:
        # Filter for the current ticker
        ticker_df = df_subset[df_subset['ticker'] == ticker]
        
        # Find the min and max values for this ticker
        min_val = ticker_df[col_name].min()
        max_val = ticker_df[col_name].max()
        
        # Highlight min and max values
        # Find rows that have the min and max value, then plot them with larger dots
        min_row = ticker_df[ticker_df[col_name] == min_val]
        max_row = ticker_df[ticker_df[col_name] == max_val]
        
        plt.scatter(x=min_row['ticker'], y=min_row[col_name], color='red', s=100, edgecolor='black', zorder=5)
        plt.scatter(x=max_row['ticker'], y=max_row[col_name], color='green', s=100, edgecolor='black', zorder=5)

    plt.title(f'{col_name} (Subset)')
    plt.xticks(rotation='vertical')
    plt.show()

def visualize_by_col(plottype,tickers_subset, df, column_names, size, palette='dark'):
    
    for col_name in column_names:
        # Filter dataframe for tickers in the current subset
        df_subset = df[df['ticker'].isin(tickers_subset)]
        
        plt.figure(figsize=(15, 6))

        if plottype == 'swarm':
            sns.swarmplot(x='ticker', y=col_name, data=df_subset, size=size, palette=palette)
        elif plottype == 'strip':
            sns.stripplot(x='ticker', y=col_name, data=df_subset, size=size, palette=palette)
        elif plottype == 'hist':
            sns.histplot(x='ticker', y=col_name, data=df_subset, palette=palette)                     
        elif plottype == 'violin':
            sns.violinplot(x='ticker', y=col_name, data=df_subset, size=size, palette=palette)                                
        elif plottype == 'box':
            sns.boxplot(x='ticker', y=col_name, data=df_subset, palette=palette)                                                
        
        # Highlighting the min and max values for each ticker
        for ticker in tickers_subset:
            # Filter for the current ticker
            ticker_df = df_subset[df_subset['ticker'] == ticker]
            
            # Find the min and max values for this ticker
            min_val = ticker_df[col_name].min()
            max_val = ticker_df[col_name].max()
            
            # Highlight min and max values
            # Find rows that have the min and max value, then plot them with larger dots
            min_row = ticker_df[ticker_df[col_name] == min_val]
            max_row = ticker_df[ticker_df[col_name] == max_val]
            
            plt.scatter(x=min_row['ticker'], y=min_row[col_name], color='red', s=100, edgecolor='black', zorder=5)
            plt.scatter(x=max_row['ticker'], y=max_row[col_name], color='green', s=100, edgecolor='black', zorder=5)

        plt.title(f'{col_name} (Subset)')
        plt.xticks(rotation='vertical')
        plt.show()    

        
#### Signal Functions ###

def calc_atr (df, lookback = 30, window = 1):
    hl = abs(df.High.rolling(window).max() - df.Low.rolling(window).min())
    hc = abs(df.High.rolling(window).max() - df.Close.shift(window))
    lc = abs(df.Low.rolling(window).max() - df.Close.shift(window))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window=lookback).mean()
    return atr

def calc_sma (series, lookback = 30):
    return series.rolling(lookback).mean()

def calc_vwap (df, lookback = 30):
    pv = df.Close * df.Volume
    rollsum_pv = pv.rolling(window=lookback).sum()
    rollsum_v = df.Volume.rolling(window=lookback).sum()
    vwap = rollsum_pv / rollsum_v
    return vwap    

def calc_vwstd(df, lookback):
    """ Calculate the volume-weighted standard deviation """
    vwap = calc_vwap(df, lookback)
    # Squared deviation from VWAP
    squared_deviation = (df['Close'] - vwap)**2
    # Volume-weighted sum of squared deviations
    vw_sum_of_squares = (squared_deviation * df['Volume']).rolling(window=lookback).sum()
    # Volume over the window
    rolling_volume = df['Volume'].rolling(window=lookback).sum()
    # Volume-weighted variance
    vw_variance = vw_sum_of_squares / rolling_volume
    # Standard deviation is the square root of variance
    vw_std_dev = vw_variance**0.5
    return vw_std_dev

def regression2d (Y, X = None):

    # Step 1: Prepare X (predictor) and Y (response) for linear regression
    n, m = Y.shape
    if X is None:
        X = np.vstack([np.arange(m), np.ones(m)]).T  # Add a column of ones for intercept

    # Step 2: Fit linear regression for each series using vectorized operations
    B = np.linalg.inv(X.T @ X) @ X.T @ Y.T  # B = (X'X)^-1 X'Y, for all series at once

    # Step 3: Calculate residuals and their standard deviation for each series
    predicted = X @ B  # Predicted values for each existing data point
    residuals = Y - predicted.T

    return B.T, residuals

def regression2d_tvalue(Y, X=None):
    # Step 1: Prepare X (predictor) and Y (response) for linear regression
    n, m = Y.shape
    if X is None:
        X = np.vstack([np.arange(m), np.ones(m)]).T  # Add a column of ones for intercept

    # Step 2: Fit linear regression for each series using vectorized operations
    XtX_inv = np.linalg.inv(X.T @ X)  # Inverse of X'X
    B = XtX_inv @ X.T @ Y.T  # B = (X'X)^-1 X'Y, for all series at once

    # Step 3: Calculate residuals and their variance
    predicted = X @ B  # Predicted values for each existing data point
    residuals = Y - predicted.T
    RSS = (residuals**2).sum(axis=1)  # Residual sum of squares
    sigma_squared = RSS / (m - X.shape[1])  # Variance of residuals

    # Step 4: Calculate standard error for the slope coefficient
    se_slope = np.sqrt(sigma_squared[:, np.newaxis] * XtX_inv[0, 0])  # Extract variance of slope

    # Step 5: Calculate t-values for the slope coefficient
    t_values = B[0] / se_slope.flatten()  # t-value = estimated_coefficient / standard_error

    return t_values

def price_vol_zscore (df_ohlcv, params):
    lookback = params['lookback']
    log_vol = np.log(df_ohlcv.Volume + 2.719)
    pv = log_vol*df_ohlcv.Close
    atr = calc_atr(df_ohlcv, lookback)*log_vol
    zscore = (pv - pv.rolling(lookback).mean())/atr
    return zscore

def vol_zscore (df_ohlcv, params):
    lookback = params['lookback']
    log_vol = np.log(df_ohlcv.Volume + 2.719)
    zscore = (log_vol - log_vol.rolling(lookback).mean())/log_vol.rolling(lookback).std()
    return zscore

def get_normalizer(df, type, lookback):
    if type == 'atr':
        normalizer = calc_atr(df, lookback).values
    elif type == 'std':
        normalizer = df.Close.rolling(lookback).std().values
    elif type =='close':
        normalizer = df.Close.shift(lookback).values
    elif type == None:
        normalizer = 1
    else:
        raise ValueError("Invalid normalizer")
    normalizer[normalizer==0] = np.nan
    return normalizer

def bollinger_vwap (df_ohlcv, params):
    lookback = params['lookback']
    vwap = calc_vwap(df_ohlcv, lookback)
    std = df_ohlcv.Close.rolling(lookback).std()
    deviation = (df_ohlcv.Close - vwap)/std
    return deviation

def bollinger_std (df_ohlcv, params):
    lookback = params['lookback']
    sma = calc_sma(df_ohlcv.Close, lookback)
    std = df_ohlcv.Close.rolling(lookback).std()
    blgr = (df_ohlcv.Close - sma)/std
    return blgr

def consec_count(df_ohlcv, param = {'series': None}):

    if param['series'] == 'Close':
        series = df_ohlcv.Close
    elif param['series'] == 'High':
        series = df_ohlcv.High
    elif param['series'] == 'Low':
        series = df_ohlcv.Low
    else:
        raise ValueError ("Unknown input for series type. Select 1 from Close, High and Low")

    # Calculate the difference between consecutive elements
    diff = series.diff()

    # Identify increasing and decreasing sequences
    increasing = diff > 0
    decreasing = diff < 0

    # Group by the changes in sequence (from increasing to decreasing and vice versa)
    groups = increasing.ne(increasing.shift()).cumsum()

    # Assign positive values to increases and negative values to decreases
    result = increasing.groupby(groups).cumsum() - decreasing.groupby(groups).cumsum()

    return result

def sigfig_cross (df_ohlcv, params = {'n_sigfig':2}):
    # 유의미한 저항선 = Sigfig 두번째 자리가 바뀔때
    sigfigs = pd.Series(extract_first_n_digits(df_ohlcv.Close, params['n_sigfig']))
    sigfigs_shifted = sigfigs.shift(1)
    signal = (sigfigs > sigfigs_shifted).astype(int) - (sigfigs < sigfigs_shifted).astype(int)
    return signal

def buyer_v_taker (df_ohlcv, lookback = 30):
    takerV = df_ohlcv['Taker buy base asset volume'].rolling(lookback).mean()
    totalV = df_ohlcv['Volume'].rolling(lookback).mean()
    return takerV/totalV
    
def FX (df_ohlcv, df_ohlcv2, lookback = 30):
    # Calculate FX
    avg_C = df_ohlcv.Close.rolling(lookback).mean()
    avg_C2 = df_ohlcv2.Close.rolling(lookback).mean()
    fx = avg_C2/avg_C

    # Calculated expected price & deviation from exp price
    exp_price = fx*df_ohlcv.Close
    dev = df_ohlcv.Close - exp_price

    # Normalize with atr
    atr = calc_atr(df_ohlcv, lookback)
    normalized_deviation = dev/atr
    return normalized_deviation



# Helper Functions

def get_sigfig(arr, n_sig_figs, round_up):
    """
    Round each number in the array to the next value with specified significant figures,
    either up or down, while maintaining the original scale of the number.

    Parameters:
    arr (numpy array): Array of floats
    n_sig_figs (int): Number of significant figures to round to
    round_up (bool): If True, round up, else round down

    Returns:
    numpy array: Array of floats rounded to the specified significant figures.
    """
    def round_num(num):
        if num == 0:
            return 0
        # Determine the order of magnitude of the number
        order = math.floor(math.log10(abs(num)))
        # Scale the number so that the last significant digit is in the tens place
        scaled_num = abs(num) / (10 ** order)
        # Round up or down and scale back
        if round_up:
            rounded_scaled_num = math.ceil(scaled_num * (10 ** (n_sig_figs - 1))) / (10 ** (n_sig_figs - 1))
        else:
            rounded_scaled_num = math.floor(scaled_num * (10 ** (n_sig_figs - 1))) / (10 ** (n_sig_figs - 1))
        # Rescale to the original order of magnitude and apply the original sign
        return np.sign(num) * rounded_scaled_num * (10 ** order)

    # Apply the round_num function to each element in the numpy array
    vfunc = np.vectorize(round_num)
    return vfunc(arr)

def resample_into_daily(signal_lrets_2d, freq, signal_index_1d):
    n_per_day = 24*60//freq # This is number of candles per day
    signal_day_index_1d = signal_index_1d//n_per_day
    unique_day_index_1d, inverse_indices = np.unique(signal_day_index_1d, return_inverse=True)
    signal_daily_lrets_2d = np.zeros((len(unique_day_index_1d), signal_lrets_2d.shape[1]))
    np.add.at(signal_daily_lrets_2d, inverse_indices, signal_lrets_2d)
    return signal_daily_lrets_2d, unique_day_index_1d

def convert_index_to_boolean(signal_bool_2d_template,signal_index_2d ):
    signal_bool_2d = signal_bool_2d_template.copy()
    signal_index_2d[np.isnan(signal_index_2d)] = -1 # Replace nan with -1 for now
    signal_index_2d = signal_index_2d.astype(int) # Convert to integers so that we can do advanced indexing
    signal_bool_2d[signal_index_2d] = True 
    signal_bool_2d[-1,:] = False # The last row has become all false because we replaced nan with -1. Undoing this. 
    return signal_bool_2d

def fetch_matrix_values(index_matrix, target_matrix, fill_value=np.nan):
    '''This function is intended to use a matrix of index number/position numbers to get the corresponding
    values in another matrix.'''
    # Create index_matrix with nans replaced with 0
    adjusted_indices = np.nan_to_num(index_matrix, nan=0).astype(int)

    # Create a mask marking non-nans
    valid_mask = ~np.isnan(index_matrix)

    # Use np.take_along_axis to fetch values based on adjusted indices
    # Ensure the result is of a floating-point type to accommodate NaN
    fetched_values = np.take_along_axis(target_matrix.astype(float), adjusted_indices, axis=0)

    # Replace the values fetched using invalid indices with the fill_value
    fetched_values[~valid_mask] = fill_value

    return fetched_values

def rolling_sum_next_n(arr, n, fill_value=np.nan):
    # Compute the cumulative sum
    cumsum_arr = np.cumsum(arr)

    # Pad the end of the cumsum array with n zeros (for sum calculation) and one zero (for alignment)
    cumsum_arr_padded = np.append(cumsum_arr, [0] * (n + 1))

    # Calculate the rolling sum by subtracting the cumsum array from its shifted version
    rolling_sum = cumsum_arr_padded[n:-1] - cumsum_arr_padded[:-n-1]

    # Replace the last 'n' elements with the specified fill value
    rolling_sum[-n:] = fill_value

    return rolling_sum

def extract_first_n_digits (arr,n):
    """
    Extract the first two non-zero digits from each float in a numpy array.

    Parameters:
    arr (numpy array): Array of floats

    Returns:
    numpy array: Array of strings with the first two non-zero digits.
    """
    # Convert to string
    str_arr = np.array(arr, dtype=str)

    # Remove decimal points
    str_arr = np.char.replace(str_arr, '.', '')

    # Define a vectorized function for extracting digits
    vfunc = np.vectorize(lambda x: ''.join([d for d in x if d != '0'])[:n])

    # Apply the function
    return vfunc(str_arr)

def signal_weight (signal_series, n_candles):

    #1) Add rows to the beginning and end for shifting
    new_rows = pd.Series([np.nan] * n_candles)
    signal_series_extended = pd.concat([new_rows, signal_series.astype(int), new_rows]).fillna(0)

    #2) Get overlap count through rolling sum, closed
    ret_overlaps = signal_series_extended.shift(1).rolling(window=n_candles).sum().fillna(0)

    #3) Get inverse
    ret_weight = pd.Series(np.where(ret_overlaps == 0, 0, 1/ret_overlaps))

    #4) Normalize with candle bar count
    ret_weight = ret_weight/n_candles

    #5) Shift up
    ret_weight = ret_weight.shift(-n_candles)

    #6) Roll sum
    ret_weight = ret_weight.rolling(n_candles).sum()

    #7) Cut off, reindex
    ret_weight = ret_weight[n_candles:-n_candles]
    ret_weight.index = signal_series.index

    #8) Mask
    signal_weight = ret_weight*signal_series

    return signal_weight



######################## Deprecated ####################################
######################## Deprecated ####################################
######################## Deprecated ####################################
######################## Deprecated ####################################
######################## Deprecated ####################################
######################## Deprecated ####################################

# def save_log (log_message, log_filepath):
#     current_time = datetime.now().time().strftime("%H:%M")
#     log_message = str(current_time+"    "+log_message)        
#     print(log_message)
#     with open(log_filepath, 'a') as file:
#         file.write(log_message + '\n')

# def find_threshold_v2 (transf, target_per_month, overlap_dist, bounds, center = 0, crossing = 'exit'):
#     def obj_func (threshold, transf, target_size, overlap_dist, center, crossing):

#         if (threshold > center) == (crossing == 'exit'):
#             signal = (transf >= threshold) & (transf.shift() < threshold)
#         else:
#             signal = (transf <= threshold) & (transf.shift() > threshold)
#         signal = signal & (signal.shift().rolling(overlap_dist).sum() == 0)
#         return abs(signal.sum() - target_size)
    
#     total_months = (transf.index[-1] - transf.index[0]).days/30
#     target_size = total_months * target_per_month

#     result = minimize_scalar(obj_func, args=(transf, target_size, overlap_dist, center, crossing),
#                          bounds=bounds, method='bounded')

#     return result.x

# def find_threshold_v3(transf, target_per_month, overlap_dist, bounds, center=0, crossing='exit'):
#     def obj_func(threshold):
#         if (threshold[0] > center) == (crossing == 'exit'):
#             signal = (transf >= threshold[0]) & (transf.shift() < threshold[0])
#         else:
#             signal = (transf <= threshold[0]) & (transf.shift() > threshold[0])
#         signal = signal & (signal.shift().rolling(overlap_dist).sum() == 0)
#         return abs(signal.sum() - target_size)
    
#     total_months = (transf.index[-1] - transf.index[0]).days / 30
#     target_size = total_months * target_per_month
    
#     # Initial guess for the threshold
#     initial_guess = [(bounds[0] + bounds[1]) / 2]

#     # Define minimizer_kwargs for the local optimization step
#     minimizer_kwargs = {"method": "L-BFGS-B", "bounds": [bounds]}

#     # Use basinhopping with the objective function and initial guess
#     result = basinhopping(obj_func, initial_guess, minimizer_kwargs=minimizer_kwargs, niter=100)

#     return result.x[0]

# def get_lrets_lvg (strat, dir_path, leverage):
#     file = strat['file']
#     price_action = getattr(signals, strat['strategy'])
#     threshold = strat['threshold']
#     direction = strat['cross_dir']
#     side = strat['side']
#     timeframe = strat['timeframe']
#     barrier = ast.literal_eval(strat['barrier'])
#     #leverage = strat['leverage']
#     timezone = strat['timezone']
#     dayofweek = strat['dayofweek']
#     tradingsession = strat['tradingsession']
#     macro_tf = strat['macro_tf']
#     macro_threshold = strat['macro_threshold']
#     macro_dir = strat['macro_direction']
    
#     df_ohlcv = pd.read_csv(os.path.join(dir_path, file))
#     df_ohlcv.Time = pd.to_datetime(df_ohlcv.Time, utc=True)  
    
#     if macro_tf != "all":
#         candle_ratio = int(timeframe_to_minutes(macro_tf)/timeframe_to_minutes(timeframe)) # Candle ratio means how many micro candles we would need to construct for one macro candle. 
#         macro_blgr_values = signals.get_macro_blgr(df_ohlcv, candle_ratio)

    
#     signal_priceAction, df_entry_priceAction = price_action(df_ohlcv, threshold, direction)
#     signal_time = signals.time_constraint(df_ohlcv.Time, timezone, dayofweek, tradingsession)
#     signal_priceTime = signal_priceAction & signal_time
#     if macro_tf != "all":
#         if macro_dir == "above":
#             signal_macro = macro_blgr_values > macro_threshold
#         elif macro_dir == "below":
#             signal_macro = macro_blgr_values < macro_threshold
#         elif macro_dir == "around":
#             signal_macro = (macro_threshold-1 < macro_blgr_values) & (macro_blgr_values < macro_threshold+1)
#         signal = signal_priceTime & signal_macro
#     else:
#         signal = signal_priceTime

#     clean_indices = removeOverlaps_1d (signal, barrier[2])
#     pct_overlap = 1-len(clean_indices)/signal.sum()
#     df_entry = df_entry_priceAction.loc[clean_indices]
#     df_result = triple_barrier_dfstyle(df_ohlcv, df_entry, barrier, side) 

#     # Bootstrap the outlier-adjusted returns (output is 2d np.array: iterations x block_size)
#     rets = df_result.rets - 2*0.0008
#     #rets_adj = rets[rets <rets.quantile(0.97)]
#     rets_adj = rets
#     rets_adj_levered = rets_adj*leverage 
#     lrets_adj_levered = np.log(1+rets_adj_levered)

#     lrets_adj_levered.index = df_ohlcv.Time[lrets_adj_levered.index]

#     return lrets_adj_levered

# def get_lrets (strat, dir_path):
#     # Modify File Name
#     file = strat['file'].replace(".","_").split("_")[:-1]
#     start_date = os.listdir(dir_path)[0].replace(".","_").split("_")[2]
#     end_date = os.listdir(dir_path)[0].replace(".","_").split("_")[3]
#     file[2]=start_date
#     file[3]=end_date
#     file = "_".join(file)+".csv"
    
#     # Load other parameters
#     price_action = getattr(signals, strat['strategy'])
#     threshold = strat['threshold']
#     direction = strat['cross_dir']
#     side = strat['side']
#     timeframe = strat['timeframe']
#     barrier = ast.literal_eval(strat['barrier'])
#     leverage = strat['leverage']
#     timezone = strat['timezone']
#     dayofweek = strat['dayofweek']
#     tradingsession = strat['tradingsession']
#     macro_tf = strat['macro_tf']
#     macro_threshold = strat['macro_threshold']
#     macro_dir = strat['macro_direction']
    

#     # Load ohlcv
#     df_ohlcv = pd.read_csv(os.path.join(dir_path, file))
#     df_ohlcv.Time = pd.to_datetime(df_ohlcv.Time, utc=True)  
    
#     # Prepare macro bollinger values
#     if macro_tf != "all":
#         candle_ratio = int(timeframe_to_minutes(macro_tf)/timeframe_to_minutes(timeframe)) # Candle ratio means how many micro candles we would need to construct for one macro candle. 
#         macro_blgr_values = signals.get_macro_blgr(df_ohlcv, candle_ratio)

#     # Create Signal
#     signal_priceAction, df_entry_priceAction = price_action(df_ohlcv, threshold, direction)
#     signal_time = signals.time_constraint(df_ohlcv.Time, timezone, dayofweek, tradingsession)
#     signal_priceTime = signal_priceAction & signal_time
#     if macro_tf != "all":
#         if macro_dir == "above":
#             signal_macro = macro_blgr_values > macro_threshold
#         elif macro_dir == "below":
#             signal_macro = macro_blgr_values < macro_threshold
#         elif macro_dir == "around":
#             signal_macro = (macro_threshold-1 < macro_blgr_values) & (macro_blgr_values < macro_threshold+1)
#         signal = signal_priceTime & signal_macro
#     else:
#         signal = signal_priceTime

#     clean_indices = removeOverlaps_1d (signal, barrier[2])

#     # Run Triple Barrier
#     df_entry = df_entry_priceAction.loc[clean_indices]
#     df_result = triple_barrier_dfstyle(df_ohlcv, df_entry, barrier, side) 

#     # Prepare Output
#     rets = df_result.rets - 2*0.0008
#     rets.index = df_result.entry_time
#     rets_adj = rets
#     rets_adj_levered = rets_adj*leverage 
#     lrets_adj_levered = np.log(1+rets_adj_levered)

#     return lrets_adj_levered

# def get_trade_history (strat, dir_path):
#     # Modify File Name
#     file = strat['file'].replace(".","_").split("_")[:-1]
#     start_date = os.listdir(dir_path)[0].replace(".","_").split("_")[2]
#     end_date = os.listdir(dir_path)[0].replace(".","_").split("_")[3]
#     file[2]=start_date
#     file[3]=end_date
#     file = "_".join(file)+".csv"
    
#     # Load other parameters
#     price_action = getattr(signals, strat['strategy'])
#     threshold = strat['threshold']
#     direction = strat['cross_dir']
#     side = strat['side']
#     timeframe = strat['timeframe']
#     barrier = ast.literal_eval(strat['barrier'])
#     leverage = strat['leverage']
#     timezone = strat['timezone']
#     dayofweek = strat['dayofweek']
#     tradingsession = strat['tradingsession']
#     macro_tf = strat['macro_tf']
#     macro_threshold = strat['macro_threshold']
#     macro_dir = strat['macro_direction']
    

#     # Load ohlcv
#     df_ohlcv = pd.read_csv(os.path.join(dir_path, file))
#     df_ohlcv.Time = pd.to_datetime(df_ohlcv.Time, utc=True)  
    
#     # Prepare macro bollinger values
#     if macro_tf != "all":
#         candle_ratio = int(timeframe_to_minutes(macro_tf)/timeframe_to_minutes(timeframe)) # Candle ratio means how many micro candles we would need to construct for one macro candle. 
#         macro_blgr_values = signals.get_macro_blgr(df_ohlcv, candle_ratio)

#     # Create Signal
#     signal_priceAction, df_entry_priceAction = price_action(df_ohlcv, threshold, direction)
#     signal_time = signals.time_constraint(df_ohlcv.Time, timezone, dayofweek, tradingsession)
#     signal_priceTime = signal_priceAction & signal_time
#     if macro_tf != "all":
#         if macro_dir == "above":
#             signal_macro = macro_blgr_values > macro_threshold
#         elif macro_dir == "below":
#             signal_macro = macro_blgr_values < macro_threshold
#         elif macro_dir == "around":
#             signal_macro = (macro_threshold-1 < macro_blgr_values) & (macro_blgr_values < macro_threshold+1)
#         signal = signal_priceTime & signal_macro
#     else:
#         signal = signal_priceTime

#     clean_indices = removeOverlaps_1d (signal, barrier[2])

#     # Run Triple Barrier
#     df_entry = df_entry_priceAction.loc[clean_indices]
#     df_result = triple_barrier_dfstyle(df_ohlcv, df_entry, barrier, side)

#     # Apply Leverage & Commission
#     df_result.rets = df_result.rets - 2*0.0008
#     df_result.lrets = np.log(1+df_result.rets*leverage )

#     return df_result

# def get_rets (strat, dir_path):
#     file = strat['file']
#     price_action = getattr(signals, strat['strategy'])
#     threshold = strat['threshold']
#     direction = strat['cross_dir']
#     side = strat['side']
#     timeframe = strat['timeframe']
#     barrier = ast.literal_eval(strat['barrier'])
#     leverage = strat['leverage']
#     timezone = strat['timezone']
#     dayofweek = strat['dayofweek']
#     tradingsession = strat['tradingsession']
#     macro_tf = strat['macro_tf']
#     macro_threshold = strat['macro_threshold']
#     macro_dir = strat['macro_direction']
    
#     df_ohlcv = pd.read_csv(os.path.join(dir_path, file))
#     df_ohlcv.Time = pd.to_datetime(df_ohlcv.Time, utc=True)  
    
#     if macro_tf != "all":
#         candle_ratio = int(timeframe_to_minutes(macro_tf)/timeframe_to_minutes(timeframe)) # Candle ratio means how many micro candles we would need to construct for one macro candle. 
#         macro_blgr_values = signals.get_macro_blgr(df_ohlcv, candle_ratio)

    
#     signal_priceAction, df_entry_priceAction = price_action(df_ohlcv, threshold, direction)
#     signal_time = signals.time_constraint(df_ohlcv.Time, timezone, dayofweek, tradingsession)
#     signal_priceTime = signal_priceAction & signal_time
#     if macro_tf != "all":
#         if macro_dir == "above":
#             signal_macro = macro_blgr_values > macro_threshold
#         elif macro_dir == "below":
#             signal_macro = macro_blgr_values < macro_threshold
#         elif macro_dir == "around":
#             signal_macro = (macro_threshold-1 < macro_blgr_values) & (macro_blgr_values < macro_threshold+1)
#         signal = signal_priceTime & signal_macro
#     else:
#         signal = signal_priceTime

#     clean_indices = removeOverlaps_1d (signal, barrier[2])
#     pct_overlap = 1-len(clean_indices)/signal.sum()
#     df_entry = df_entry_priceAction.loc[clean_indices]
#     df_result = triple_barrier_dfstyle(df_ohlcv, df_entry, barrier, side) 

#     # Bootstrap the outlier-adjusted returns (output is 2d np.array: iterations x block_size)
#     rets = df_result.rets - 2*0.0008
#     rets.index = df_entry['exit_time']
#     rets.index = df_ohlcv.Time[rets.index]
#     return rets

# def get_lrets_no_barriers (strat, dir_path, leverage =1):
#     file = strat['file']
#     price_action = getattr(signals, strat['strategy'])
#     threshold = strat['threshold']
#     direction = strat['cross_dir']
#     side = strat['side']
#     timeframe = strat['timeframe']
#     barrier = ast.literal_eval(strat['barrier'])

#     timezone = strat['timezone']
#     dayofweek = strat['dayofweek']
#     tradingsession = strat['tradingsession']
#     macro_tf = strat['macro_tf']
#     macro_threshold = strat['macro_threshold']
#     macro_dir = strat['macro_direction']
    
#     df_ohlcv = pd.read_csv(os.path.join(dir_path, file))
#     df_ohlcv.Time = pd.to_datetime(df_ohlcv.Time, utc=True)  
    
#     if macro_tf != "all":
#         candle_ratio = int(timeframe_to_minutes(macro_tf)/timeframe_to_minutes(timeframe)) # Candle ratio means how many micro candles we would need to construct for one macro candle. 
#         macro_blgr_values = signals.get_macro_blgr_old(df_ohlcv, candle_ratio)

    
#     signal_priceAction, df_entry_priceAction = price_action(df_ohlcv, threshold, direction)
#     signal_time = signals.time_constraint(df_ohlcv.Time, timezone, dayofweek, tradingsession)
#     signal_priceTime = signal_priceAction & signal_time
#     if macro_tf != "all":
#         if macro_dir == "above":
#             signal_macro = macro_blgr_values > macro_threshold
#         elif macro_dir == "below":
#             signal_macro = macro_blgr_values < macro_threshold
#         elif macro_dir == "around":
#             signal_macro = (macro_threshold-1 < macro_blgr_values) & (macro_blgr_values < macro_threshold+1)
#         signal = signal_priceTime & signal_macro
#     else:
#         signal = signal_priceTime

#     clean_indices = removeOverlaps_1d (signal, barrier[2])
#     pct_overlap = 1-len(clean_indices)/signal.sum()
#     df_entry = df_entry_priceAction.loc[clean_indices]
#     df_result = triple_barrier_dfstyle(df_ohlcv, df_entry, [0,0,barrier[2]], side) 

#     # Bootstrap the outlier-adjusted returns (output is 2d np.array: iterations x block_size)
#     rets = df_result.rets - 2*0.0008
#     #rets_adj = rets[rets <rets.quantile(0.97)]   
#     rets_adj = rets
#     rets_adj_levered = rets_adj*leverage 
#     lrets_adj_levered = np.log(1+rets_adj_levered)

#     lrets_adj_levered.index = df_ohlcv.Time[lrets_adj_levered.index]

#     return lrets_adj_levered

# def tearsheet (strat, dir_path, pct_diff=7):
#     file = strat['file']
#     price_action = getattr(signals, strat['strategy'])
#     threshold = strat['threshold']
#     direction = strat['cross_dir']
#     side = strat['side']
#     timeframe = strat['timeframe']
#     barrier = ast.literal_eval(strat['barrier'])
#     leverage = strat['leverage']
#     timezone = strat['timezone']
#     dayofweek = strat['dayofweek']
#     tradingsession = strat['tradingsession']
#     macro_tf = strat['macro_tf']
#     macro_threshold = strat['macro_threshold']
#     macro_dir = strat['macro_direction']
    
#     df_ohlcv = pd.read_csv(os.path.join(dir_path, file))
#     df_ohlcv.Time = pd.to_datetime(df_ohlcv.Time, utc=True)  
    
#     if macro_tf != "all":
#         candle_ratio = int(timeframe_to_minutes(macro_tf)/timeframe_to_minutes(timeframe)) # Candle ratio means how many micro candles we would need to construct for one macro candle. 
#         macro_blgr_values = signals.get_macro_blgr(df_ohlcv, candle_ratio)

    
#     signal_priceAction, df_entry_priceAction = price_action(df_ohlcv, threshold, direction)
#     signal_time = signals.time_constraint(df_ohlcv.Time, timezone, dayofweek, tradingsession)
#     signal_priceTime = signal_priceAction & signal_time
#     if macro_tf != "all":
#         if macro_dir == "above":
#             signal_macro = macro_blgr_values > macro_threshold
#         elif macro_dir == "below":
#             signal_macro = macro_blgr_values < macro_threshold
#         elif macro_dir == "around":
#             signal_macro = (macro_threshold-1 < macro_blgr_values) & (macro_blgr_values < macro_threshold+1)
#         signal = signal_priceTime & signal_macro
#     else:
#         signal = signal_priceTime

#     clean_indices = removeOverlaps_1d (signal, barrier[2])
#     df_entry = df_entry_priceAction.loc[clean_indices]
#     df_result = triple_barrier_dfstyle(df_ohlcv, df_entry, barrier, side) 

#     # Bootstrap the outlier-adjusted returns (output is 2d np.array: iterations x block_size)
#     rets = df_result.rets - 2*0.0008
#     rets_adj = rets[rets <rets.quantile(0.97)]   
#     lrets = df_result.lrets - 2*0.0008
#     lrets_adj = lrets[lrets < lrets.quantile(0.97)]         
    
#     # w7
#     bts_lrets_w7 = block_bootstrap(series = lrets_adj, block_size=2, num_blocks=3, num_iterations=100000, pct_diff=pct_diff)
#     bts_lrets_levered_w7 = np.log((np.exp(bts_lrets_w7)-1) * leverage +1)
#     bts_sample_lrets_w7 = np.sum(bts_lrets_levered_w7, axis=1)
#     # w1
#     bts_lrets_w1 = block_bootstrap(series = lrets_adj, block_size=2, num_blocks=3, num_iterations=100000, pct_diff=1)
#     bts_lrets_levered_w1 = np.log((np.exp(bts_lrets_w1)-1) * leverage +1)
#     bts_sample_lrets_w1 = np.sum(bts_lrets_levered_w1, axis=1)

#     # Latest Performance by sample size
#     sample_size = 6
#     lrets_adj_levered = np.log(1+rets_adj*leverage)
#     latest_ret1 = (lrets_adj_levered.iloc[-sample_size:]).sum()
#     latest_ret2 = (lrets_adj_levered.iloc[-sample_size*2:-sample_size]).sum()
#     latest_ret3 = (lrets_adj_levered.iloc[-sample_size*3:-sample_size*2]).sum()
#     latest_avg_ret= np.mean([latest_ret1, latest_ret2, latest_ret3])
#     latest_cumret = np.round((lrets_adj_levered.iloc[-sample_size*3:]).sum(),2)
#     pct1 = np.round(np.mean(bts_sample_lrets_w7 >= latest_ret1),2)
#     pct2 = np.round(np.mean(bts_sample_lrets_w7 >= latest_ret2),2)
#     pct3 = np.round(np.mean(bts_sample_lrets_w7 >= latest_ret3),2)
#     pct_avg = np.round(np.mean(bts_sample_lrets_w7 >= latest_avg_ret),2)
    
#     # Latest Performance by 3 samples grouped
#     bts_lrets_3samples = block_bootstrap(series = lrets_adj, block_size=2, num_blocks=3*3, num_iterations=100000, pct_diff=pct_diff)
#     bts_lrets_levered_3samples = np.log((np.exp(bts_lrets_3samples)-1) * leverage +1) 
#     bts_sample_lrets_3samples = np.sum(bts_lrets_levered_3samples, axis=1)
#     pct_3samples = np.round(np.mean(bts_sample_lrets_3samples >= latest_cumret),2)
    
#     # Overal Performance
#     total_cumret = (lrets_adj).sum()

#     # riskreward
#     riskreward_w7 = bts_sample_lrets_w7[bts_sample_lrets_w7>0].sum()/abs(bts_sample_lrets_w7[bts_sample_lrets_w7<0].sum())
#     riskreward_w1 = bts_sample_lrets_w1[bts_sample_lrets_w1>0].sum()/abs(bts_sample_lrets_w1[bts_sample_lrets_w1<0].sum())    
    
#     # Simple Stat
#     winRate =(rets>0).sum()/len(lrets)
#     avgWin = rets_adj[rets_adj>0].mean()
#     avgLoss = -rets_adj[rets_adj<0].mean()

#     print(strat['key'])
#     print()
#     print("riskreward_w7:", riskreward_w7)
#     print("riskreward_w1:", riskreward_w1)
#     print("Latest Percentile", pct_3samples)    

#     print()
#     print("Latest percentiles", pct1,pct2,pct3, "Avg:",pct_avg) 
#     print("Expected Return", np.mean(bts_sample_lrets_w7))
#     print("Total cumret", total_cumret)
#     print("Latest Avg Returns",latest_avg_ret)
#     print("Last 18 cumret", latest_cumret)

#     print()
#     print("Win rate", winRate)
#     print("Avg Win", avgWin)
#     print("Avg Loss", avgLoss)
#     print("Kelly",winRate - (1-winRate)/(avgWin/avgLoss))
    

    
#     # Plot Scatter      
#     plt.figure(figsize=(10, 5))  # (width, height)
#     plt.scatter(lrets_adj.index,lrets_adj)
#     plt.axhline(y=0, color='red')
#     plt.show()

#     # Plot the bar chart
#     series = pd.Series(bts_sample_lrets_w7)
#     percentiles = [1,5,10,15,20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
#     values = [series.quantile(1-p/100) for p in percentiles]
#     plt.figure(figsize=(10, 5)) 
#     plt.bar(percentiles, values, width=3)
#     plt.yticks(np.arange(-0.2, 0.2, 0.025))
#     plt.xticks(percentiles)  # Set x-tick marks for clarity
#     plt.grid(axis='y', linestyle='--', linewidth=1)
#     plt.show()
    
# def timeframe_to_minutes(timeframe_str):
#     if 'm' in timeframe_str:
#         return int(timeframe_str.replace('m', ''))
#     elif 'h' in timeframe_str:
#         return int(timeframe_str.replace('h', '')) * 60
#     elif 'd' in timeframe_str:
#         return int(timeframe_str.replace('d', '')) * 60 * 24    
#     elif 'w' in timeframe_str:
#         return int(timeframe_str.replace('w', '')) * 60 * 24 * 7      
#     else:
#         raise ValueError("Invalid time format")
    
# def shift_matrix(matrix, n, fill_value = np.nan):
#     shift_matrix = np.roll(matrix,n, axis = 0)
#     if n > 0:
#         shift_matrix[:n] = fill_value
#     elif n < 0:
#         shift_matrix[n:] = fill_value
#     return shift_matrix

# def get_subperiod_indices (max_index,n):
    
#     subperiod_length = (max_index+1) // n
#     result = []
#     for i in range(n):
#         start_idx = i * subperiod_length
#         end_idx = (i + 1) * subperiod_length if i < n - 1 else max_index
#         result.append([start_idx,end_idx])
#     return np.vstack(result)

# def macro_bollinger (df_ohlcv, candle_ratio, params = {'lookback': 30}):

    base_index = np.array(df_ohlcv.index)
    count = 0
    atrs = []
    prices = []

    for i in range(params['lookback']):

        highs = []
        lows = []
        closes = []

        for j in range(candle_ratio):

            if count > len(df_ohlcv):
                raise ValueError("Insufficient Data Period.")
            high = np.array(df_ohlcv.High)[base_index - count]
            high[base_index-count<0] = np.nan
            highs.append(high)

            low = np.array(df_ohlcv.Low)[base_index - count]
            low[base_index-count<0] = np.nan
            lows.append(low)

            close = np.array(df_ohlcv.Close)[base_index - count]
            close[base_index-count<0] = np.nan
            closes.append(close)           
            
            count += 1

        highs = np.array(highs)
        highs = np.max(highs,axis=0)
        lows = np.array(lows)
        lows = np.min(lows,axis=0)
        closes = np.array(closes)
        closes = closes[0]        

        atrs.append(highs - lows)
        prices.append(closes)

    atrs = np.mean(np.array(atrs),axis=0)
    smas = np.mean(np.array(prices), axis=0)
    prices = np.array(prices)[0]
    
    result = (prices - smas) / atrs
    
    return pd.Series(result, index=df_ohlcv.index)

# def key_to_stratConfig(key):
#     match = re.match(
#         r'(?P<file>.*?\.csv)_(?P<price_action>\w+)_(?P<threshold>-?\d+)_'
#         r'(?P<direction>\w+)_(?P<side>long|short)_(?P<barrier>\[.*?\])_'
#         r'(?P<leverage>\d+(\.\d+)?)x_(?P<timezone>[\w/]+(?:_[\w/]+)*)_'
#         r'(?P<dayofweek>\w+(-\w+)*)_(?P<tradingsession>\w+)_'
#         r'(?P<actual_lvg>\d+)_'
#         '(?P<order_size>\d+(\.\d+)?)$',
#         key
#     )
#     if match:
#         file = match.group('file')
#         func_name = match.group('func_name')
#         side = match.group('side')
#         n_atr = int(match.group('n_atr'))
#         barrier = [int(x) for x in match.group('barrier').strip('[]').split(', ')]
#         leverage = int(match.group('leverage'))
        
#         # Check if the function name exists in the signals module
#         if hasattr(signals, func_name):
#             func = getattr(signals, func_name)
#         else:
#             print(f"The function {func_name} does not exist in the signals module.")
#             func = None

#     return file, func, n_atr, side, barrier, leverage

# def key_to_stratString(input_string):
#     # Using regular expressions to match and extract the desired components
#     match = re.match(
#         r'(?P<file>.*?\.csv)_(?P<price_action>\w+)_(?P<threshold>-?\d+)_'
#         r'(?P<direction>\w+)_(?P<side>long|short)_(?P<barrier>\[.*?\])_'
#         r'(?P<leverage>\d+(\.\d+)?)x_(?P<timezone>[\w/]+(?:_[\w/]+)*)_'
#         r'(?P<dayofweek>\w+(-\w+)*)_(?P<tradingsession>\w+)_'
#         r'(?P<actual_lvg>\d+)_'
#         '(?P<order_size>\d+(\.\d+)?)$',
#         input_string
#     )

#     if match:
#         file = match.group('file')
#         price_action = match.group('price_action')
#         threshold = int(match.group('threshold'))
#         direction = match.group('direction')
#         side = match.group('side')
#         barrier = [int(x) for x in match.group('barrier').strip('[]').split(', ')]
#         leverage = float(match.group('leverage'))
#         timezone = match.group('timezone')
#         dayofweek = match.group('dayofweek')
#         tradingsession = match.group('tradingsession')
#         actual_lvg = int(match.group('actual_lvg'))
#         order_size = match.group('order_size')
            
#     symbol = file.split("_")[0]
#     timeframe = file.split("_")[1]

#     # Constructing the new string
#     strat_string = f'''Strategy(portfolio = portfolio, strat_name ='{input_string}', symbol = '{symbol}', timeframe = '{timeframe}',
#     signal_func = {price_action}, n_atr = {threshold}, direction = '{direction}', positionSide = "{side}", barrier = {barrier}, lookback = 30,
#     allocation = 1, order_size = {order_size}, leverage = {actual_lvg}, timezone = '{timezone}', dayofweek = "{dayofweek}", tradingsession = "{tradingsession}", priority = 1, entry_premium = 0,
#     SIM_filepath = "data/tick/{symbol}_tick.csv",)'''

#     return strat_string

# def parallel_process_files (function, directory, params = {}, num_cpus=8):

#     # Create list of files to process
#     files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

#     # Initiate Executor
#     with ProcessPoolExecutor(max_workers= num_cpus) as executor:
        
#         # Map a future to each of the files (future as key, file as dict)
#         future_file_dict = {executor.submit(function, file,  **params): file for file in files}

#         # Loop through each future
#         for future in tqdm(as_completed(future_file_dict), total=len(files), desc="Processing files"):

#             # Retrieve filename for the future
#             file = future_file_dict[future]
#             try:
#                 # Get results
#                 result = future.result()
#                 print(f'{file} done.')
            
#             except Exception as e:
#                 print(f'\n{file} generated an exception: {e.__class__.__name__}')
#                 print(f'\nArguments:', e.args)
#                 print('\nDetailed traceback:')
#                 traceback.print_tb(e.__traceback__)

# def create_return_matrix (ohlcv, hold_periods, commission, pos_sizing = None):
#     '''This creates a return matrix that will serve as a "database" for signals to pull from.
#     Columns represent each signal, and there are three rows:
#     Row 1 = holding period in number of candles
#     Row 2 = gap, the distance from firing of signal to the beginning of holding period.
#     Row 3 = 1 for long, -1 for short.'''

#     '''Updated 240113: Corrected the error in converting long returns into short returns
#                        Also incorporated position sizing using percentage ATR'''
    
#     return_matrix = []
#     # path_low_matrix = []
#     # path_high_matrix = []
#     param_matrix = [] # Has 3 rows: period, gap, and side

#     for p in hold_periods:
#         rets_1d = ohlcv.Close.shift(-p)/ohlcv.Close-1
#         # path_low_1d = ohlcv.Low.rolling(p).min().shift(-p)/ohlcv.Close - 1
#         # path_high_1d = ohlcv.High.rolling(p).max().shift(-p)/ohlcv.Close - 1
#         return_matrix.append(rets_1d.values)
#         # path_low_matrix.append(path_low_1d)
#         # path_high_matrix.append(path_high_1d)
#         param_matrix.append([int(p), 0, 1])

#     return_matrix = np.column_stack(return_matrix)
#     # path_low_matrix = np.column_stack(path_low_matrix)
#     # path_high_matrix = np.column_stack(path_high_matrix)
#     param_matrix = np.column_stack(param_matrix)

#     nan_mask = np.isnan(return_matrix)
#     # nan_mask_path = np.isnan(path_low_matrix)

#     for side in ['long','short']: # *****
#         if side == 'long':
#             return_matrix_long = return_matrix.copy()
#             return_matrix_long[~nan_mask] -= 2*commission
#             return_matrix_long[nan_mask] = 0

#             # lowest_return_matrix_long = path_low_matrix
#             # lowest_return_matrix_long[~nan_mask_path] -= 2*commission
#             # lowest_return_matrix_long[nan_mask_path] = 0


#         else:            
#             return_matrix_short = -return_matrix
#             return_matrix_short[~nan_mask] -= 2*commission
#             return_matrix_short[nan_mask] = 0
#             # lowest_return_matrix_short = -path_high_matrix
#             # lowest_return_matrix_short[~nan_mask_path] -= 2*commission
#             # lowest_return_matrix_short[nan_mask_path] = 0            


#     final_return_matrix = np.hstack((return_matrix_long, return_matrix_short))
#     # final_lowest_return_matrix = np.hstack((lowest_return_matrix_long, lowest_return_matrix_short))

#     # Normalize
#     if pos_sizing == 'atr':
#         normalizer = (calc_atr(ohlcv, 30)/ohlcv.Close).values * 100
#         normalizer = normalizer[:, np.newaxis]
#         final_return_matrix = final_return_matrix/normalizer

#     final_return_matrix = np.log(1+final_return_matrix)
#     # final_lowest_return_matrix = np.log(1+final_lowest_return_matrix)

#     param_matrix_to_add = param_matrix.copy()
#     param_matrix_to_add[2,:] = -1 # Adjusting for short
#     final_param_matrix = np.hstack((param_matrix, param_matrix_to_add))    

#     return final_return_matrix, final_param_matrix#, final_lowest_return_matrix

# def create_signal_matrix(series, thresholds, method, min_data, include_all = False):
#     '''Creates a m*n matrix of boolean values where each column represents distinct signal.
#     each of m columns maps to each of the thresholds, and n is the length of the series.
#     Available methods: crossing_2way, crossing_1way, macro_mode_zscore, between '''

#     signal_matrix = []
#     signal_header = []

#     if method == "crossing_2way":
#         # Process each threshold for upward crossing signals.
#         for threshold in thresholds:
#             # An upward signal is true when the series crosses above the threshold.
#             signal = (series >= threshold) & (series.shift(1) < threshold)
#             signal_matrix.append(list(signal))  # Add the signal to the matrix.
#             signal_header.append((threshold, 'up'))  # Record the threshold and signal direction.

#         # Process each threshold for downward crossing signals.
#         for threshold in thresholds:
#             # A downward signal is true when the series crosses below the threshold.
#             signal = (series <= threshold) & (series.shift(1) > threshold)
#             signal_matrix.append(list(signal))  # Add the signal to the matrix.
#             signal_header.append((threshold, 'down'))  # Record the threshold and signal direction.

#     elif method == "crossing_1way":
        
#         for threshold in thresholds:
#             if threshold > 0:
#                 # Process each threshold for upward crossing signals.
#                 signal = (series >= threshold) & (series.shift(1) < threshold)
#                 signal_matrix.append(list(signal))  # Add the signal to the matrix.
#                 signal_header.append((threshold, 'up'))  # Record the threshold and signal direction.
        
#             else: # threshold < 0
#                 # Process each threshold for downward crossing signals.
#                 signal = (series <= threshold) & (series.shift(1) > threshold)
#                 signal_matrix.append(list(signal))  # Add the signal to the matrix.
#                 signal_header.append((threshold, 'down'))  # Record the threshold and signal direction.            

#     elif method == "between":

#         thresholds.sort()

#         # Handle the first column (values less than the first threshold)
#         signal = (series <= thresholds[0])
#         signal_matrix.append(list(signal))
#         signal_header.append(((np.nan,thresholds[0]), 'between'))

#         # Handle the middle columns (values between pairs of thresholds)
#         for i in range(len(thresholds) - 1):
#             signal = (series >= thresholds[i]) & (series < thresholds[i + 1])
#             signal_matrix.append(list(signal))
#             signal_header.append(((thresholds[i],thresholds[i+1]), 'between'))

#         # Handle the last column (values greater than or equal to the last threshold)
#         signal = (series >= thresholds[-1])
#         signal_matrix.append(list(signal))
#         signal_header.append(((thresholds[-1],np.nan), 'between'))

#     elif method == "macro_mode_zscore":

#         for threshold in thresholds:
#             # Above
#             signal = series > threshold
#             signal_matrix.append(list(signal))
#             signal_header.append((threshold, 'above'))

#             # Below
#             signal = series < threshold
#             signal_matrix.append(list(signal))
#             signal_header.append((threshold, 'below'))            

#             # Around
#             signal = (series > threshold - 1) & (series < threshold + 1)
#             signal_matrix.append(list(signal))
#             signal_header.append((threshold, 'around'))                    

#     else:
#         raise ValueError("Method unspecified or incorrect")
    
#     # Convert to np matrix
#     signal_matrix = np.array(signal_matrix).T
#     signal_header = np.array(signal_header, dtype=object).T

#     # Minimum Data and All True Filter
#     min_data_filter = np.sum(signal_matrix, axis = 0) > min_data
#     not_all_true_filter = ~np.all(signal_matrix, axis = 0)

#     signal_matrix = signal_matrix[:,min_data_filter&not_all_true_filter]
#     signal_header = signal_header[:,min_data_filter&not_all_true_filter]

#     # Filter out Duplicates using set hash
#     n = signal_matrix.shape[1]
#     duplicate_filter = np.zeros(n, dtype=bool)
#     seen_hashes = set()

#     for col_idx in range(n):
#         # Compute a hash for the current column
#         # You can use a tuple conversion as a simple hash for boolean values
#         col_hash = tuple(signal_matrix[:, col_idx])

#         if col_hash in seen_hashes:
#             # Mark as duplicate if hash was seen before
#             duplicate_filter[col_idx] = True
#         else:
#             # Add hash to seen_hashes
#             seen_hashes.add(col_hash)

#     # Update signal_matrix and signal_header
#     signal_header = signal_header[:, ~duplicate_filter]
#     signal_matrix = signal_matrix[:, ~duplicate_filter]

#     # Add a blank signal
#     if include_all:
#         all_true = np.array([True] * len(series)).reshape(-1,1)
#         all_true_header = np.array([None,'all']).reshape(-1,1)
#         signal_matrix = np.hstack([all_true,signal_matrix])
#         signal_header = np.hstack([all_true_header, signal_header])        

#     return signal_matrix, signal_header

# def remove_overlapping_signals (entry_index_matrix, distance, fill_value):

#     while True:
#         ''' Purpose: How do we know if there are overlaps?
#         We know signals overlap if its entry index is greater than the sum of previous index + holding period.
#         In which case, we'll copy down that index. We do this because we now can use this copied down index number
#         to see if there is a consecutive overlap. If so, the same logic will detect it. '''
#         shifted_matrix = shift_matrix(entry_index_matrix,+1, fill_value=-1000) #Using -100 to make sure it doesn't overlap with anything, and keep it integer
#         overlap_mask = (entry_index_matrix < shifted_matrix+distance) & (entry_index_matrix != shifted_matrix)
#         overlap_mask = overlap_mask & ~shift_matrix(overlap_mask,1, fill_value = False)
#         # If no more overlap, break out of loop
#         if overlap_mask.sum () == 0: 
#             break
#         # If there's still overlap, copy down the index for the overlapping position. 
#         entry_index_matrix[overlap_mask] = shifted_matrix[overlap_mask]

#     # Once done, get the actual mask by checking if current index is the same as the index above. 
#     final_mask = (entry_index_matrix == shift_matrix(entry_index_matrix,+1, fill_value=-1000))

#     # Using -1 for the entry index. 
#     entry_index_matrix[final_mask] = fill_value

#     return entry_index_matrix.astype(int), final_mask

# def create_thresholds (series, method, params):
#     '''This function creates a list of threshold values given a series of interest
#     Creates a total number of 2n threshold values, n for positive and negative side respectively'''
#     thresholds = []
#     if method == "as_is":
#         for i in params:
#             thresholds.append(i)

#     elif method == "zscore":
#         mean = series.mean()
#         pos_std = abs(series[series>mean].mean())
#         neg_std = abs(series[series<mean].mean())
#         for i in params:
#             if i > 0:
#                 thresholds.append(round(mean+i*pos_std,2))
#             else: # i < 0
#                 thresholds.append(round(mean+i*neg_std,2))

#     elif method == "percentile":
#         percentiles = [p for p in params]
#         thresholds = np.round(np.nanpercentile(series, percentiles).tolist(),4)

#     else:
#         raise ValueError("Method unspecified or incorred")
#     return thresholds

# def create_list_of_barriers (verticals, tops = [0], bottoms = [0]):
#     return [[t,b,v] 
#             for v in verticals
#             for t in tops
#             for b in bottoms]

# def create_time_signal_matrix(time_index_series, min_data):

#     signal_matrix = []
#     signal_header = []

#     exclusions = {("America/New_York", 'all', 'all'), ("Asia/Hong_Kong", 'all', 'all')}

#     time_constraints = [
#         (timezone, dayofweek, tradingsession)
#         for timezone in ["UTC", "America/New_York", "Asia/Hong_Kong"]
#         for dayofweek in ["all", "weekday", "weekend", "monday", "friday", "tue-thu"]
#         for tradingsession in ["all", "trading", "nontrading"]
#         if (timezone, dayofweek, tradingsession) not in exclusions
#     ]

    
#     for timezone, dayofweek, tradingsession in time_constraints:
#         signal = time_constraint(time_index_series, timezone, dayofweek, tradingsession)
#         if signal.sum()>= min_data:
#             signal_matrix.append(list(signal))
#             signal_header.append((timezone,dayofweek,tradingsession))

#     return np.array(signal_matrix).T, np.array(signal_header, dtype=object).T

# def time_constraint(timestamp_series, timezone, dayofweek, tradingsession):
    
#     # Ensure the timestamp series is localized to UTC before conversion
#     if timestamp_series.dt.tz is None:
#         localized_series = timestamp_series.dt.tz_localize('UTC')
#     else:
#         localized_series = timestamp_series.copy()

#     # Convert the timestamp series to the specified timezone
#     localized_series = localized_series.dt.tz_convert(timezone)        

#     # Directly use pandas functionality for checking days of the week
#     if dayofweek.lower() == 'all':
#         dayofweek_filter = pd.Series(True, index=localized_series.index)
#     elif dayofweek.lower() == 'weekday':
#         dayofweek_filter = localized_series.dt.weekday < 5
#     elif dayofweek.lower() == 'weekend':
#         dayofweek_filter = localized_series.dt.weekday >= 5
#     elif dayofweek.lower() in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
#         dayofweek_filter = localized_series.dt.day_name().str.lower() == dayofweek.lower()
#     elif dayofweek.lower() == 'tue-thu':
#         dayofweek_filter = localized_series.dt.weekday.between(1, 3)
#     else:
#         raise ValueError("Invalid dayofweek value")

#     # Define the trading hours (8am to 5pm)
#     trading_start, trading_end = 8, 17
    
#     # Directly use pandas functionality for checking trading sessions
#     if tradingsession.lower() == 'all':
#         tradingsession_filter = pd.Series(True, index=localized_series.index)
#     elif tradingsession.lower() == 'trading':
#         tradingsession_filter = (localized_series.dt.hour >= trading_start) & (localized_series.dt.hour < trading_end)
#     elif tradingsession.lower() == 'nontrading':
#         tradingsession_filter = (localized_series.dt.hour < trading_start) | (localized_series.dt.hour >= trading_end)
#     else:
#         tradingsession_filter = pd.Series(False, index=localized_series.index)

#     # Combine the boolean series for final output
#     result = dayofweek_filter & tradingsession_filter
#     return result

# def resample_ohlcv (df, time_multiple):
#     # Determine the most common frequency in the dataframe
#     original_freq = df.index.to_series().diff().mode()[0]
    
#     new_freq = original_freq * time_multiple
#     new_freq_str = f'{new_freq.seconds // 60}T'

#     resampled_df = df.resample(new_freq_str).agg({'Open': 'first',
#                                            'High': 'max',
#                                            'Low': 'min',
#                                            'Close': 'last',
#                                            'Volume': 'sum'})
#     return resampled_df

# def remove_overlapping_signals_deprecated (entry_index_matrix_, distance):

#     entry_index_matrix = entry_index_matrix_.copy()

#     # Then replace overlapping index with np.nans
#     while True:
#         overlap_mask = (shift_matrix(entry_index_matrix,+1, fill_value=np.nan)+distance) > entry_index_matrix
#         overlap_mask = overlap_mask & ~shift_matrix(overlap_mask,1, fill_value = False)
#         if overlap_mask.sum () == 0: 
#             break
#         entry_index_matrix[overlap_mask] = np.nan
#         entry_index_matrix = np.sort(entry_index_matrix, axis=0)
#         test =1 

#     return entry_index_matrix

# def removeOverlaps_1d(signal, p):
#     # Extract the indices where the signal is True
#     start_index = np.array(signal.index[signal])

#     # Calculate the end index for each signal, assuming each signal lasts for 'p' periods
#     end_index = start_index + p

#     # Create a shifted version of the end_index array for overlap comparison
#     # The first element is set to 0 as a placeholder since there's no preceding element for the first signal
#     end_index_shift = np.insert(end_index[:-1], 0, 0)

#     # Determine if there is an overlap. An overlap occurs if the start of the current signal is before
#     # the end of the previous signal, but after the start of the previous signal
#     is_overlap = (end_index_shift <= end_index) & (end_index_shift > start_index)

#     # Update the overlap array to ensure that each signal is only compared with its immediate predecessor
#     is_overlap = is_overlap & ~np.insert(is_overlap[:-1], 0, 0)

#     # Continue removing overlaps until there are no more overlaps
#     while np.sum(is_overlap) > 0:
#         # Keep only the start indices of signals that don't overlap
#         start_index = start_index[~is_overlap]

#         # Recalculate the end indices for the remaining signals
#         end_index = start_index + p

#         # Recreate the shifted end_index array for the next round of overlap checking
#         end_index_shift = np.insert(end_index[:-1], 0, 0)

#         # Check for overlaps again with the updated indices
#         is_overlap = (end_index_shift <= end_index) & (end_index_shift > start_index)

#     # Return the final list of start indices with all overlaps removed
#     return start_index

# def create_return_matrix_old (ohlcv, ret_periods, gaps, commission, fill_value = np.nan):
#     '''This creates a return matrix that will serve as a "database" for signals to pull from.
#     Columns represent each signal, and there are three rows:
#     Row 1 = holding period in number of candles
#     Row 2 = gap, the distance from firing of signal to the beginning of holding period.
#     Row 3 = 1 for long, -1 for short.'''

#     close_1d = ohlcv.Close.values
#     lrets_1d = np.log(close_1d/np.insert(close_1d[:-1],0,ohlcv.Open[0]))

#     return_matrix = []
#     param_matrix = [] # Has 3 rows: period, gap, and side

#     for p in ret_periods:
#         rollsum_1d = rolling_sum_next_n(lrets_1d,p, fill_value=np.nan)
#         return_matrix.append(rollsum_1d)
#         param_matrix.append([int(p), 0, 1])

#     return_matrix = np.column_stack(return_matrix)
#     param_matrix = np.column_stack(param_matrix)

#     for g in gaps:
#         if g > 0:
#             return_matrix_to_add = np.vstack((return_matrix[g:, :], np.full((g, return_matrix.shape[1]), np.nan)))
#             return_matrix = np.hstack((return_matrix, return_matrix_to_add))
            
#             param_matrix_to_add = param_matrix.copy()
#             param_matrix_to_add[1,:] = int(g)            
#             param_matrix = np.hstack((param_matrix, param_matrix_to_add))

#     nan_mask = np.isnan(return_matrix)

#     for side in ['long','short']: # *****
#         if side == 'long':
#             return_matrix_long = return_matrix.copy()
#             return_matrix_long[~nan_mask] -= 2*commission
#             return_matrix_long[nan_mask] = 0
#         else:
#             nonlog_returns = np.exp(return_matrix)-1
#             nonlog_returns = - nonlog_returns
#             return_matrix_short = np.log(nonlog_returns)
            
#             return_matrix_short = -return_matrix
#             return_matrix_short[~nan_mask] -= 2*commission
#             return_matrix_short[nan_mask] = 0

#     final_return_matrix = np.hstack((return_matrix_long, return_matrix_short))

#     param_matrix_to_add = param_matrix.copy()
#     param_matrix_to_add[2,:] = -1
#     final_param_matrix = np.hstack((param_matrix, param_matrix_to_add))    

#     return final_return_matrix, final_param_matrix

# def macro_transform_dep (transform, df_ohlcv, time_multiple, params):
#     '''Transforms ohlcv into bollinger series, using a timeframe scaled by time multiple'''

#     df = df_ohlcv.copy()
#     df.set_index('Time', inplace=True)
    
#     original_freq_minutes = df.index.to_series().diff().dt.total_seconds().div(60).mode()[0]
#     result = []
    
#     # Resample first candle
#     resampled_df = resample_ohlcv(df, time_multiple)
#     resampled_df.index = (resampled_df.index)
#     blgr_series = transform(resampled_df, params) 
#     result.append(blgr_series)
    
#     # Resample the following candles by shiftinig time by the original frequency
#     for i in range(1,time_multiple):
        
#         # Subtract original frequency
#         df.index = (df.index - pd.Timedelta(minutes=original_freq_minutes))
        
#         # Remove the first candle that gets left out
#         df = df[1:]
        
#         # Resample into bigger candle
#         resampled_df = resample_ohlcv(df, time_multiple)
        
#         # Re-adjust the time correctly
#         resampled_df.index = resampled_df.index + pd.Timedelta(minutes=original_freq_minutes*i)
        
#         # Get bollinger values
#         blgr_series = transform(resampled_df, params)
        
#         # Add to the result list
#         result.append(blgr_series)

#     # Combine to a single series, sort by time, and align past-future
#     result =  pd.concat(result).sort_index().shift(time_multiple-1)

#     result.index = df_ohlcv.index
    
#     return result

# def comparison_bar_charts (list1, list2):

#     # Setting the positions of the bars on the x-axis
#     x = np.arange(len(list1))
#     width = 0.35  # width of the bars

#     # Creating the bar chart
#     plt.figure(figsize=(5, 3))
#     plt.bar(x - width/2, list1, width, label='List 1', color='blue')
#     plt.bar(x + width/2, list2, width, label='List 2', color='orange')
#     plt.show()

# def create_exit_matrix_dep (ohlcv, atr, list_of_barriers, list_of_gaps=[0]):
#     exit_matrix, exit_header = [], []
#     full_signal = np.array([True]*len(ohlcv))
#     for side in ['long','short']:
#         for barrier in list_of_barriers:
#             lrets = triple_barrier_(ohlcv, full_signal, atr, barrier, side = side)
#             for gap in list_of_gaps:
#                 lrets_shifted = shift_matrix(lrets,-gap) 
#                 exit_matrix.append(lrets_shifted)
#                 exit_header.append((side, barrier, gap))
#     exit_matrix, exit_header = np.array(exit_matrix).T, np.array(exit_header, dtype=object)
#     exit_matrix[np.isnan(exit_matrix)] = 0
#     return exit_matrix, exit_header

# def lnrg_slope (df, params):
#     # Slope of linear regression
#     lookback = params['lookback']
#     normalizer = get_normalizer(df, params['normalizer'], lookback)
#     slope = talib.LINEARREG_SLOPE(df.Close.values, 30)
#     slope = slope/normalizer
#     transf = pd.Series(data=slope, index=df.index)
#     return transf

# def calc_rsi (df, params):
#     lookback = params['lookback']
#     rsi = talib.RSI(df.Close, lookback)
#     transf = pd.Series(data=rsi, index=df.index)
#     if params['center_to_zero']:
#         transf -= 50
#     return transf

# def modified_adx (df, params):
#     # Modifies to get rid of absolute value
#     plus = talib.PLUS_DI(df.High, df.Low, df.Close, params['lookback'])
#     minus = talib.MINUS_DI(df.High, df.Low, df.Close, params['lookback'])
#     adx = ((plus - minus)/(plus+minus)*100).rolling(params['lookback']).mean()
#     return adx

# def aroonosc(df, params):
#     # Useless
#     transf = talib.AROONOSC(df.High, df.Low, params['lookback'])
#     transf = pd.Series(data=transf, index=df.index)
#     return transf

# def bop (df, params):
#     # Seems useless. 
#     transf = talib.BOP(df.Open, df.High, df.Low, df.Close)
#     transf = pd.Series(data=transf, index=df.index)
#     return transf

# def cci (df, params):
#     # KP comment: Sketchy version of mean rveresion
#     transf = talib.CCI(df.High, df.Low, df.Close, params['lookback'])
#     transf = pd.Series(data=transf, index=df.index)
#     return transf

# def cmo (df, params):
#     # RSI 랑 거의 동일함. 
#     transf = talib.APO(df.Close, params['lookback']) * 1e5
#     transf = pd.Series(data=transf, index=df.index)
#     return transf

# def mfi (df, params):
#     # It's like RSI, but applied to Signed Volume
#     transf = talib.MFI(df.High, df.Low, df.Close, df.Volume, params['lookback'])
#     transf = pd.Series(data=transf, index=df.index)
#     if params['center_to_zero']:
#         transf -= 50    
#     return transf

# def ultosc (df, params):
#     # Linear combination of various momentum periods normalized by TR
#     transf = talib.ULTOSC(df.High, df.Low, df.Close, params['lookback1'], params['lookback2'], params['lookback3'])
#     transf = pd.Series(data=transf, index=df.index)
#     if params['center_to_zero']:
#         transf -= 50      
#     return transf

# def macd (df, params):
#     macd, macdsignal, macdhist = talib.MACD(df.Close, fastperiod=params['fast'], slowperiod=params['slow'], signalperiod=params['signal'])
#     transf = pd.Series(data= macdhist, index = df.index)
#     normalizer = get_normalizer(df, params['normalizer'], lookback = params['slow'])
#     transf = transf/normalizer
#     return transf

# def ppo (df, params):
#     # basically macd,but normalized into percentage. 
#     transf = talib.PPO(df.Close, fastperiod=params['fast'], slowperiod=params['slow'], matype=0)
#     transf = pd.Series(data= transf, index = df.index)
#     normalizer = get_normalizer(df, params['normalizer'], lookback = params['slow'])
#     transf = transf/normalizer
#     return transf

# def stochrsi (df, params):
#     # RSI 자체를 Stochastic (referenced to rolling High and Low)으로 normalize한것. 
#     fastk, fastd = talib.STOCHRSI(df.Close, timeperiod=params['lookback'], fastk_period=params['fastk'], fastd_period=1, fastd_matype=0)
#     transf = fastk
#     if params['center_to_zero']:
#         transf -= 50
#     return transf