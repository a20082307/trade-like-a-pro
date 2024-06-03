## ==================== Import Modules ==================== ##
import csv
import datetime
import os

import numpy as np
import pandas as pd

import talib as ta
from TTBHelp import *
## ============================================================ ##

## ==================== Constants ==================== ##
RECORD_FILE_PATH = './record.csv'
DATA_PATH = './data.csv'
MORNING_START = datetime.time(9, 30)
MORNING_END = datetime.time(13, 40)
NIGHT_START = datetime.time(15, 45)
NIGHT_END = datetime.time(4, 55)
## ============================================================ ##

## ==================== Global Variables ==================== ##
ttbModule = None
data = None
record = None

first_round = True
current_time = None
next_update_time = None
tem_data = None

kdj = None
bband = None
current_position_type = None

is_squeeze = False
open_long_ready = False
open_short_ready = False
sell_ready = False
sl_price = None

## ==================== Functions ==================== ##
def is_morning(time: datetime.datetime):
    return MORNING_START <= time.time() <= MORNING_END

def is_night(time: datetime.datetime):
    return time.time() <= NIGHT_END or time.time() >= NIGHT_START

def is_trading_time(time: datetime.datetime):
    return is_morning(time) or is_night(time)

def KDJ(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, signal_k: int, signal_d: int, prev_info: dict):
    return_data = {
        'k': None,
        'd': None,
        'j': None,
    }
    _alpha_k = 2 / (signal_k + 1)
    _alpha_d = 2 / (signal_d + 1)

    high = high[-period - 1 : ]
    low = low[-period - 1 : ]
    close = close[-period - 1 : ]

    rsv = int((close[-1] - min(low)) / (max(high) - min(low)) * 100 + 0.5 if max(high) - min(low) != 0 else 0)

    last_k = prev_info['k'] if prev_info is not None else 50
    cur_k = int(_alpha_k * ((last_k + 2 * rsv) / 3) + (1 - _alpha_k) * last_k + 0.5)

    last_d = prev_info['d'] if prev_info is not None else 50
    cur_d = int(_alpha_d * ((last_d + 2 * cur_k) / 3) + (1 - _alpha_d) * last_d + 0.5)

    cur_j = 3 * cur_k - 2 * cur_d

    return_data['k'] = cur_k
    return_data['d'] = cur_d
    return_data['j'] = cur_j
    # return_data['rsv'] = rsv
    # return_data['high'] = max(high)
    # return_data['low'] = min(low)
    # return_data['close'] = close[-1]

    return return_data

def bbands_keltner(high: np.ndarray, low: np.ndarray, close: np.ndarray, bb_period: int, bb_mult: int, kelt_period: int, kelt_mult: int):
    # Calculate the Bollinger Bands
    bb_mid = ta.SMA(close[-30 : ], timeperiod = bb_period)
    bb_std = ta.STDDEV(close[-30 : ], timeperiod = bb_period)
    bb_upper = bb_mid + bb_std * bb_mult
    bb_lower = bb_mid - bb_std * bb_mult

    # Calculate the Keltner Channel
    kelt_mid = ta.EMA(close[-30 : ], timeperiod = kelt_period)
    kelt_trange = np.array([])
    for i in range(30):
        tem_trange = max(
            high[-i] - low[-i],
            abs(high[-i] - close[-i - 1]),
            abs(low[-i] - close[-i - 1])
        )   
        kelt_trange = np.append(tem_trange, kelt_trange)
    kelt_atr = ta.EMA(kelt_trange, timeperiod = kelt_period)
    kelt_upper = kelt_mid + kelt_atr * kelt_mult
    kelt_lower = kelt_mid - kelt_atr * kelt_mult

    return [
        bb_upper[-1] <= kelt_upper[-1] and bb_lower[-1] >= kelt_lower[-1],
        bb_upper,
        bb_lower,
    ]
    return ta.RSI(close, timeperiod = period)

def update_data(obj):
    global first_round, current_time, next_update_time
    if first_round:
        current_time = datetime.datetime.strptime(obj['TickTime'], '%H:%M:%S')
        next_update_time = current_time + datetime.timedelta(minutes = 1)
        first_round = False

    tem_data = pd.DataFrame(columns = ['datetime', 'open', 'high', 'low', 'close', 'volume'])
## ============================================================ ##

## ==================== Start the strategy with TTB Process ==================== ##
class TTBProcess(TTBModule):
    def SHOWQUOTEDATA(self, obj):
        global ttbModule, data, record
        global kdj, bband, current_position_type
        global is_squeeze, open_long_ready, open_short_ready, sell_ready, sl_price

        # Check if the current time is in trading time
        if not is_trading_time(datetime.datetime.now()):
            return

        # Set the current_position_type to 'None' if there is no position
        
        if len(ttbModule.QUERYRESTOREFILLREPORT()['Data']) == 0:
            current_position_type = 'None'

        # Close the position if current time reaches the end of the time we set up
        if current_position_type != 'None' and current_time.time() > MORNING_END:
            if current_position_type == 'long':
                pass
            elif current_position_type == 'short':
                pass

        # Update the data
        update_data(obj)

        # Check if BBand is squeezing
        if bband[0]:
            is_squeeze = True
            open_long_ready = False
        else:
            is_squeeze = False

        latest_close = data.Close.iloc[-1]
        latest_open = data.Open.iloc[-1]
        latest_bb_up = bband[1][-1]
        latest_bb_down = bband[2][-1]
        latest_j = kdj['j']

        # Check if the precondition to open long/short position is satisfied
        if current_position_type == 'None' and is_squeeze:
            if latest_close < latest_bb_down and latest_j < 20:
                open_long_ready = True
                print("開多單條件符合")
            elif latest_close > latest_bb_up and latest_j > 80:
                open_short_ready = True
                print("開空單條件符合")

        # Check if it's time to open long/short position
        if current_position_type == 'None' and open_short_ready:
            if latest_j < 80 and latest_close < latest_open:
                trader.sell(size = 1)
                sl_price = max(data.High.iloc[-i] for i in range(1, 6))
                sell_ready = False
                print("開空單")
                return
            elif latest_j < 80 and latest_close < latest_open:
                trader.sell(size = 1)
                sl_price = max(data.High.iloc[-i] for i in range(1, 6))
                sell_ready = False
                print("開空單")

        # Check if the precondition to close long/short position is satisfied
        if current_position_type == 'long':
            if latest_j > 80:
                sell_ready = True
                print("準備平倉")
                return
        elif current_position_type == 'short':
            if latest_j < 20:
                sell_ready = True
                print("準備平倉")
                return
            
        # Check if it's time to close long/short position
        if current_position_type == 'long':
            if sell_ready and latest_j <= 80:
                trader.sell(size = 1)
                sell_ready = False
                print("平倉")
                return

            elif latest_close < sl_price:
                trader.sell(size = 1)
                sell_ready = False
                print("停損平倉")
                return
        elif current_position_type == 'short':
            if sell_ready and latest_j >= 20:
                trader.buy(size = 1)
                sell_ready = False
                print("平倉")
                return

            elif latest_close > sl_price:
                trader.buy(size = 1)
                sell_ready = False
                print("停損平倉")
                return
## ============================================================ ##

## ==================== Main ==================== ##
if __name__ == "__main__":
    # Read data
    if not os.path.exists(DATA_PATH):
        print('Data file not found. Creating new data file.')
        with open(DATA_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['datetime', 'open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j'])
            data = pd.DataFrame(columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j'])
    else:
        print('Data file found. Reading data file.')
        data = pd.read_csv(DATA_PATH)
        print('Data file read successfully.')

    # get the previous kdj
    if len(data.j) > 0:
        print('Reading previous KDJ data.')
        kdj = {'k': data.k.iloc[-1], 'd': data.d.iloc[-1], 'j': data.j.iloc[-1]}
        print('KDJ data read successfully.')

    # Start connection to TTB
    ttbModule = TTBProcess('http://localhost:8080', 51141)
    ttbModule.QUOTEDATA('TXFF4')

    # Read trading record
    if not os.path.exists(RECORD_FILE_PATH):
        print('Record file not found. Creating new record file.')
        with open(RECORD_FILE_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'Profit', 'Volume'])
    else:
        print('Record file found. Reading record file.')
        record = pd.read_csv(RECORD_FILE_PATH)
        print('Record file read successfully.')
## ============================================================ ##