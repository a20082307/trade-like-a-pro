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
data = None
record = None

kdj = None
bband = None

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
## ============================================================ ##







class TTBProcess(TTBModule):
    def SHOWQUOTEDATA(self, obj):
        print(type(obj))
        print(obj)
        # print("Symbol:{}, BidPs:{}, BidPv:{}, AskPs:{}, AskPv:{}, T:{}, P:{}, V:{}, Volume:{}".format(obj['Symbol'],obj['BidPs'],obj['BidPv'],obj['AskPs'],obj['AskPv'],obj['TickTime'],obj['Price'],obj['Qty'],obj['Volume']))  

if __name__ == "__main__":
    # Read data
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['datetime', 'open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j'])
            data = pd.DataFrame(columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'k', 'd', 'j'])
    else:
        data = pd.read_csv(DATA_PATH)

    # Start connection to TTB
    ttbModule = TTBProcess('http://localhost:8080', 51141)
    ttbModule.QUOTEDATA('TXFF4')

    # Read trading record
    if not os.path.exists(RECORD_FILE_PATH):
        with open(RECORD_FILE_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'Profit', 'Volume'])
    else:
        record = pd.read_csv(RECORD_FILE_PATH)
