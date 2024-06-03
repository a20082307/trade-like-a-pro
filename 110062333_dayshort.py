## ==================== Import modules ==================== ##
import datetime

import numpy as np
import pandas as pd
import time
import threading

import pytrader as pyt
import talib as ta
from app import getData


## ==================== Constants ==================== ##
api_key = 'GgyvFPLdwiSGBXC2kuerM6F29SaCVAecccnPxvAiW7z1'
secret_key = 'Gc2JfqQuV6tXyFv42GGPvU34wm29WUxfPtTGsZ12GAPn'
id_strategy = '110062333_dayshort'

MORNING_START = datetime.time(9, 30)
MORNING_END = datetime.time(13, 40)
NIGHT_START = datetime.time(15, 45)
NIGHT_END = datetime.time(4, 55)

DAY_STOCK_BEGIN = datetime.time(8, 45)
DAY_STOCK_END = datetime.time(13, 45)
NIGHT_STOCK1_BEGIN = datetime.time(15, 0)
NIGHT_STOCK1_END = datetime.time(23, 59)
NIGHT_STOCK2_BEGIN = datetime.time(0, 0)
NIGHT_STOCK2_END = datetime.time(5, 0)

## ==================== Global variables ==================== ##
non_real_time_data = None
real_time_data1 = None
real_time_data2 = None
new_data1 = None
new_data2 = None
kdj = None
bband = None
rsi = None

is_squeeze = False
open_long_ready = False
open_short_ready = False
sell_ready = False
sl_price = None

current_time = None
next_update_time = None
trader = None
api = None
logged_in = False

## ==================== Functions ==================== ##
def change_data_period(kbar_df: pd.DataFrame, interval: int):   
    new_kbars_df = kbar_df.copy(deep = True)
    new_kbars_df.index = pd.to_datetime(new_kbars_df.datetime)
    
    morning = new_kbars_df.between_time('08:45', '13:45')
    morning = morning.resample(f'{interval}Min', closed = 'right', label = 'right', origin = datetime.datetime(2020, 3, 22, 8, 45)).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        '大戶買進': 'sum',
        '散戶買進': 'sum',
        '大戶掛單': 'sum',
        '散戶掛單': 'sum'
    })
    morning = morning.dropna()

    night = new_kbars_df.between_time('15:00', '5:00')
    night = night.resample(f'{interval}Min', closed = 'right', label = 'right', origin = datetime.datetime(2020, 3, 22, 15, 0)).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        '大戶買進': 'sum',
        '散戶買進': 'sum',
        '大戶掛單': 'sum',
        '散戶掛單': 'sum'
    })
    night = night.dropna()

    return pd.concat([morning, night]).sort_values('datetime')

def is_morning(time):
    return MORNING_START <= time.time() <= MORNING_END

def is_night(time):
    return time.time() <= NIGHT_END or time.time() >= NIGHT_START

def is_trading_time(time):
    return is_morning(time) or is_night(time)

def is_day_stock(time):
    return DAY_STOCK_BEGIN <= time.time() <= DAY_STOCK_END

def is_night_stock(time):
    return time.time() <= NIGHT_STOCK2_END or time.time() >= NIGHT_STOCK1_BEGIN

def KDJ(high, low, close, period, signal_k, signal_d, prev_info):
    return_data = {
        'k': None,
        'd': None,
        'j': None,
        'rsv': None,
        'high': None,
        'low': None,
        'close': None
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
    return_data['rsv'] = rsv
    return_data['high'] = max(high)
    return_data['low'] = min(low)
    return_data['close'] = close[-1]

    return return_data

def bbands_keltner(high, low, close, bb_period, bb_mult, kelt_period, kelt_mult):
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

def RSI(close, period):
    return ta.RSI(close, timeperiod = period)

def time_to_update_data(period):
    global current_time
    global next_update_time

    if current_time.minute == next_update_time.minute:
        next_update_time = current_time + datetime.timedelta(minutes = period)
        print(f"下次更新時間: {next_update_time}")
        return False
    else:
        return True
    
def update_data(api, period):
    global new_data1
    global new_data2
    global current_time

    print('開始抓取資料...')
    while True:
        current_time = datetime.datetime.now()
        if is_day_stock(current_time) and time_to_update_data(period):    # 確保還在開盤時間 & 在K棒更新前
            time.sleep(1)

        else:
            # 更新台指期數據
            new_data1 = pd.DataFrame().from_records([x.to_dict() for x in list(trader.tick)])
            new_data1.rename(columns = {
                'open': 'Open', 
                'high': 'High', 
                'low': 'Low', 
                'close': 'Close',
                'volume': 'Volume'
            }, inplace = True)
            new_data1 = new_data1.set_index('datetime')
            new_data1 = new_data1[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

            # 更新指標數據
            api.concatData()                               
            new_data2 = pd.DataFrame(api.data)     
            if(len(new_data2) == 0):
                print("沒有指標數據")
            new_data2["datetime"] = pd.to_datetime(new_data2["datetime"])
            new_data2 = new_data2.set_index("datetime")  
            print(f"結束這階段的抓取資料, {datetime.datetime.now().time()}")
            return

def running(api):
    api.main()

def strategy():
    ### Check if the current time is in trading time ###
    global current_time
    current_time = datetime.datetime.now()
    if not is_trading_time(current_time):
        return
    ##################################################

    ### Find the type of current position ###
    current_position_type = None
    if len(trader.position()) == 0:
        current_position_type = 'None'
    elif trader.position()['is_long']:
        current_position_type = 'long'
    elif trader.position()['is_short']:
        current_position_type = 'short'
    else:
        current_position_type = 'Hold'
    ##################################################

    ### Close the position if current time reaches the end of the time we set up ###
    if current_position_type != 'None' and DAY_STOCK_END > current_time.time() >= MORNING_END:
        if current_position_type == 'long':
            trader.sell(size = 1)
        elif current_position_type == 'short':
            trader.buy(size = 1)
    ##################################################

    ### Organize the received data ###
    global real_time_data1, real_time_data2
    global new_data1, new_data2

    time.sleep(1)
    new_data1 = new_data1.resample(f'1Min', closed = 'right', label = 'right').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    new_data1 = new_data1.dropna()
    real_time_data1.update(new_data1)
    to_be_added = new_data1.loc[new_data1.index.difference(real_time_data1.index)]
    real_time_data1 = pd.concat([real_time_data1, to_be_added])
    real_time_data1 = real_time_data1.tail(40)

    new_data2 = new_data2.resample('1T', label='right', closed='right').agg({
        '大戶買進' : 'sum',
        '散戶買進' : 'sum',
        '大戶掛單' : 'sum',
        '散戶掛單' : 'mean'
    })
    new_data2["大戶買進"] = new_data2["大戶買進"].cumsum()
    new_data2["散戶買進"] = new_data2["散戶買進"].cumsum()
    new_data2["大戶掛單"] = new_data2["大戶掛單"].cumsum()
    real_time_data2.update(new_data2)
    to_be_added = new_data2.loc[new_data2.index.difference(real_time_data2.index)]
    real_time_data2 = pd.concat([real_time_data2, to_be_added])
    real_time_data2 = real_time_data2.tail(40)

    print(f"資料更新完畢，{datetime.datetime.now().time()} 的資料: ")
    print(real_time_data1.tail(3))
    print()
    print(real_time_data2.tail(3))
    print('-' * 50)
    ##################################################

    ### Calculate the indicators with real-time data ###
    global kdj, bband, rsi
    
    print("計算指標中...")
    kdj = KDJ(
        real_time_data1.High.iloc[- 30 : ].values,
        real_time_data1.Low.iloc[- 30 : ].values,
        real_time_data1.Close.iloc[- 30 : ].values,
        25, 3, 3,
        kdj
    )
    bband = bbands_keltner(
        real_time_data1.High.iloc[-30 : ].values,
        real_time_data1.Low.iloc[-30 : ].values,
        real_time_data1.Close.iloc[-30 : ].values,
        20, 2,
        20, 1.5
    )
    rsi = RSI(real_time_data2.大戶買進.iloc[-30 : ].values, 25)
    print("指標計算完畢")
    ##################################################

    ### Check if BBand is squeezing ###
    global open_short_ready, is_squeeze

    if bband[0]:
        is_squeeze = True
        open_short_ready = False
    else:
        is_squeeze = False
    ################################################

    ### Data will be used later ###
    latest_close = real_time_data1.Close.iloc[-1]
    latest_open = real_time_data1.Open.iloc[-1]
    latest_bb_up = bband[1][-1]
    latest_bb_down = bband[2][-1]
    latest_j = kdj['j']
    latest_rsi = rsi[-1]
    ################################################

    ### Check if the precondition to open long position is satisfied ###
    if current_position_type == 'None' and is_squeeze:
        if latest_close > latest_bb_up and latest_j > 80:
            open_short_ready = True
            print("開空單條件符合")
    ################################################

    ### Check if open long position now ###
    if current_position_type == 'None' and open_short_ready:
        if latest_j < 80 and latest_close < latest_open and latest_rsi < 80:
            trader.sell(size = 1)
            sl_price = max(real_time_data1.High.iloc[-i] for i in range(1, 6))
            sell_ready = False
            print("開空單")
            return
    ################################################

    ### Check if the precondition to close long position is satisfied ###
    if current_position_type == 'short':
        if latest_j < 20 and latest_rsi < 20:
            sell_ready = True
            print("準備平倉")
            return
    ################################################

    ### Check if close long position now ###
    if current_position_type == 'short':
        if sell_ready and (latest_j >= 20 or latest_rsi >= 20):
            trader.buy(size = 1)
            sell_ready = False
            print("平倉")
            return

        elif latest_close > sl_price:
            trader.buy(size = 1)
            sell_ready = False
            print("停損平倉")
            return
    ################################################

## ==================== Deal with data ==================== ##
non_real_time_data = pd.read_csv('TXF_1T.csv', encoding = 'utf-8')
non_real_time_data.rename(columns = {'Unnamed: 0': 'datetime'}, inplace = True)
non_real_time_data.datetime = pd.to_datetime(non_real_time_data.datetime)

non_real_time_data = change_data_period(non_real_time_data, 1)
real_time_data1 = pd.DataFrame({
    'Open': non_real_time_data.Open,
    'High': non_real_time_data.High,
    'Low': non_real_time_data.Low,
    'Close': non_real_time_data.Close,
    'Volume': non_real_time_data.Volume
})
real_time_data2 = pd.DataFrame({
    '大戶買進': non_real_time_data.大戶買進,
    '散戶買進': non_real_time_data.散戶買進,
    '大戶掛單': non_real_time_data.大戶掛單,
    '散戶掛單': non_real_time_data.散戶掛單,
})

## ==================== Calculate indicators with non-real-time data ==================== ##
print("計算先前資料的指標中...")
pre_data_num = len(non_real_time_data)
for i in range(40, pre_data_num):
    kdj = KDJ(
        non_real_time_data.High.iloc[i - 30 : i].values,
        non_real_time_data.Low.iloc[i - 30 : i].values,
        non_real_time_data.Close.iloc[i - 30 : i].values,
        25, 3, 3,
        kdj
    )

bband = bbands_keltner(
    non_real_time_data.High.iloc[-30 : ].values,
    non_real_time_data.Low.iloc[-30 : ].values,
    non_real_time_data.Close.iloc[-30 : ].values,
    20, 2,
    20, 1.5
)
rsi = RSI(non_real_time_data.大戶買進.iloc[-30 : ].values, 25)
print("指標計算完畢")

## ==================== Start the strategy with real-time data ==================== ##
while NIGHT_STOCK2_END < datetime.datetime.now().time() < DAY_STOCK_BEGIN:
    if logged_in:
        print("已登入 等待開盤")
    else:
        trader = pyt.pytrader(strategy = id_strategy, api_key = api_key, secret_key = secret_key)
        aapi = getData(apiKey = api_key, secretKey = secret_key, beginTime = MORNING_START, endTime = MORNING_END)
        logged_in = True
        print("登入")

        time.sleep(1)

if is_day_stock(datetime.datetime.now()):
    if logged_in:
        print("已登入")
    else:
        trader = pyt.pytrader(strategy = id_strategy, api_key = api_key, secret_key = secret_key)
        api = getData(apiKey = api_key, secretKey = secret_key, beginTime = MORNING_START, endTime = MORNING_END)
        logged_in = True

    current_time = datetime.datetime.now()
    next_update_time = current_time + datetime.timedelta(minutes = 1)
    while is_day_stock(datetime.datetime.now()):
        thread1 = threading.Thread(target = update_data, args = (api, 1))
        thread2 = threading.Thread(target = running, args = (api, ))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        strategy()

    api.logout()
    trader.logout()
    logged_in = False
    time.sleep(1)
    
print("等待下次開盤...")
    