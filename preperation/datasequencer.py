import pandas as pd
import numpy as np
from util.calculations.customMath import percent
from util.calculations.indicators import calculate_rsi, calculate_ema
from config import PREDICTION_LENGTH, MINIMUM_PROFIT
# Generating Training Sequences
def create_sequences(candles, seq_length=100):
    x_close,x_open,x_high,x_low, x_ema1,x_ema2,x_ema3, y_long1, y_long2, y_long3, y_short1, y_short2, y_short3, y_direction= [], [], [], [], [],[], [], [], [], [], [], [], [], []
    pred_len = PREDICTION_LENGTH
    close = candles[:,0]
    open = candles[:,1]
    high = candles[:,2]
    low = candles[:,3]

    ema1 = calculate_ema(close, pred_len)
    ema2 = calculate_ema(close, pred_len*2)
    ema3 = calculate_ema(close, pred_len*4)

    for i in range(pred_len, len(candles) - (seq_length+pred_len*4)):

        close_seq = close[i:i+seq_length]
        open_seq = open[i:i+seq_length]
        high_seq = high[i:i+seq_length]
        low_seq = low[i:i+seq_length]

        ema1_seq = ema1[i:i + seq_length]
        ema2_seq = ema2[i:i + seq_length]
        ema3_seq = ema3[i:i + seq_length]

        close0 = close[seq_length+i]
        close1 = close[i+seq_length+pred_len]
        close2 = close[i+seq_length+(pred_len*2)]
        close3 = close[i+seq_length+(pred_len*4)]
        percentage_dev = percent(close1, close0)
        multlong = 1+(MINIMUM_PROFIT/100)
        multshort = 1-(MINIMUM_PROFIT/100)
        long1 = close0*multlong < close1
        long2 = close0*multlong < close2
        long3 = close0*multlong < close3
        short1 = close0*multshort > close1
        short2 = close0*multshort > close2
        short3 = close0*multshort > close3

        x_close.append(close_seq)
        x_open.append(open_seq)
        x_high.append(high_seq)
        x_low.append(low_seq)
        x_ema1.append(ema1_seq)
        x_ema2.append(ema2_seq)
        x_ema3.append(ema3_seq)
        
        y_direction.append(percentage_dev)
        y_long1.append(long1)
        y_long2.append(long2)
        y_long3.append(long3)
        y_short1.append(short1)
        y_short2.append(short2)
        y_short3.append(short3)
    return np.transpose(np.array([x_close,x_open,x_high,x_low]), (1, 2, 0)), np.transpose(np.array([x_ema1,x_ema2,x_ema3]), (1, 2, 0)), np.transpose(np.array([y_direction]), (1, 0)), np.transpose(np.array([y_long1, y_long2, y_long3]), (1, 0)), np.transpose(np.array([y_short1, y_short2, y_short3]), (1, 0))
# def create_predsequences(candles, seq_length=100):
    #Todo LATER
    # return None