import pandas as pd
import numpy as np
from util.calculations import calculate_rsi, calculate_ema

# Generating Training Sequences
def create_sequences(candles, seq_length=128):
    x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3, y_candle= [], [], [], [], [], [], [], []

    close = candles[:,0]
    open = candles[:,1]
    high = candles[:,2]
    low = candles[:,3]

    ema1 = calculate_ema(close, int(seq_length/4))
    ema2 = calculate_ema(close, int(seq_length/2))
    ema3 = calculate_ema(close, int(seq_length))

    for i in range(seq_length, len(candles) - (seq_length)): 
        close_seq = close[i:i+seq_length]
        open_seq = open[i:i+seq_length]
        high_seq = high[i:i+seq_length]
        low_seq = low[i:i+seq_length]

        ema1_seq = ema1[i:i + seq_length]
        ema2_seq = ema2[i:i + seq_length]
        ema3_seq = ema3[i:i + seq_length]

        x_close.append(close_seq)
        x_open.append(open_seq)
        x_high.append(high_seq)
        x_low.append(low_seq)

        x_ema1.append(ema1_seq)
        x_ema2.append(ema2_seq)
        x_ema3.append(ema3_seq)
        
        y_candle.append([close[i+seq_length], open[i+seq_length],high[i+seq_length],low[i+seq_length]])
    
    
    return np.array(x_close), np.array(x_open), np.array(x_high), np.array(x_low), np.array(x_ema1), np.array(x_ema2), np.array(x_ema3), np.array(y_candle)

def create_predsequences(candles, seq_length=128):
    x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3 = [], [], [], [], [], [], []

    close = candles[:,0]
    open = candles[:,1]
    high = candles[:,2]
    low = candles[:,3]

    ema1 = calculate_ema(close, int(seq_length/4))
    ema2 = calculate_ema(close, int(seq_length/2))
    ema3 = calculate_ema(close, int(seq_length))

    for i in range(seq_length, len(candles) - (seq_length)): 
        close_seq = close[i:i+seq_length]
        open_seq = open[i:i+seq_length]
        high_seq = high[i:i+seq_length]
        low_seq = low[i:i+seq_length]

        ema1_seq = ema1[i:i + seq_length]
        ema2_seq = ema2[i:i + seq_length]
        ema3_seq = ema3[i:i + seq_length]

        x_close.append(close_seq)
        x_open.append(open_seq)
        x_high.append(high_seq)
        x_low.append(low_seq)

        x_ema1.append(ema1_seq)
        x_ema2.append(ema2_seq)
        x_ema3.append(ema3_seq)
    
    
    return np.array(x_close), np.array(x_open), np.array(x_high), np.array(x_low), np.array(x_ema1), np.array(x_ema2), np.array(x_ema3)