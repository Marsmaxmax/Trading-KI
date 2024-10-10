import pandas as pd
import numpy as np
from util.calculations import calculate_rsi, calculate_ema

# Generating Training Sequences
def create_sequences(candles, seq_length=100):
    x_close, x_open, x_high, x_low, x_rsi, x_ema20, x_ema50, x_ema100, y_up, y_down = [], [], [], [], [], [], [], [], [], []

    close = candles[:,0]
    open = candles[:,1]
    high = candles[:,2]
    low = candles[:,3]

    rsi = calculate_rsi(close, 20)
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)
    ema100 = calculate_ema(close, 100)

    for i in range(len(candles) - (seq_length + 10)): 
        close_seq = close[i:i+seq_length]
        open_seq = open[i:i+seq_length]
        high_seq = high[i:i+seq_length]
        low_seq = low[i:i+seq_length]

        rsi_seq = rsi[i + seq_length]
        ema20_seq = ema20[i + seq_length]
        ema50_seq = ema50[i + seq_length]
        ema100_seq = ema100[i + seq_length]

        x_close.append(close_seq)
        x_open.append(open_seq)
        x_high.append(high_seq)
        x_low.append(low_seq)

        x_rsi.append(rsi_seq)
        x_ema20.append(ema20_seq)
        x_ema50.append(ema50_seq)
        x_ema100.append(ema100_seq)
        
        # Berechnung des Trends
        if close[i+seq_length+10] > close[i+seq_length]*1.005:
            diffup = close[i+seq_length+10] - close[i+seq_length]
            y_up.append(diffup)
            y_down.append(0)
        elif close[i+seq_length+10] < close[i+seq_length]*1.005:
            diffdown = close[i+seq_length]
            y_up.append(0)
            y_down.append(diffdown)
        else:
            y_up.append(0)
            y_down.append(0)

        print(x_close.shape)
        print(y_down.shape)
    
    
    return np.array(x_close), np.array(x_open), np.array(x_high), np.array(x_low), np.array(x_rsi), np.array(x_ema20), np.array(x_ema50), np.array(x_ema100), np.array(y_up), np.array(y_down)
