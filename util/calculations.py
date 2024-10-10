import pandas as pd
import numpy as np

# RSI-Calc
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi = np.concatenate((np.full(period, np.nan), rsi[period:]))
    
    return rsi

# EMA-Calc
def calculate_ema(prices, period=14):
    ema = np.zeros_like(prices)
    ema[period - 1] = np.mean(prices[:period])  # Initiales EMA auf Durchschnitt der ersten Perioden setzen
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    # Auffüllen der ersten Werte mit NaN für einheitliche Länge
    ema = np.concatenate((np.full(period - 1, np.nan), ema[period - 1:]))
    
    return ema