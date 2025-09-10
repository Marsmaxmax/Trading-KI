import random
import pandas as pd
import numpy as np
from util.calculations.customMath import percent
from util.calculations.indicators import calculate_rsi, calculate_ema
from config import PREDICTION_LENGTH, MINIMUM_PROFIT

# Lokale Normalisierung pro Sequenz
def normalize_sequence(seq):
    mean = np.mean(seq, axis=0)
    std = np.std(seq, axis=0)
    return (seq - mean) / (std + 1e-8)

# Generating Training Sequences
def create_sequences(candles, seq_length=100):
    pred_len = PREDICTION_LENGTH
    
    x_seq, x_position, x_balance = [], [], []
    y_long, y_short, y_hold, y_close = [], [], [], []

    close = candles[:, 0]
    open_ = candles[:, 1]
    high = candles[:, 2]
    low = candles[:, 3]

    end_idx = len(candles) - seq_length - pred_len

    for i in range(end_idx):
        action_price = close[i + seq_length]
        # Rohsequenz zusammensetzen (shape: [seq_length, 4])
        seq = np.stack([
            close[i:i + seq_length],
            open_[i:i + seq_length],
            high[i:i + seq_length],
            low[i:i + seq_length]
        ], axis=-1)

        # Lokale Normalisierung der Sequenz
        # seq = normalize_sequence(seq)
        x_seq.append(seq)
        balance = random.uniform(0.9, 1.1)*action_price
        x_balance.append(balance)

        position_type = random.choice([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 1]) # 0: neutral, 1: long, -1: short
        position_profitable = random.choice([1,1,1,-1, -1]) # 1: profitabel, -1: unprofitabel
        position_difference = random.triangular(0.0001, 0.05, 0.03) # 0.01% bis 4%
        position_open = (action_price * (1 + (position_difference * position_profitable * position_type)) if not position_type == 0 else 0)

        position = np.stack([
            position_type, # long, short, neutral
            position_open
        ], axis=-1)
        x_position.append(position)

        future_high = np.max(high[i + seq_length : i + seq_length + pred_len])
        future_low  = np.min(low[i + seq_length : i + seq_length + pred_len])

        # Bedingungen für Long und Short
        long_profitable = future_high >= action_price * (1 + MINIMUM_PROFIT)
        short_profitable = future_low <= action_price * (1 -MINIMUM_PROFIT)

        trade_executable = balance >= action_price
        

        if position_type == 0: #neutral
            y_long.append(1.0 if long_profitable and trade_executable else 0.0)
            y_short.append(1.0 if short_profitable and trade_executable else 0.0)
            y_hold.append(0.0)
            y_close.append(0.0)
        elif position_type == 1: #long
            y_long.append(0.0)
            y_short.append(0.0)
            y_hold.append(1.0 if not short_profitable else 0.0)
            y_close.append(1.0 if short_profitable else 0.0)
        elif position_type == -1: #short
            y_long.append(0.0)
            y_short.append(0.0)
            y_hold.append(1.0 if short_profitable else 0.0)
            y_close.append(1.0 if not short_profitable else 0.0)


    X_candles = np.array(x_seq, dtype=np.float32)  # Shape: (batch, seq_length, 4)
    X_balance = np.array(x_balance, dtype=np.float32).reshape(-1, 1)  # Shape: (batch, 1)
    X_position = np.array(x_position, dtype=np.float32)  # Shape: (batch, 2)
    Y_long = np.array(y_long, dtype=np.float32)
    Y_short = np.array(y_short, dtype=np.float32)
    Y_hold = np.array(y_hold, dtype=np.float32)
    Y_close = np.array(y_close, dtype=np.float32)

    return X_candles, X_balance, X_position, Y_long, Y_short, Y_hold, Y_close
