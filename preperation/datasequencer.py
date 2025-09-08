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
    
    x_seq = []  # statt einzelne Features
    y_long, y_short = [], []

    close = candles[:, 0]
    open_ = candles[:, 1]
    high = candles[:, 2]
    low = candles[:, 3]

    end_idx = len(candles) - seq_length - pred_len

    for i in range(end_idx):
        # Rohsequenz zusammensetzen (shape: [seq_length, 4])
        seq = np.stack([
            close[i:i + seq_length],
            open_[i:i + seq_length],
            high[i:i + seq_length],
            low[i:i + seq_length]
        ], axis=-1)

        # Lokale Normalisierung der Sequenz
        seq = normalize_sequence(seq)
        x_seq.append(seq)

        # Einstiegspreis = Close direkt nach der Sequenz
        entry_price = close[i + seq_length]

        # Maximaler/Minimaler Preis innerhalb der Vorhersageperiode
        future_high = np.max(high[i + seq_length : i + seq_length + pred_len])
        future_low  = np.min(low[i + seq_length : i + seq_length + pred_len])

        # Bedingungen für Long und Short
        long_ok = future_high >= entry_price * (1 + MINIMUM_PROFIT)
        short_ok = future_low <= entry_price * (1 - MINIMUM_PROFIT)

        y_long.append(float(long_ok))
        y_short.append(float(short_ok))

    X = np.array(x_seq, dtype=np.float32)  # Shape: (batch, seq_length, 4)
    y_long = np.array(y_long, dtype=np.float32)
    y_short = np.array(y_short, dtype=np.float32)

    return X, y_long, y_short
