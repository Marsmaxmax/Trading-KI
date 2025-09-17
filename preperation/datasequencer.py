import numpy as np
from util.calculations.customMath import percent
from util.calculations.indicators import calculate_rsi, calculate_ema
from config import PREDICTION_LENGTH, MINIMUM_PROFIT

# Lokale Normalisierung pro Sequenz
def normalize_with_last(seq=np.stack):
    '''Normalises np.stack by dividing through the most current close price and returns last price, for price reconstruction'''
    open = seq[:,0]
    close = seq[:,1]
    high = seq[:,2]
    low = seq[:,3]

    last_close = close[-1]

    open = open/last_close
    close = close/last_close
    high = high/last_close
    low = low/last_close

    return np.stack([open, close, high, low], axis=-1), last_close


# Generating Training Sequences


def create_sequences(candles: np.ndarray, seq_length=100):
    X_seq = []
    Y_candle = []

    end_idx = len(candles) - seq_length - 1

    for i in range(end_idx):
        # Eingabe-Sequenz
        seq = candles[i:i + seq_length, :]
        input_seq, last_close = normalize_with_last(seq)

        # Ziel: nächste Candle (1 Schritt nach der Sequenz), relativ zum last_close
        next_candle = candles[i + seq_length+1, :] / last_close

        X_seq.append(input_seq)
        Y_candle.append(next_candle)

    return np.array(X_seq), np.array(Y_candle)
