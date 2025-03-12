import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM
from tensorflow.keras.models import save_model, load_model
import os

from config import INPUT_LENGTH, MODEL_FILE
from preperation.datasequencer import create_sequences
from util.custom.customfunctions import load_custom_model

# CSV-Datei laden
input_file = 'data/XAUUSD/output.csv'  # Name der Eingabedatei
training_set_1 = 'data/XAUUSD/train_1.csv'
training_set_2 = 'data/XAUUSD/train_2.csv'
training_set_3 = 'data/XAUUSD/train_3.csv'
training_set_4 = 'data/XAUUSD/train_4.csv'
training_sets = [training_set_1, training_set_2, training_set_3, training_set_4]

model =load_custom_model(MODEL_FILE)

data = pd.read_csv(training_set_4, header=None)
candles = data.values
x_candle, x_ema, y_direction, y_long, y_short = create_sequences(candles, INPUT_LENGTH)

results = model.evaluate([x_candle,x_ema], [y_direction, y_long, y_short], verbose=7)

up_accuracy = results[3]
down_accuracy = results[4]

test_acc = (up_accuracy + down_accuracy)/2
print(results)
