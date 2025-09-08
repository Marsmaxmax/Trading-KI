import pandas as pd
import numpy as np
from sqlalchemy import false
import tensorflow as tf
import keras
from config import INPUT_LENGTH, MODEL_FILE
from keras.api.losses import CategoricalCrossentropy
from util.custom.customError import SemiLinearSquared
from util.custom.customfunctions import load_custom_model
from preperation.datasequencer import create_sequences

input_file = 'data/XAUUSD/train_1.csv'  # Name der Eingabedatei
data = pd.read_csv(input_file, header=None)
candles = data.values  # Close, Open, High, Low
x_candle, x_ema, y_direction, y_long, y_short = create_sequences(candles, INPUT_LENGTH)
# Überprüfen, ob das Modell existiert und geladen werden kann
model = load_custom_model(MODEL_FILE)

initial_learning_rate = 1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.1,
    staircase=False
)

customoptimizer = keras.optimizers.RMSprop(
    learning_rate=lr_schedule,
    rho=0.9,           
    momentum=0.7,      
    epsilon=1e-7,      
    centered=True,      
    clipnorm=1.0,       
    clipvalue=None,       
    global_clipnorm=None 
    )

model.compile(optimizer=customoptimizer, loss= CategoricalCrossentropy(),metrics=['accuracy','accuracy','accuracy'])
model.save(MODEL_FILE)