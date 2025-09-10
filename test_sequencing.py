from re import X
from config import INPUT_LENGTH
from preperation.datasequencer import create_sequences
import pandas as pd
import random

data = pd.read_csv('data/BTCUSDT_1m/output.csv', header=None)
candles = data.values
print('checkpoint 1 - data loaded')
num_tests = 100000

X_candles, X_balance, X_position, Y_long, Y_short, Y_hold, Y_close = create_sequences(candles, INPUT_LENGTH)
print('checkpoint 2 - data sequenced')
rel_balance=0
total_position_type=0
total_position_open=0
total_rel_profit=0
total_y_long=0
total_y_short=0
total_y_hold=0
total_y_close=0

total_open_trades=0

for i in range(num_tests):
    modulo = 1000
    if i % modulo == 0:
        print(f'checkpoint 3.{i/1000} - data is processing...')
    p = random.randint(0, len(X_candles)-1)
    profit = 0
    if X_position[p][0] == 1:
        total_open_trades += 1
        profit = X_candles[p][1][0] - X_position[p][1]
    elif X_position[p][0] == -1:
        total_open_trades += 1
        profit = X_position[p][1] - X_candles[p][1][0]

    total_position_type += X_position[p][0]
    total_position_open += X_position[p][1]
    rel_balance += (X_balance[p]/X_candles[p][1])  #because format OCHL and index 0 is open
    total_rel_profit += (profit/X_candles[p][1]) if X_position[p][0] != 0 else 0

    total_y_long += Y_long[p]
    total_y_short += Y_short[p]
    total_y_hold += Y_hold[p]
    total_y_close += Y_close[p]


    if num_tests <= 20:  #nur ausgeben wenn wenige Tests
        print(f'Balance of {i}: {X_balance[p]}')
        print(f'Position of {i}: {X_position[p]}')
        print(f'Y_long of {i}: {Y_long[p]}')
        print(f'Y_short of {i}: {Y_short[p]}')
        print(f'Y_hold of {i}: {Y_hold[p]}')
        print(f'Y_close of {i}: {Y_close[p]}')


avr_position_type = total_position_type / num_tests
avr_position_open = total_position_open / num_tests
rel_avr_balance = rel_balance / num_tests
rel_avr_profit = total_rel_profit / total_open_trades


avr_y_long = total_y_long / num_tests
avr_y_short = total_y_short / num_tests
avr_y_hold = total_y_hold / num_tests
avr_y_close = total_y_close / num_tests


print(f'Total Times started with open trade: {total_open_trades}\nAverage Position Type: {total_position_type}\nAverage Position Open: {avr_position_open}')
print(f'Total times closes: {total_y_close}\n Total times holds: {total_y_hold}\nTotal times longs: {total_y_long}\nTotal times shorts: {total_y_short}')
print(f'Relative Average Balance: {rel_avr_balance}\nRelative Average Profit: {rel_avr_profit}')
print(f'Average Y_long: {avr_y_long}\nAverage Y_short: {avr_y_short}\nAverage Y_hold: {avr_y_hold}\nAverage Y_close: {avr_y_close}') # all of thos should be arount 0.25 so every case is equally represented
print(f'total action cases (in %): {avr_y_close+avr_y_hold+avr_y_long+avr_y_short}')
print(f'total long and shorts (in %): {avr_y_long+avr_y_short}')