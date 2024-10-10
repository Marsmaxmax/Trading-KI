import pandas as pd

# CSV-Datei laden
input_file = 'train1.csv'  # Name der Eingabedatei
data = pd.read_csv(input_file, header=None)

candles = data.values

def close(x):
    return candles[x][0]
print(close(66))