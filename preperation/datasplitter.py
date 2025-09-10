
import pandas as pd

input_file = 'data/BTCUSDT_1m/output.csv'
size_packeages = 10000

data = pd.read_csv(input_file, header=None)
for i in range(len(data)//size_packeages): 
    data_chunk = data[((i-1)*size_packeages):(i*size_packeages)]
    data_chunk.to_csv(f'data/BTCUSDT_1m/10k_packs/pack_{i}.csv', header=False, index=False)