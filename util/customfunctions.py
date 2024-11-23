import os
from tensorflow.keras.models import load_model
from util.customError import Profitloss

def load_custom_model(model_file):
    if os.path.exists(model_file):
        model = load_model(model_file, custom_objects={"ProfitLoss": Profitloss})
        print(f'Modell "{model_file}" erfolgreich geladen.')
        return model
    else:
        print(f'Modell nicht"{model_file}" gefunden')
        exit()