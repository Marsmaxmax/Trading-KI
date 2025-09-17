import os
from keras.models import load_model
from util.custom.customLoss import SemiLinearSquared, NonZeroBCELoss

def load_custom_model(model_file):
    if os.path.exists(model_file):
        model = load_model(model_file, custom_objects={
            "SemiLinearSquared": SemiLinearSquared,
            "NonZeroBCELoss": NonZeroBCELoss}
            )
        print(f'Modell "{model_file}" erfolgreich geladen.')
        return model
    else:
        print(f'Modell nicht"{model_file}" gefunden')
        exit()