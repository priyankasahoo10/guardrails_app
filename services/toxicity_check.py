# toxicity_check.py
from detoxify import Detoxify
import numpy as np

def toxicity_check(response):
    model = Detoxify('original')
    score = model.predict(response)
    toxicity_score = score['toxicity']
    toxicity_score = float(np.float32(toxicity_score))
    if toxicity_score > 0.7:
        return 'Toxic', toxicity_score
    return 'Non-toxic', toxicity_score
