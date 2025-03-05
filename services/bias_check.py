from Dbias.bias_classification import classifier

def bias_check(text):
    label = classifier(text)
    if label[0]["label"] == 'Non-biased':
        score =  1.00 - label[0]["score"]
        return "Low bias", score
    else:
        return "High bias", label[0]["score"]
