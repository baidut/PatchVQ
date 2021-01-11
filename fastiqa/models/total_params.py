from fastai.vision.all import *

def abbreviate(x):
    abbreviations = ["", "K", "M", "B", "T", "Qd", "Qn", "Sx", "Sp", "O", "N",
                     "De", "Ud", "DD"]
    thing = "1"
    a = 0
    while len(thing) < len(str(x)) - 3:
        thing += "000"
        a += 1
    b = int(thing)
    thing = round(x / b, 2)
    return str(thing) + " " + abbreviations[a]

def total_params(self):
    total_params = total_trainable_params = 0
    info = layers_info(self)
    for layer, size, params, trainable in info:
        if size is None: continue
        total_params += int(params)
        total_trainable_params += int(params) * trainable
    return {'Total params': total_params,
            'Total trainable params': total_trainable_params,
            'Total non-trainable params': total_params - total_trainable_params,
            'Summary': f'{abbreviate(total_trainable_params)} / {abbreviate(total_params)}'
            }

Learner.total_params = total_params
