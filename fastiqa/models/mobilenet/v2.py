from ._v2 import get_model
__all__ = ['mobilenet3d_v2']
# https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3

import torch
from collections import OrderedDict

def remove_module_in_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def mobilenet3d_v2(pretrained=True, **kwargs):
    path_to_model_state = '/media/zq/DB/pth/kinetics_mobilenetv2_1.0x_RGB_16_best.pth'
    model = get_model(num_classes=600, sample_size=112, width_mult=1.)
    if pretrained:
        model_state = torch.load(path_to_model_state, map_location=lambda storage, loc: storage)
        model.load_state_dict(remove_module_in_state_dict(model_state['state_dict']))
    return model
