from ._loss import *
from fastai.vision.all import *
# EDMLoss

@use_kwargs_dict(reduction='mean')
def EDMLossFlat(*args, axis=-1, floatify=True, **kwargs):
    "Same as `nn.EDMLoss`, but flattens input and target."
    return BaseLoss(EDMLoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)

@use_kwargs_dict(reduction='mean')
def HuberLossFlat(*args, axis=-1, floatify=True, **kwargs):
    "Same as `nn.L1Loss`, but flattens input and target."
    return BaseLoss(nn.SmoothL1Loss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
