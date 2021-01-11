from fastai.vision.all import *

def in_channels_3d(m):
    "Return the shape of the first weight layer in `m`."
    for l in flatten_model(m):
        if getattr(l, 'weight', None) is not None and l.weight.ndim==5: #4
            return l.weight.shape[1]
    raise Exception('No weight layer')

def dummy_eval_3d(m, size=(64,64)):
    "Evaluate `m` on a dummy input of a certain `size`"
    ch_in = in_channels_3d(m) #  weight of size [64, 3, 7, 7, 7]
    # 1, ch_in, 64 64
    # 1, ch_in, 2, 64 64 # 2 frames per video
    x = one_param(m).new(1, ch_in, 2, *size).requires_grad_(False).uniform_(-1.,1.)
    with torch.no_grad(): return m.eval()(x)

# Cell
def model_sizes_3d(m, size=(64,64)):
    "Pass a dummy input through the model `m` to get the various sizes of activations."
    with hook_outputs(m) as hooks:
        _ = dummy_eval_3d(m, size=size)
        return [o.stored.shape for o in hooks]

def num_features_model_3d(m):
    "Return the number of output features for `m`."
    sz,ch_in = 32,in_channels_3d(m)
    while True:
        #Trying for a few sizes in case the model requires a big input size.
        try:
            return model_sizes_3d(m, (sz,sz))[-1][1]
        except Exception as e:
            sz *= 2
            if sz > 2048: raise e

class AdaptiveConcatPool3d(Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool3d(self.size)
        self.mp = nn.AdaptiveMaxPool3d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def create_head_3d(nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True, bn_final=False, lin_first=False, y_range=None):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    ps = L(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool3d() if concat_pool else nn.AdaptiveAvgPool3d(1)
    layers = [pool, Flatten()]
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += LinBnDrop(ni, no, bn=True, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if y_range is not None: layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)
