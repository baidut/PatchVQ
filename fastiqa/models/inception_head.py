
from fastai.vision.all import *
from ._body_head import *
from ..basics import IqaModel
from ._inceptiontime import *
import logging

class InceptionTimeModel(IqaModel):
    """
    c_out inception_time output
    n_out model output
    """
    siamese = False
    def __init__(self, c_in, bottleneck=32,ks=40,nb_filters=32,residual=True,depth=6, **kwargs):
        super().__init__(**kwargs)
        c_out = 1 if self.siamese else self.n_out
        self.inception_time = InceptionTime(c_in=c_in,c_out=c_out, bottleneck=bottleneck,ks=ks,nb_filters=nb_filters,residual=residual,depth=depth)

    @classmethod
    def from_dls(cls, dls, n_out=None, **kwargs):
        if n_out is None: n_out = dls.c
        return cls(c_in = dls.vars, n_out=n_out, **kwargs)

    def forward(self, x, x2=None): # more features
        # if self.training == False:
        #     self.siamese = False
        #     self.n_out = 1

        if x2 is not None:
            x = torch.cat([x, x2], dim=-1)  # 4096 features
        if self.siamese:
            x = x.view(self.n_out*x.shape[0], -1, x.shape[-1])  # *x.shape[2:]
            # [bs, n_out*length, features] -->  [bs*n_out, length, features]
        # [bs*n_out, length, features] -->  [bs*n_out, features, length ]
        y = self.inception_time(x.transpose(1, 2))
        return y.view(-1, self.n_out) if self.siamese else y

    def input_sois(self, clip_num=16):
        raise NotImplementedError
