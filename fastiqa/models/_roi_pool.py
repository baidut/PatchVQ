import sys, os

########################## ROIPOOL<
# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/faster-rcnn.pytorch/lib")
# from model.roi_layers import ROIPool, ROIAlign  # PyTorch 1.0 specific!
########################## >ROIPOOL
from torchvision.ops import RoIPool, RoIAlign
# from .bunches._rois import Rois0123
from ._body_head import BodyHeadModel, num_features_model
from fastai.vision.all import *  # Tensor
import logging
from ..qmap import *


def get_idx(batch_size, n_output, device=None):  # idx of rois
    idx = torch.arange(batch_size, dtype=torch.float, device=device).view(1, -1)
    # 1 scores: idx = idx.t()
    # 4 scores: idx = torch.cat((idx, idx, idx, idx), 0).t()
    idx = idx.repeat(n_output, 1, ).t()
    idx = idx.contiguous().view(-1, 1)
    return idx  # .cuda()

# -1 to 1
def bbox2xyhw(bboxes, height, width):
    # Note that fastai gives percentage, not the coord
    # also pls concern the limitation
    # [16, n_output, 4]
    # b = torch.squeeze(bboxes, 1).t()
    # first remove negative values
    b = bboxes.view(-1, 4).t()
    # top, left, bottom, right = b[0], b[1], b[2], b[3] # fastai1
    left, top, right, bottom = b[0], b[1], b[2], b[3] # fastai2
    x, y = (left + 1) * width // 2, (top + 1) * height // 2
    h, w = (bottom - top) * height // 2, (right - left) * width // 2
    return torch.cat((x, y, h, w), 0).view(4, -1).t()


def bbox2roi(bboxes, height, width):
    b = bboxes.view(-1, 4).t()
    # (-1, 1) to (0, W) or (0, H)
    left, top, right, bottom = (b[0]+1)*width//2, (b[1]+1)*height//2, (b[2]+1)*width//2, (b[3]+1)*height//2 # fastai2
    return torch.cat((left, top, right, bottom), 0).view(4, -1).t()

# TODO multiscale
# also image score
def get_blockwise_rois(blk_size, img_size=None):  # [height, width]
    """
    a = get_blockwise_rois([3, 4], [400, 500])
    len(a), a

    :param blk_size:
    :param img_size: [height width]
    :return: a 1d list [x1, y1, x2, y2, x1, y1, x2, y2, ... ]
    """

    if img_size is None: img_size = [1, 1]
    y = np.linspace(0, img_size[0], num=blk_size[0] + 1)
    x = np.linspace(0, img_size[1], num=blk_size[1] + 1)
    # careful about the order of m, n! x increase first, so it's in inner loop
    a = []
    for n in range(len(y) - 1):
        for m in range(len(x) - 1):
            a += [x[m], y[n], x[m + 1], y[n + 1]]
    return a


def get_rand_rois(self, img_size, batch_size, device):
    """
    # idx with rois
    # %%

    # %%
    """
    h, w = img_size
    n = self.n_crops
    patch_h, patch_w = self.crop_sz
    # https://github.com/pytorch/pytorch/issues/10655
    sz = [batch_size * n, 1]
    x1 = torch.empty(sz, dtype=torch.float, device=device).uniform_(0, w - patch_w)
    y1 = torch.empty(sz, dtype=torch.float, device=device).uniform_(0, h - patch_h)
    # RoIPool will use interger
    # output size is 32x32
    x2 = x1 + patch_w
    y2 = y1 + patch_h
    rois = torch.cat((x1, y1, x2, y2), 1)
    idx = get_idx(batch_size, n, device)
    # print(rois.size(), idx.size()) [128, 4] [128, 1]
    return torch.cat((idx, rois), 1)


class NoRoIPoolModel(BodyHeadModel):
    # also use padded image
    # bunch opt
    # label_

    @classmethod
    def get_label_opt(cls, label_cls):
        # fixed size
        if label_cls.img_raw_size is not None:
            return {}
        elif label_cls.img_pad_size is None:
            return {'val_bs': 1}
        else: # use padded data
            return {'folder': '<=640_padded',
                'csv_labels': 'labels<=640_padded.csv'}



class RoIPoolModel(NoRoIPoolModel):
    rois = None
    drop = 0.5
    pool_size = (2, 2)
    joint = False
    bbox2xyhw = False
    # bunch_type = Rois0123
    output_features = False # output features instead of quality scores
    input_roi_sz = -1
    splitter = lambda self, model: [params(model.body), params(model.head)]
    head_kws = {'concat_pool': True}
    # roi_pool = None
    # create_head(self):
    # if self.joint:
    #     return create_head(nf * 4, 4)
    # Note that the output is 1 if not joint, output 1 for each roi

    def __init__(self, **kwargs):
        def next_power_of_2(x):
            return 1 if x == 0 else 2**(x - 1).bit_length()
        # remove simply fc
        # one could try only modify the last layer
        super().__init__(**kwargs)
        base_feat = self.body(torch.empty(1,3,640,640))
        scale = next_power_of_2(640// base_feat.size(-1))
        self.roi_pool = self.create_roi_pool(1/scale)
        logging.warning(f'roipool: size = {self.pool_size}, scale = 1/{scale}')

    def create_head(self):
        nf = self._num_features_model(self.body) * (2 if self.head_kws['concat_pool'] else 1)
        # roi pool output is 1 for each roi
        if self.joint:
            return self._create_head(nf * 4, 4, **self.head_kws)
        else:
            return self._create_head(nf, 1, **self.head_kws)

    def create_roi_pool(self, scale): # 32 resnet18
        return RoIPool(self.pool_size, scale)

    def input_fixed_rois(self, rois=None, img_size=None, batch_size=1, include_image=True, device=None): # , cuda=True
        """
        Note: img_size = (height, width)
        rois = np.array([324, 321, 733, 594, 701, 330, 1008, 534, 63, 259, 267, 395], np.float32)
        rois[::2] /= 1024
        rois[1::2] /= 768
        rois
        """
        if img_size is None:
            img_size = [1, 1]

        if rois is None:
            rois = [0.316406, 0.417969, 0.71582, 0.773438,
                    0.68457, 0.429688, 0.984375, 0.695312,
                    0.061523, 0.33724, 0.260742, 0.514323]
        rois = np.array(rois).reshape(-1, 4)
        rois[:, 0::2] *= img_size[1]
        rois[:, 1::2] *= img_size[0]
        a = [0, 0, img_size[1], img_size[0]] if include_image else []
        a += rois.reshape(-1).tolist()  # 1 dim list
        t = tensor(a, device=device)
        # if cuda:
        #     t = t.cuda()
        self.rois = t.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size, -1)

    def input_block_rois(self, blk_size=None, img_size=None, batch_size=1, include_image=True, device=None):
        # same for each item in a batch , so repeat it with batch size
        """
        blk_size = 32x32, then output  [1, (32*32+1)*4]
        [375, 500] to [12, 16] no need to bigger than 16x16
        :param blk_size:
        :param img_size:
        :param batch_size:
        :param include_image:
        :return:
        """
        if img_size is None:
            img_size = [1, 1]
        if blk_size is None:
            blk_size = [30, 30] # [[2, 2], [4, 4], [8, 8], [16, 16]]

        a = [0, 0, img_size[1], img_size[0]] if include_image else []
        if len(blk_size) == 2: # only one type of block
            blk_size = [blk_size]
        for sz in blk_size:
            a += get_blockwise_rois(sz, img_size)
        t = tensor(a, device=device)
        # if cuda:
        #     t = t.cuda()
        self.rois = t.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size, -1)
        # self.scale_rois = (img_size == 1)
        # https://discuss.pytorch.org/t/repeat-examples-along-batch-dimension/36217/3

    # def extract_features(self, im_data):


    def forward(self, im_data: Tensor, rois_data: Tensor = None, labels: Tensor = None, **kwargs) -> Tensor:
        # print(im_data) # float images [0 to 1]

        logging.debug(f'im_data {im_data.size()}')
        if self.is_3d:
            if im_data.size()[-4] != 3 and im_data.size()[-3] == 3:
                im_data = im_data.transpose(-4,-3)

        base_feat = self.body(im_data)  # torch.Size([16, 512, 12, 16])
        bbox_mode = labels is not None # bbox format

        if self.output_features and self.pool_size is None:
            return base_feat

        """
        [375, 500] to [12, 16]
        16/500,  12/375
        = 0.032
        """
        # the image shrinks! the position is wrong
        # print(base_feat.size())

        # TODO note when printing learn.summary batch size == 1
        batch_size = im_data.size(0)  # 16 or less if not enough image to pack
        # self.rois will overwrite the input rois

        # always generate it accordingly is to inefficient, but after generating one, we could re-use it until we train on next video
        if self.rois is None:
            if self.input_roi_sz is not -1: # (32,32)
                self.input_block_rois([self.input_roi_sz], [im_data.shape[-2], im_data.shape[-1]])


        if self.rois is not None:
            rois_data = self.rois.view(-1, 4)  # provide rois to predict patch quality map
        else:
            # print(type(rois_data))
            if rois_data is not None:
                if bbox_mode: #self.bbox2xyhw:
                    # bboxes, labels = rois_data
                    bboxes = rois_data
                    # [channels, height, width]
                    height, width = im_data.size(-2), im_data.size(-1)
                    rois_data = bbox2roi(bboxes, height, width)  # torch.Size([16, 4])
                else:
                    rois_data = rois_data.view(-1, 4)
            else: # no information provided
                # x1, y1, x2, y2: 0, 0, width, height
                t = tensor([0, 0, im_data.size(-1), im_data.size(-2)]).cuda()
                rois_data = t.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, 4)

            # bboxes: torch.Size([16, n_output, 4])
            # print(bboxes); print(bboxes.size())
            # print(xyhw, xyhw.size())  # [16*n_output, 4]  % torch.Size([16, 4])
            # xyhw = xyhw.view(-1,4)
        # print(rois_data)
        """
        idx: [bs*n_rois, 1]
        rois_data: [bs*n_rois, 4]
        indexed_rois: [bs*n_rois, 5]
        """
        n_output = int(rois_data.size(0) / batch_size)
        idx = get_idx(batch_size, n_output, im_data.device)  # torch.Size([16, n_output])
        # print(n_output, batch_size, idx.size(), rois_data.size())
        indexed_rois = torch.cat((idx, rois_data), 1)


        if self.is_3d:
            assert base_feat.size(2) == 1
            # print('base_feat:', base_feat.size()) # torch.Size([16, 512, 1, 17, 30])
            # no feature along the time axis? (one clip temporal output 1)
            base_feat = base_feat.squeeze(2)

        # automatically compute the downsample scale
        # if not hasattr(self, 'roi_pool'): #self.roi_pool is None: # maynot be correct... must be power of 2
        #     scale = next_power_of_2(im_data.size(-1) // base_feat.size(-1))
        #     self.roi_pool = self.create_roi_pool(1/scale)
        #     logging.warning(f'roipool: size = {self.pool_size}, scale = 1/{scale}')

        #  base feat [16, 512, 1, 17, 30] --> after roi pool    [16, 512, 2, 2]
        pooled_feat = self.roi_pool(base_feat, indexed_rois)
        logging.debug(f'base feat {base_feat.size()} --> pooled feat {pooled_feat.size()}')

        if self.output_features:
            # print('pooled_feat', pooled_feat.size())
            # is the sequence order correct?
            # compared with full video version and p1 version to double check
            sz = (batch_size, -1, pooled_feat.size(1)) + self.pool_size
            return pooled_feat.view(*sz)

        # if self.output_pooled_features:
        #     return pooled_feat # [bs*n_rois/n_gpu, 512, 20, 20]

        if self.joint:
            pooled_feat = pooled_feat.view(batch_size, -1, self.roi_pool_size[0], self.roi_pool_size[1])

        # print(batch_size, pooled_feat.size()) # ([1, 2048, 2, 2])
        pred = self.head(pooled_feat).view(batch_size, -1)  # 1 scores or 4 scores
        # print(pred.size())
        # print(base_feat.size())  # [bs, 512, 20, 20]
        # print(pooled_feat.size()) # [bs*n_rois, 512, 2, 2]
        # print(n_output)  # n_rois
        # print(pred.size())  # torch.Size([bs*n_rois, 1])
        logging.debug(f'batch_size={batch_size} pred:{pred.size()}')

        return pred

    def predict_quality_map(self, sample, blk_size=None):
        """
        :param sample: fastai.vision.image.Image  open_image()
        :param blk_size:
        :return:
        if you simply wanna quality map matrix:
        sample = open_image('img.jpg')
        self.model.input_block_rois(blk_size, [sample.shape[-2], sample.shape[-1]])
        t = self.predict(sample)
        """

        if blk_size is None:
            blk_size = [[32, 32]]  #[[8, 8]]

        if not isinstance(blk_size[0], list):
            blk_size = [blk_size]
        # input_block_rois(self, blk_size=None, img_size=[1, 1], batch_size=1, include_image=True):
        # [8, 8] #  [5, 5]
        # cuda = self.dls.device.type == 'cuda'
        # self.model.input_block_rois(blk_size, [sample.shape[-2], sample.shape[-1]], cuda=cuda)  # self.dls.img_raw_size
        self.input_block_rois(blk_size, [sample.shape[-2], sample.shape[-1]], device=sample.device)  # self.dls.img_raw_size
        # [768, 1024] [8, 8] is too big learn.data.batch_size

        # predict will first convert image to a batch according to learn.data
        # TODO backup self.dls
        # this is very inefficient
        # data = self.dls
        # self.dls = TestImages()
        # if type(sample) == fastai.vision.core.PILImage:
        #     sample = image2tensor(sample)
        t = self(sample)
        # self.dls = data

        a = t[0].data[1:].reshape(blk_size[0])  # TODO only allow one blk size

        # convert sample to PIL image
        # image = PIL_Image.fromarray((255 * image2np(sample.px)).astype(np.uint8))
        return QualityMap(a, sample, t[0].data[0])


class LegacyRoIPoolModel(RoIPoolModel): # old version, use to load CVPR2020 public model
    def create_head(self):
        return nn.Sequential(
          AdaptiveConcatPool2d(),
          nn.Flatten(),
          # assume backbone = resnet18, RuntimeError: running_mean should contain 2560 elements not 1024
          nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.Dropout(p=0.25, inplace=False),
          nn.Linear(in_features=1024, out_features=512, bias=True),
          nn.ReLU(inplace=True),
          nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=512, out_features=1, bias=True)
        )



# learn to predict distributions
# distr = True?

import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np # use gpu might be better? whatever

def get_mean_score(score): # this is just for one sample, not for a batch
    buckets = np.arange(1, 11)
    mu = (buckets * score).sum()
    return mu


def get_std_score(scores):
    si = np.arange(1, 11)
    mean = get_mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std

# buggy
class DistRoIPoolModel(RoIPoolModel):
    n_out_per_roi = 10
    # n_out is confusing when we have rois
    # n_out_per_roi

    def create_head(self):
        nf = self._num_features_model(self.body) * (2 if self.head_kws['concat_pool'] else 1)
        return self._create_head(nf, self.n_out, **self.head_kws)

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if self.training:
            return out
        else:
            mean_scores = []
            for prob in out.data.cpu().numpy():
                mean_scores.append(get_mean_score(prob))
            # std_score = get_std_score(prob)
            return torch.tensor(mean_scores)


class PaQ2PiQ_2(RoIPoolModel):
    head_kws = {'concat_pool': False}
    pool_size = (1,1)
