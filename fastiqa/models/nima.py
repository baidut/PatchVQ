# https://github.com/kentsyx/Neural-IMage-Assessment/blob/master/model/model.py
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

class NIMA(nn.Module):

    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model=None, num_classes=10):
        if base_model is None:
            base_model = models.vgg16(pretrained=True)
        super().__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out
        # if self.training:
        #     return out
        # else:
        #     mean_scores = []
        #     for prob in out.data.cpu().numpy():
        #         mean_scores.append(get_mean_score(prob))
        #     # std_score = get_std_score(prob)
        #     return torch.tensor(mean_scores)
