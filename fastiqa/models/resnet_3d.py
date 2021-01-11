from .resnet3d import resnet, resnet2p1d, resnext
import torch
# from fastiqa.models import resnet

DIR = 'pth/'

def r3d18_K_200ep(pretrained=False, **kwargs):
    model = resnet.generate_model(model_depth=18, n_classes=700, **kwargs)
    PATH = DIR + 'fastai-r3d18_K_200ep.pth'
    # PATH = '/media/zq/DB/code/fastiqa2_dev/r2p1d18_K_200ep.pth'
    if pretrained:
        model.load_state_dict(torch.load(PATH))
    return model

# K 700
# KM 1039
# KMS 1139
def r3d50_KM_200ep(pretrained=False, **kwargs):
    model = resnet.generate_model(model_depth=50, n_classes=1039, **kwargs)
    PATH = DIR + 'fastai-r3d50_KM_200ep.pth'
    if pretrained:
        model.load_state_dict(torch.load(PATH))
    return model


def r2p1d50_K_200ep(pretrained=False, **kwargs):
    model = resnet2p1d.generate_model(model_depth=50, n_classes=700, **kwargs)
    PATH = DIR + 'fastai-r2p1d50_K_200ep.pth'
    if pretrained:
        model.load_state_dict(torch.load(PATH))
    return model


def r2p1d18_K_200ep(pretrained=False, **kwargs):
    model = resnet2p1d.generate_model(model_depth=18, n_classes=700, **kwargs)
    PATH = DIR + 'fastai-r2p1d18_K_200ep.pth'
    if pretrained:
        model.load_state_dict(torch.load(PATH))
    return model


# def resnext3d(pretrained=False, depth=50, **kwargs):
#     model = resnext.generate_model(model_depth=depth, n_classes=700, **kwargs)
#     #PATH = '/media/zq/DB/code/fastiqa2/fastai-r2p1d18_K_200ep.pth'
#     if pretrained:
#         pass
#         #model.load_state_dict(torch.load(PATH))
#     return model
#
# def resnext3d18(pretrained=False, **kwargs):
#     return resnext3d(pretrained, depth=18)
#
# def resnext3d50(pretrained=False, **kwargs):
#     return resnext3d(pretrained, depth=50)
