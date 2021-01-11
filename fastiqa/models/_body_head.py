__all__ = ['BodyHeadModel']

"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.models._body_head import *
from fastiqa.models.resnet_3d import *
from fastai.vision.all import *

BodyHeadModel.compare_paramters(
    backbones = {
    'ResNet-18'         : resnet18,
    'ResNet(2+1)D-18'   : r2p1d18_K_200ep,
    'ResNet3D-18'       : r3d18_K_200ep,
    'ResNet-50'         : resnet50,
    'ResNet(2+)1D-50'   : r2p1d50_K_200ep,
    'ResNet3D-50'       : r3d50_KM_200ep
    }
)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

from fastai.vision.all import *
from ..basics import * # IqaModel
from ._3d import *
# from ..utils.cached_property import cached_property

class BodyHeadModel(IqaModel):
    backbone = None
    is_3d = None
    clip_num = 1
    _num_features_model = None
    _create_head = None
    cut = -1 # https://docs.fast.ai/vision.learner#Cut-a-pretrained-model
    n_out_per_roi = 1

    @staticmethod
    def split_on(m):
        return [[m.body], [m.head]]

    def create_body(self):
        try:
            return create_body(self.backbone)
        except StopIteration:
            logging.warning('Cut pretraiend {self.backbone.__name__} at -1')
            return create_body(self.backbone, cut=self.cut)

    def create_head(self):
        num_features = self._num_features_model(self.body)
        return self._create_head(num_features * 2 * self.clip_num, self.n_out_per_roi)
        # output 1 score per image/video location
        # to learn distributions, set it to 5

    def __init__(self, backbone=None, **kwargs):
        # remove simply fc
        # one could try only modify the last layer
        super().__init__(**kwargs)
        if backbone is not None:
            # self.__name__ += f' ({backbone.__name__})'
            self.backbone = backbone

        if self.is_3d is None:
            name = self.backbone.__name__
            self.is_3d = '3d' in name or '2p1d' in name or '2plus1d' in name or 'mc3' in name

        self._num_features_model = num_features_model_3d if self.is_3d else num_features_model
        self._create_head = create_head_3d if self.is_3d else create_head
        self.body = self.create_body()
        self.head = self.create_head()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # multi clip:
        #       bs x clip_num x clip_size x 3 x  H x W
        # -->   bs x clip_num x 3 x clip_size x H x W
        if self.is_3d and x.size()[-4] != 3 and x.size()[-3] == 3:
            x = x.transpose(-4,-3)

        batch_size = x.size(0)
        # video data
        # if x.size()[1] != 3 and x.size()[2] == 3:
        #     x = x.transpose(1,2)
        base_feat = self.body(x)
        # print('base_feat:',  base_feat.size()) # torch.Size([64, 8192, 128])
        pred = self.head(base_feat)
        return pred.view(batch_size, -1)

    @staticmethod
    def compare_paramters(backbones):
        body_params = []
        head_params = []

        if type(backbones) is dict:
            labels = backbones.keys()
            backbones = backbones.values()
        elif type(backbones) is list or tuple:
            labels = [backbone.__name__ for backbone in backbones]
        else:
            raise TypeError('backbones must be a list, tuple or dict')

        for backbone in backbones:
            model = BodyHeadModel(backbone=backbone)
            body_params.append(total_params(model.body)[0])
            head_params.append(total_params(model.head)[0])

        width = 0.35       # the width of the bars: can also be len(x) sequence
        ind = np.arange(len(backbones))

        p1 = plt.barh(ind, body_params, width)
        p2 = plt.barh(ind, head_params, width, left=body_params) # botoom for bar, left for barh

        plt.xlabel('#parameters')
        plt.title('Model parameters')
        plt.yticks(ind, labels)
        #plt.yticks(np.arange(0, 81, 10))
        plt.legend((p1[0], p2[0]), ('backbone', 'head'))

        plt.show()



# if fc:
#     return nn.Sequential(
#             nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#             nn.Linear(in_features=num_features, out_features=self.n_out, bias=True)
#         )
        # if fc:
        #     return nn.Sequential(
        #             nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
        #             nn.Linear(in_features=num_features, out_features=self.n_out, bias=True)
        #         )
