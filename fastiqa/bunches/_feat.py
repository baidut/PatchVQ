__all__ = ['Feature', 'FeatureBlock', 'Feature2dBlock', 'Feature3dBlock']

"""
TODO: should use avg pool instead of simply sampling
################################################################################
# %%
from cli_pool_features import *
# vid = 'ia-batch2/01HoneyDont'
vid = 'ia-batch8/North_Reading_School_Committee_Part_1_-_April_23_2012'
for feat in ['resnet-50-res5c', 'r2plus1d_18-2x2']:
    print(feat + ':', load_feature(vid, \
     feat, path='/media/zq/DB/db/LIVE_FB_VQA/features').shape)
# resnet-50-res5c: torch.Size([197, 4096, 1, 1])
# r2plus1d_18-2x2: torch.Size([34, 1, 512, 2, 2]) # 3 roi?
# %%
from cli_pool_features import *
for file in ['725.npy', '0_resnet-50_res5c.npy']:
    npy_file = Path('/media/zq/DB/tmp/merge_features')/file
    with open(npy_file, 'rb') as f:
        feat = np.load(f).reshape(-1, 4096)

    print(feat)
    print(file, feat.shape)
###########################
# %% Check broken features
###########################
# load the database and check features
from fastiqa.vqa import *
json_file = 'json/LIVE_FB_VQA_v1.json'
feats2d = FeatureBlock('paq2piq2x2_roi_soi_avg_pooled(sz=16)', clip_num=None, clip_size=None),
feats3d = FeatureBlock('r3d18-2x2_roi_soi_avg_pooled(sz=40to16)', clip_num=None, clip_size=None),
dls = Feat2MOS.from_json(json_file, feats = feats2d+feats3d)
df = dls.get_df()
path = dls.path

files = df.name.tolist()
for feat in dls.feats:
    folder = feat.name
    for file in files:
        npy_file = path/'features'/folder/(file+'.npy')
        if npy_file.exists():
            try:
                with open(npy_file, 'rb') as f:
                    feat = np.load(f)
                    if feat.shape!=(64, 2048):
                        print(file)

            except:
                print('borken: ',file)
                continue

#  ValueError: Cannot load file containing pickled data when allow_pickle=False
# just one borken
# borken:  yfcc-batch20/16108

# %%



'r3d18-2x2', 'r3d18-2x2_p013', 'r3d18-2x2-40clips_p013',
'paq2piq2x2', 'paq2piq2x2_p013',
'paq2piq2x2_roi_soi_avg_pooled(sz=16)',
'r3d18-2x2_roi_soi_avg_pooled(sz=40to16)'
'r3d18-2x2_roi_soi_avg_pooled(sz=16)'

# %%
from cli_pool_features import *

# vid = 'ia-batch8/North_Reading_School_Committee_Part_1_-_April_23_2012'
# vid = 'ia-batch3/02_12_12_Historic_Zoning_Commission'
# vid = 'backup2/b1' # 'ia-batch2/01-28-12_SF_News_pt2'
# vid = 'yfcc-batch9/7075'
vid = 'yfcc-batch8/6074' # [2, 3, 1280, 2, 2] only 2 clips
for feat in [
    'mobilenet_v2-2x2_p013(16clip)',
    'mobilenet3d_v2-2x2_p013'
    ]:
    print(feat + ':', load_feature(vid, \
    feat, path='/media/zq/DB/db/LIVE_FB_VQA/features').shape)
# only 16 frames?

# %%
1280*2*2*2
total_params(PVQ(c_in=4096))[0]
total_params(PVQ(c_in=10240))[0]
# %% [16, 3, 512, 2, 2]
# [11, 3, 1280, 2, 2]

r3d18-2x2: torch.Size([16, 512, 2, 2])
r3d18-2x2_p013: torch.Size([16, 3, 512, 2, 2])

paq2piq2x2: torch.Size([322, 512, 2, 2])
paq2piq2x2_p013: torch.Size([279, 3, 512, 2, 2])
paq2piq2x2_roi_soi_pooled: torch.Size([64, 2048])
# %%
######################################################################################
# doublecheck feat size

from fastiqa.vqa import *
json_file = 'json/LIVE_FB_VQA_v1.json'
feats2d = FeatureBlock('paq2piq2x2_roi_soi_avg_pooled(sz=16)', clip_num=None, clip_size=None),
feats3d = FeatureBlock('r3d18-2x2_roi_soi_avg_pooled(sz=40to16)', clip_num=None, clip_size=None),
dls = Feat2MOS.from_json(json_file, feats = feats2d+feats3d)
df = dls.get_df()
path = dls.path
for feat in dls.feats:
    folder = feat.name
    for file in ['yfcc-batch20/16108']:
        npy_file = path/'features'/folder/(file+'.npy')
        if npy_file.exists():
            try:
                with open(npy_file, 'rb') as f:
                    feat = np.load(f, allow_pickle=True)
                    if feat.shape!=(64, 2048):
                        print(file)

            except:
                print('borken: ',file)
                raise
######################################################################################
# df update
# %%
from fastiqa.iqa import *
dls = IqaDataBunch() << LIVE_FB_IQA
df = dls.get_df()
df[dls.fn_col]
df['ext'] = df[dls.fn_col].str.split('.').str[-1]
df['ext'].unique()
df[df.ext=='JPG']

open_image(dls.path/dls.folder/'blur_dataset/out_of_focus0098.JPG')
from fastai.vision import *
PILImage.create(dls.path/dls.folder/'blur_dataset/out_of_focus0098.JPG').show()
PILImage.create(dls.path/dls.folder/'blur_dataset/out_of_focus0098.jpg').show()

'blur_dat.aset/out_of_focus0098.tmp.jpg'.rsplit('.', 1)[0]

# %%
######################################################################################
"""

"""features as input block"""
from fastai.vision.all import *
from functools import partial
from .vqa._clip import takespread

# multiple features, then use fastuple

"""
dim=4: image feature: 4roi, 1536channels, 1x1
dim=5: video feature: 279frames, 3roi, 512channels, 2x2
dim=2 or 3: pooled video feature: [64, 2048]
"""
class Feature():
    @classmethod
    def create(cls, fn, clip_num=None, clip_size=None, roi_index=None): # one file name: id
        """fn: csv file name"""
        #df = pd.read_csv(fn+'.csv')
        # l = df['output'].tolist()
        # not sure about the ext, this doesn't work for files that contain . in their filenames
        # str(fn).rsplit('.', 1)[0]
        npy_file = fn + '.npy'
        if not Path(npy_file).exists():
            raise FileNotFoundError(npy_file)
        with open(npy_file, 'rb') as f:
            features = np.load(f)

        if roi_index is not None:
            if features.ndim == 2: # roi_soi_pooled
                features = features.reshape([4, -1, features.shape[-1]])
                features = features[roi_index, :]
            elif features.ndim == 4:
                features = features[roi_index, :]
            else:
                features = features[:, roi_index, :]

        L = features.shape[0]

        if clip_num is None:
            pass # do nothng
        elif clip_size is not None:
            index = takespread(list(range(L)),clip_num,clip_size)
            features = list((features[x, :] for x in index))
            features = np.concatenate(features).reshape(clip_num, -1)
        else: # 3d features
            if clip_num != features.shape[0]:
                index = takespread(list(range(L)),clip_num,clip_size=1)
                features = list((features[x, :] for x in index))
                features = np.concatenate(features).reshape(clip_num, -1)
            else:
                features = features.reshape(clip_num, -1)

        # print(f.shape) # (clip_num, 2048)
        return torch.tensor(features)

# def FeatureBlock(clip_num=None, clip_size=None, pref=None):
#     f = partial(Feature.create, clip_num=clip_num, clip_size=clip_size)
#     return TransformBlock(type_tfms=f)

class FeatureBlock:
    clip_num = None #8
    clip_size = None #8
    roi_index = None
    def __init__(self, name=None, **kwargs):
        self.name = self.__class__.__name__ if name is None else name
        self.__dict__.update(kwargs)

    def __call__(self):
        f = partial(Feature.create,
                   clip_num=self.clip_num,
                   clip_size=self.clip_size,
                   roi_index=self.roi_index)
        return TransformBlock(type_tfms=f)


class Feature2dBlock(FeatureBlock):
    clip_num = 16
    clip_size = 1

class Feature3dBlock(FeatureBlock):
    clip_num = 16
    clip_size = None
