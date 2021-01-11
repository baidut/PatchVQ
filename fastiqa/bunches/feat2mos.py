"""
load features for image/video quality assessment databases
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load_ext autoreload
%autoreload 2
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export configurations
from fastiqa.bunches.feat2mos import *
feats = Feature2dBlock('paq2piq2x2', clip_num=8), Feature3dBlock()
dls = Feat2MOS.from_json('json/LIVE_VQC.json', bs=16, feats = feats)
dls.to_json()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.feat2mos import *
dls = Feat2MOS.from_json('json/KoNViD.json', bs=3)
dls.show_batch()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.feat2mos import *
feats = FeatureBlock('effnet3'),
dls = Feat2MOS.from_json('json/LIVE_FB_IQA.json', bs=3, feats=feats)
dls.show_batch()
# %%
dls.vars
dls.c
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load_ext autoreload
%autoreload 2
from fastiqa.bunches.feat2mos import *
dls = Feat2MOS.from_json('json/LIVE_FB_VQA_1k.json', bs=3)
dls.show_batch()
#%%
# TODO: feature visualization?

#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

from . import *
from ._feat import *

class Feat2MOS(IqaDataBunch):
    bs = 8
    feat_folder = 'features'
    feats = [] # (None,) # Feature2dBlock('paq2piq2x2'), Feature3dBlock('r3d18-2x2')
    suff = None
    # @property
    # def roi_index(self):
    #     return self.feats[0].roi_index
    #
    # @roi_index.setter
    # def roi_index(self, index):
    #     for feat in self.feats:
    #         feat.roi_index = val
    #     #assert len(self.label_col) == len(index)
    #     if len(self.label_col) != len(index):
    #         self._label_col = self.label_col[index]
    #     # self._roi_col = np.array(self.roi_col)[[0,2], :].flatten().tolist()
    #     # roi_col not used

    def get_block(self):
        return DataBlock(
            blocks = self.feats + (RegressionBlock, ),
            getters = [ColReader(self.fn_col, pref=self.path/self.feat_folder/f.name) for f in self.feats] + [ColReader(self.label_col)],
            splitter = self.get_splitter(),
            n_inp=len(self.feats),
        )

    def get_df(self):
        df = super().get_df()
        # remove file extension if exists
        if self.suff is '':
            df[self.fn_col] = df[self.fn_col].str.rsplit('.',1).str[0]
        return df

    @property
    def vars(self):
        b = self._data.one_batch()
        return sum([b[x].shape[-1] for x in range(len(self.feats))])


    def check_file_exists(self):
        df = self.get_df()
        files = []
        dir = self.path/self.feat_folder
        for f in df[self.fn_col]:
            #if not (path/'jpg'/f).exists(): # file all exists
            for feat in self.feats:
                name = feat.name
                if not (dir/f'{name}/{f}.npy').exists():
                    files.append(dir/f'{name}'/f)
                #print(path/f'features/{feature}'/f)
        return files

    def show_batch(self, *args, **kwargs):
        b = self._data.one_batch()
        print(f'batch size = {self.bs}')
        for x in b[:-1]:
            print(list(x.shape))
        print('--->', list(b[-1].shape))

    def bunch(self, x, **kwargs):
        # format as the same database class using the new configurations (json)
        a = super().bunch(x, **kwargs)
        a.feats = self.feats
        return a




#
