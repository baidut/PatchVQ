__all__ = ['Im2MOS']

"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.iqa.im2mos import *
dls = Im2MOS.from_json('json/CLIVE.json', bs=3)
# dls = Im2MOS.from_json('json/LIVE_FB_IQA.json', bs=3)
dls.show_batch()
dls.path

# %%
from fastiqa.bunches.iqa.im2mos import *
dls = Im2MOS() << 'json/CLIVE.json'
dls.show_batch()
# %%
dls.path
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

from . import *

"""
dls = CLIVE(bs=2).get_data()
dls.show_batch()
"""

# https://docs.fast.ai/vision.data#ImageDataLoaders.from_df
class Im2MOS(IqaDataBunch):
    def get_block(self):
        getters = [
           ColReader(self.fn_col, pref=self.path/self.folder, suff=self.fn_suffix),
           ColReader(self.label_col),
        ]
        return DataBlock(blocks=(ImageBlock, RegressionBlock),
                            getters=getters,
                            item_tfms = self.item_tfms,
                            splitter = self.get_splitter())
