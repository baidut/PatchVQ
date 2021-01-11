__all__ = ['TestFeatures']

"""No need to use this, simply forward the features!
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.vqa.test_feats import *
from fastiqa.bunches.vqa._feat import *
dls = TestFeatures.from_json('json/TestVideos.json', bs=2, feats=Feature2dBlock('paq2piq2x2', clip_num=16))
b = dls.set_vid_id('VID_20201024_113442')
b = dls.one_batch()
b[0].shape
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

from . import *
from .test_videos import *
from tqdm import tqdm

class TestFeatures(TestVideos):
    def get_input_blocks(self):
        return self.feats

    def get_input_getters(self):
        return [ColReader(self.fn_col, pref=self.path/self.feat_folder/f.name) for f in self.feats]
