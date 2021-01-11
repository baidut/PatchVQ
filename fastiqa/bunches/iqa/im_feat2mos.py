"""will be removed soon. use feat2mos.py
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.iqa.im_feat2mos import *
dls = ImFeat2MOS.from_json('json/LIVE_FB_IQA.json', bs=3)
dls.show_batch()


FeatureBlock
#
# try the freeze version (but bn not freezed. we could save the finetuned bn version )

Step1: convert one npy file to seperate files

unpack_results


# %%

# %%
files[0]
# when finetuning the PaQ2PiQ branch, we made a new sota model
# patch quality is not important, better patch quality map should not be the direction, focus on global quality

# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

from . import *

class NpyReader():
    """ load features from a numpy file"""
    def __init__(self, file, roi_index=None):
        if not Path(file).exists():
            raise FileNotFoundError(file)
        with open(file, 'rb') as f:
            features = np.load(f)

        if roi_index is not None:
            if features.ndim == 2:
                features = features.reshape([4, -1, features.shape[-1]])
                features = features[roi_index, :]
            else:
                features = features[:, roi_index, :]
        self._data = features

    def __call__(self, row):
        print(row.index)
        return L(self._data[row.index, :])


def TensorBlock():
    return TransformBlock(type_tfms=torch.tensor)

# after loading, no index available, must use a seperate feature file for each image....
# unless we have a name_index mapping (search vs load and combine)
# seperate feature file make more sense

class ImFeat2MOS(IqaDataBunch):
    feat_folder = 'features'
    feats = 'effnet3',
    roi_index = None

    def get_block(self):
        return DataBlock(
            blocks = [TensorBlock for _ in self.feats] + [RegressionBlock],
            getters = [NpyReader(self.path/self.feat_folder/f'{feat}.npy', self.roi_index) for feat in self.feats] + [ColReader(self.label_col)],
            splitter = self.get_splitter(),
            n_inp=len(self.feats),
        )

    @property
    def vars(self):
        b = self._data.one_batch()
        return sum([b[x].shape[-1] for x in range(len(b)-1)])

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
