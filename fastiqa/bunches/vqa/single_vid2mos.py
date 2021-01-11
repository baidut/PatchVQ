__all__ = ['SingleVideo2MOS']

""" --> MOS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load_ext autoreload
%autoreload 2
# %%
from fastiqa.bunches.vqa.single_vid2mos import *
dls = SingleVideo2MOS.from_json('json/LIVE_FB_VQA.json', bs=3, clip_num=3, clip_size=2)
dls.set_vid_id('ia-batch8/North_Reading_School_Committee_Part_1_-_April_23_2012')
dls.show_batch()
dls.fn_col
# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.vqa.single_vid2mos import *
dls = SingleVideo2MOS.from_json('json/KoNViD.json', bs=3, clip_num=3, clip_size=2)
# dls.path
dls.set_vid_id('3240926995') # singing woman # 2999049224
dls.show_batch()
# %%
from fastiqa.bunches.vqa.single_vid2mos import *
file = 'json/LIVE_FB_VQA_30k.json' # LSVQ_Test
bs = 2
shuffle = False
dls = SingleVideo2MOS.from_json(
    file, # LIVE_FB_VQA_1k
    use_nan_label=True, clip_num=None, clip_size=16,
    bs=bs,
    shuffle=shuffle)
dls.set_vid_id('ia-batch8/North_Reading_School_Committee_Part_1_-_April_23_2012')
dls.show_batch()

# %%
dls._path
type(dls.path)
dls.path
# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

"""view one video as a collection of frames/clips"""
""" #73 single video as an image database """

"""
# %%
from fastiqa.all import *
dls = SingleVideo2MOS.from_json('json/KoNViD.json', bs=20, valid_bs=23)
dls.set_vid_id('2999049224')
self= dls.get_block()
self.summary(dls.df)
# %%
either use a single video database, or use learn.predict_from_video()
form the x for the model...
# %%
"""

from . import *
from .vid2mos import *
# the same corrdincate
# append df?
# build a df? no need
# generate image file list from df
#
#
# def _AllAsTrainTestSplitter(idx):
#     "Split `items` so that `val_idx` are in the validation set and the others in the training set"
#     def _inner(o):
#         return L(idx, use_list=True), L(idx, use_list=True)
#     return _inner

class SingleVideo2MOS(Vid2MOS):
    """TODO: skip frame """
    """may include roi
    clip_size is 1 for extracting 2d features
    clip_size > 1 for extracting 3d features

    """
    _vid_prop = None
    _df = None # frames and labels
    _df_vid_list = None # konvid vid list
    bs = 64
    vid_id = None
    use_nan_label = False
    clip_size = 1
    clip_num = 16 # 16x16 = 256 # 16x8 = 128 # cover all video
    # clip_num == bs, then 1 batch finish all feature extraction
    item_tfms = None # make sure no preprocessing is applied
    roi_col = None

    # @property
    # def __name__(self):
    #     return self.__class__.__name__ + '-' + self.vid_id

    def set_vid_id(self, x):
        # buggy: self.reset(vid_id=x, __name__=str(x), _df = None)
        self.vid_id = x
        self._df = None
        self.__name__ = str(x)
        return self

    # clip num is unset? -- just one clip
    def get_input_blocks(self):
        """input is one frame or one clip"""
        return ImageBlock if self.clip_size==1\
         else partial(ClipBlock, clip_size=self.clip_size),
         #else partial(ClipBlock, clip_size=self.clip_size),

    def get_input_getters(self):
        # 'fn'
        # not self.fn_col,
        # Note, here we use a generated df for each video
        return ColReader('fn', pref=self.path/self.folder/f'{self.vid_id}/'),

    @property
    def video_list(self):
        if self._df_vid_list is None:
            self._df_vid_list = super().get_df().set_index(self.fn_col)
        return self._df_vid_list

    def get_df(self):
        if self._df is None:
            assert self.clip_size >= 1
            path = self.path # Path('/media/zq/DB/db/KoNViD/')
            assert self.vid_id is not None
            if pd.api.types.is_integer_dtype(self.video_list.index.dtype):
                self.vid_id = int(self.vid_id)
            row = self.video_list.loc[self.vid_id] # KoNViD int id
            self._vid_prop = row
            # logging.debug(f'video id:   {self.vid_id} ')
            # logging.debug(f'row:   {row} ')
            # logging.debug(f'num_frame:   {row[self.frame_num_col]} ')
            num_frame = int(row[self.frame_num_col])
            # print(type(row[self.frame_num_col]))
            # print((row[self.frame_num_col])) 300.

            if self.clip_num is None:
                #  image case (self.clip_size is 1) :
                # files = [f'image_{x+1:05d}.jpg' for x in range(num_frame) ]
                # last_frame_index = (self.clip_size, row[self.frame_num_col], self.clip_num).astype(int).tolist()

                # for 2d case, we need .jpg
                # for 3d case, we don't need .jpg
                files = [f'image_{x+self.clip_size:05d}.jpg' for x in range(0, num_frame-self.clip_size+1, self.clip_size)] # drop last clip
            else:
                # fixed clip_num cases ( we used before)
                # index start with 1, not 0
                last_frame_index = np.linspace(self.clip_size, row[self.frame_num_col], self.clip_num).astype(int).tolist()
                files = [f'image_{x:05d}.jpg' for x in last_frame_index ]

                # use a different 3d backbone? smaller but faster?
                # change to use pytorch built in r3d 18
            self._df = pd.DataFrame(files, columns=['fn'])
        return self._df

    def get_block(self):
        self.get_df() # cache vid prop
        row = self._vid_prop
        InputBlocks = self.get_input_blocks()
        InputGetters = self.get_input_getters()

        if self.roi_col is not None:
            InputBlocks += RegressionBlock,
            InputGetters += lambda x: [row[x] for x in self.roi_col],

        # add roi information to this dataframe, the same roi for all models
        # do it at model side? not a good idea
        OutputBlocks = RegressionBlock,
        OutputGetters = lambda x: np.nan if self.use_nan_label else row[self.label_col].tolist() ,

        num_frame = int(row[self.frame_num_col])
        self.split_mode = 'all_as_train_test'
        return DataBlock(
            blocks = InputBlocks + OutputBlocks,
            getters = InputGetters + OutputGetters,
            n_inp = len(InputBlocks),
            splitter = self.get_splitter(),
            #_AllAsTrainTestSplitter(range(num_frame if self.clip_size==1 else self.clip_num))  #RandomSplitter(valid_pct=1), # all as valid
        )
