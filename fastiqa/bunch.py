# __all__ = ['IqaDataBunch']

"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.iqa.im2mos import *
dls = Im2MOS.from_json('json/CLIVE.json')
dls.show_batch()
# %%
from fastiqa.bunches.iqa.im2mos import *
dls = Im2MOS('json/CLIVE.json')
dls.show_batch()
# %%
# type object got multiple values for keyword argument 'path'
# dls = IqaDataBunch.from_json('json/CLIVE.json', path='/some/other/path')
# avoid changing json properties but you can still change it if you have to

dls.path = '/media/zq/DB/db/LIVE_FB_IQA'
dls.csv_labels = 'labels<=640_padded.csv'
dls.fn_col = 'name_image'
dls.label_col = 'mos_image'
dls.folder = 'images'
dls.show_batch()


from fastiqa.basics import to_json
to_json(dls)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

# __getattr__: train
# __getattr__: n_inp
# __getattr__: device
# __getattr__: device

"""
Log
* removed @cached_property
* add clear_cache
* add @cached_property back LOL (but use the package instead)
* remove self.set_bs(bs)  --> self.reset(bs=bs)
"""

from fastai.vision.all import *
# from ..utils.cached_property import cached_property
from cached_property import cached_property
import logging

"""
# https://docs.fast.ai/data.transforms#IndexSplitter
def AllAsTrainTestSplitter():
    def _inner(o):
        all_idx = np.array(range_of(o))
        return L(all_idx, use_list=True), L(all_idx, use_list=True)
    return _inner

items = list(range(5))
splitter = IndexSplitter(None)
splitter2 = AllAsTrainTestSplitter()
splitter(items), splitter2(items)
"""

def AllAsTrainTestSplitter():
    def _inner(o):
        all_idx = np.array(range_of(o))
        return L(all_idx, use_list=True), L(all_idx, use_list=True)
    return _inner

def load_json(file, **kwargs):
    with open(file) as f:
        return json.load(f) #.update(kwargs)

def All(db):
    d = db.copy()
    d['split_mode'] = 'all_as_train_test'
    d['__name__'] += '_all'
    return d

def TestSet(db):
    d = db.copy()
    d['split_mode'] = 'all_as_train_test'
    d['__name__'] += '_test'
    assert 'csv_labels_test_set' in d
    d['csv_labels'] = d['csv_labels_test_set']
    if "test_bs" in d:
        d['bs'] = d['test_bs']
    return d

class IqaDataBunch():
    __name__ = None
    # _abbr = None
    # _data = None # cached data
    _bunching = False # don't allow access to dls attributes when _bunching
    _loading = False # don't allow access to dls attributes when loading

    bs = 8 # batch_size
    item_tfms = None # RandomCrop(500)
    fn_col = 'name'
    fn_suffix = ''
    label_col = 'mos'
    valid_col = 'is_valid'
    valid_pct = 0.2
    split_mode = 'use_is_valid_col'
    split_seed = 0
    sample_seed = 0
    sample = None
    shuffle = False # shuffle df
    shuffle_seed = 0
    csv_labels = "labels.csv"
    metric_idx = 0 # if multiple outputs, use the first output to compute metrics
    item_tfms = None
    use_old_split = False # to be removed soon
    create_batch = None

    # fastiqa1
    # num_workers = 16
    # label_cls = FloatList
    # data_cls = ImageList
    # device = torch.device('cuda')

    def __init__(self, **kwargs):
        # load user setting first
        self.__dict__.update(kwargs)  # must put this after, overwrite everything
        if self.__name__ is None: self.__name__ = self.__class__.__name__
        # if self._abbr is None: self.abbr = self.__name__.split('__')[0]

    def __str__(self):
        return self.__name__.split('__')[0]

    @cached_property
    def device(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return device

    @property
    # don't cache it, otherwise remember to clear cache
    def path(self):
        return Path(self.dir)

    @cached_property
    def _data(self):
        # build data, access new attribute, not found
        self._loading = True
        print(f'Loading data...{self}')
        data = self.get_data()
        self._loading = False
        print(f'Loaded data {self}')
        return data

    def __getitem__(self, index):
        if self._loading or self._bunching:
            raise AttributeError(f"index={index}: '{self}' data not loaded yet. loading={self._loading}, bunching={self._bunching}")
        return self._data[index]

    def __getattr__(self, k: str):
        # fail to get attributes, fetch it from self.data
        if self._loading or self._bunching:
            raise AttributeError(f"key={k}: '{self}' data not loaded yet. loading={self._loading}, bunching={self._bunching}")
        return getattr(self._data, k)

    def __setattr__(self, name, value):
        # best practice: don't set dls._vid_id outside of this class, it's an internal property
        # allow directly setting for internal properties
        # _vid_prop _df_vid_list
        # otherwise, the value comparison might be problemetic, self.__dict__[name] == value:
        # set anything will be monitored and warned

        #if name not in self.__dict__:
        # some might in class dict, use dir(self) might be better
        # use try except will lead to python chaos

        propobj = getattr(self.__class__, name, None)
        if isinstance(propobj, property):
            logging.debug("setting attr %s using property's fset" % name)
            if propobj.fset is None:
                raise AttributeError(f"can't set attribute {name}")
            propobj.fset(self, value)
            return
        elif '_' != name[0]:
            if name not in self.__dict__:
                logging.warning(f'__setattr__ {name} not existed')
            elif isinstance(value, (str,int,tuple)) and self.__dict__[name] == value:
                return # lazy
            if isinstance(value, (str,int,tuple)):
                logging.info(f'{self.__class__.__name__}({self}).{name}={value}')
            else:
                logging.debug(f'{self.__class__.__name__}({self}).{name}')
        self.__dict__[name] = value
        if '_data' in self.__dict__:
            del self.__dict__['_data'] # clear cache


    def get_splitter(self):
        if self.split_mode == 'all_as_train_test':
            return AllAsTrainTestSplitter()
        elif self.split_mode == 'random' or self.use_old_split:
            return RandomSplitter(valid_pct=0.2, seed=self.split_seed) # 2020
        elif self.split_mode == 'use_is_valid_col':
            return ColSplitter()
        else:
            raise NotImplementedError

    @cached_property
    def df(self):
        return self.get_df()

    def get_df(self):
        df = pd.read_csv(self.path/self.csv_labels)
        if self.item_tfms is not None:
            print('Warning: for quality assessment, data transform is not recommended!')

        if self.shuffle:  # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
            print('Shuffling...')
            df = df.sample(frac=1, random_state=self.shuffle_seed)

        if not 'is_valid' in df.columns and self.split_mode == 'use_is_valid_col':
            print('Splitting...')
            np.random.seed(self.split_seed)
            df['is_valid'] = np.random.rand(len(df)) < self.valid_pct
            df.to_csv(self.path/self.csv_labels, index=False)

        return df.sample(n=self.sample, random_state=self.sample_seed) if self.sample else df


    def get_block(self):
        raise NotImplementedError

    def get_data(self, source=None):
        if source is None: source = self.get_df()
        return self.get_block().dataloaders(source, create_batch=self.create_batch, bs=self.bs)

    def _repr_html_(self):
        """ show database summary, batch """
        # https://docs.fast.ai/data.block#DataBlock.summary
        blk = self.get_block()
        blk.summary(self.get_df()) # , show_batch=True (instead, we call object function)
        self.show_batch()
        return

    """
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    from fastiqa.iqa import *
    dls = Im2MOS.from_json('json/CLIVE.json')
    dls.show('6.bmp')
    dls.show_batch()
    # %%
    from fastiqa.iqa import *
    dls = ImRoI2MOS.from_json('json/LIVE_FB_IQA.json') # bs = 1 will give error Axes object
    dls.show('blur_dataset/motion0003.jpg')
    dls.show_batch()
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    def show(self, fn):
        """ show one specific data sample (ds) with file name (fn)"""
        # TODO not found?
        # locate the image
        df = self.get_df()
        ds = df[df[self.fn_col] == fn]
        try: idx = int(ds.index[0])
        except IndexError: print('Not found!'); return

        # put selected sample on the top of the dataframe
        df.loc[0, :] = df.iloc[idx]
        df.loc[0, self.valid_col] = True

        dls = self
        dls._data = dls.get_data(df)
        dls.show_batch(dls.valid.one_batch())
        return ds.T # otherwise won't show all columns

    @classmethod
    def from_json(cls, file, **kwargs): # load configurations from json
        return cls(**load_json(file), **kwargs)

    @classmethod
    def from_dict(cls, d, **kwargs):
        return cls(**d, **kwargs)

    def __lshift__(self, other):
        # import from
        # dls = Im2MOS(bs=2) << KonViD
        #
        # if isinstance(other, str):
        #     dbs = self.from_json(other)
        # elif isinstance(other, dict):
        #     dbs = self.from_dict(other)
        return self.bunch(other)

    # don't change dls.c
    # @classmethod
    """Note that after bunching, the dls will be updated"""
    def bunch(self, x, **kwargs):
        self._bunching = True
        # a = self.__class__(**self.__dict__)

        # format as the same database class using the new configurations (json)
        if isinstance(x, self.__class__):
            d = x.__dict__ # not all attributes
            return x
        elif isinstance(x, IqaDataBunch):
            d = x.__dict__ # not all attributes
        elif type(x) == str:
            d = load_json(x)
        elif type(x) == dict:
            d = x
        else:
            raise NotImplementedError(f'Unknown type {type(x)} when bunching')

        # another = self.copy()
        # is dict
        return self.__class__.from_dict(d)
        # another.__dict__.update(x)
        # another.__dict__.update(kwargs)
        # another._bunching = False
        # print('Done bunching...')
        # return another
        # a = cls(**x)
        # a.__dict__.update(kwargs)
        # return a
