from . import IqaDataBunch
from fastai.vision import MSELossFlat

"""
IM2MOS
Input:  data_cls = ImageList
Output: label_cls = FloatList

# %%
%matplotlib inline
from fastiqa.bunches import *
# %%
Im2MOS(CLIVE)
Im2MOS(FLIVE640)
Patch1toMOS(FLIVE_2k, batch_size=1)

# %% for debugging
from fastiqa.bunches import *; Im2MOS(CLIVE).get_data()


#
# data = IM
"""

# dls # dataloaders
class IqaDataBunch(IqaData):
    batch_size = 64 # 16
    # num_workers = 16
    label_cls = FloatList
    data_cls = ImageList
    device = torch.device('cuda')

    name = None

    def __getattr__(self, k: str):
        # print(f'__getattr__: {k}')
        try:
            return getattr(self.label, k)
        except AttributeError:
            return getattr(self.data, k)

    @cached_property
    def data(self):
        print(f'loading data...{self.name}')
        data = self.get_data()
        print(f'DONE loading data {self.name}')
        return data

    def get_data(self):
        return NotImplementedError

    def reset(self, **kwargs):
        self.label._df = None
        self._data = None
        self.__dict__.update(kwargs)
        return self

    def _repr_html_(self):
        self.data.show_batch(rows=2, ds_type=DatasetType.Valid)
        print(self.data.__str__())
        return

    def show(self, fn):
        # TODO not found?
        # locate the image
        ds = self.df[self.df[self.fn_col] == fn]
        try: idx = int(ds.index[0])
        except IndexError: print('Not found!'); return

        # put selected sample on the top of the dataframe
        self.df.loc[0, :] = self.df.iloc[idx]
        self.df.loc[0, 'is_valid'] = True

        # avoid loading the whole database
        self.df = self.df.iloc[:self.batch_size]

        # backup the data
        data = self._data

        # reload the database
        self._data = None
        self.show_batch(ds_type=DatasetType.Single)
        self.reset(_data = data)
        return ds.T # otherwise won't show all columns


class Im2MOS(IqaDataBunch):
    _data = None
    label_col_idx = 0  # only one output
    loss_func = MSELossFlat()

    def get_data(self):
        data = self.get_list()
        data = self.get_split(data)
        data = self.get_label(data)
        data = self.get_augment(data)
        data = self.get_bunch(data)
        return data

    def get_list(self):
        return self.data_cls.from_df(self.df, self.path, suffix=self.fn_suffix, cols=self.fn_col,
                                     folder=self.folder)

    def get_split(self, data):
        if self.valid_pct is None:
            data = data.split_from_df(col='is_valid')
        else:
            if self.valid_pct == 1:  # TODO ignore_empty=True
                print('all samples are in the validation set!')
                data = data.split_by_valid_func(lambda x: True)  # All valid
                data.train = data.valid # train set cannot be empty
            elif self.valid_pct == 0:
                print('all samples are in the training set!')
                data = data.split_none()
            else: # 0 < self.valid_pct < 1
                print('We suggest using a fixed split with a fixed random seed')
                data = data.split_by_rand_pct(valid_pct=self.valid_pct, seed=2019)
        return data

    def get_label(self, data):
        return data.label_from_df(cols=self.label_cols[self.label_col_idx], label_cls=self.label_cls)

    def get_augment(self, data):
        return data if self.img_tfm_size is None else data.transform(size=self.img_tfm_size)

    def get_bunch(self, data, **kwargs):
        return data.databunch(bs=self.batch_size, val_bs=self.val_bs, **kwargs)
