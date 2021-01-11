# instead inputing 4 numbers, provide bounding box

getters = [lambda o: path/'train'/o, lambda o: img2bbox[o][0], lambda o: img2bbox[o][1]]


__all__ = ['ImBBox2MOS']

"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.iqa.im_bbox2mos import *
from fastai.vision.all import *
#dls = ImBBox2MOS.from_json('json/LIVE_FB_IQA.json', bs=3)
#dls

item_tfms = [Resize(128, method='pad'),]
# batch_tfms = [Rotate(), Flip(), Dihedral(), Normalize.from_stats(*imagenet_stats)]
dls = ImBBox2MOS(item_tfms=item_tfms, bs=3) << 'json/LIVE_FB_IQA.json'
dls.show_batch()

# %%
from fastiqa.bunches.iqa.im_bbox2mos import *
from fastai.vision.all import *
# before padding coordinates
# CropPad(800, pad_mode=PadMode.Zeros)
# item_tfms = [CropPad(800, pad_mode=PadMode.Zeros),]
# some images are broken
item_tfms = [Resize(128, method='pad'),]
# batch_tfms = [Rotate(), Flip(), Dihedral(), Normalize.from_stats(*imagenet_stats)]
dls = ImBBox2MOS(item_tfms=item_tfms, bs=3) << 'json/AVA.json'
dls.show_batch()

# TypeError: clip_remove_empty() takes 2 positional arguments but 3 were given

# %%

from fastiqa.bunches.iqa.im_bbox2mos import *
from fastai.vision.all import *
from fastiqa.iqa import *

item_tfms = [Resize(256, method='pad', pad_mode=PadMode.Zeros), RandomCrop(224),]
dls = ImBBox2MOS(item_tfms=item_tfms, bs=3) << TestSet(AVA)
dls.show_batch()
# %%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

from . import *

def my_bb_pad(samples, pad_idx=0):
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
    # don't remove empty
    # samples = [(s[0], *clip_remove_empty(*s[1:])) for s in samples]
    # samples = [(s[0], *clip_remove_empty(s[1], s[2]), s[2:]) for s in samples]
    max_len = max([len(s[2]) for s in samples])
    def _f(img,bbox,lbl,scores):
        bbox = torch.cat([bbox,bbox.new_zeros(max_len-bbox.shape[0], 4)])
        lbl  = torch.cat([lbl, lbl .new_zeros(max_len-lbl .shape[0])+pad_idx])
        return img,bbox,lbl,scores
    return [_f(*s) for s in samples]

MyBBoxBlock = TransformBlock(type_tfms=TensorBBox.create, item_tfms=PointScaler, dls_kwargs = {'before_batch': my_bb_pad})

# reshape to four coordinates
class BBoxColReader(ColReader):
    # _do_one is dealing with one cell located at (r,c)
        # return [super()._do_one(r, c)]
    def __call__(self, o, **kwargs):
        b = super().__call__(o, **kwargs)
        # a batch of coordinates,
        res = [b[i:i+4] for i in range(0, len(b), 4)]
        # print(res)
        return res

class ImBBox2MOS(IqaDataBunch):
    roi_col = ["left", "top", "right", "bottom"]
    # wont save
    # pad the images
    def get_df(self):
        # prefix = ['top', 'left', 'bottom', 'right', 'height', 'width']
        df = super().get_df()
        # wont use the padded coordinates
        if 'height' in df.columns and ('width' in df.columns):
            print('add bbox coordinates (no padding applied)')
            df['top'] = 0
            df['left'] = 0
            df['bottom'] = df['height']
            df['right'] = df['width']
        return df

    def get_block(self):
        if len(self.roi_col) == 4*4:
            bbox_lbl = ['image', 'patch1', 'patch2', 'patch3']
        elif len(self.roi_col) == 4:
            bbox_lbl = ['image']
        else:
            raise NotImplementedError
        print()
        getters = [
           ColReader(self.fn_col, pref=self.path/self.folder, suff=self.fn_suffix),
           BBoxColReader(self.roi_col),
           lambda o: bbox_lbl, # only one label
           ColReader(self.label_col),
        ]
        return DataBlock(blocks=(ImageBlock, MyBBoxBlock, BBoxLblBlock, RegressionBlock),
                        item_tfms = self.item_tfms,
                        getters=getters, n_inp=3, splitter = self.get_splitter())
