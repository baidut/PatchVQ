__all__ = ['VidSP2MOS', 'SP2MOS']

"""Video + Spatial patch --> MOS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.vqa.vid_sp2mos import *
from fastai.vision.all import CropPad
dls = VidSP2MOS.from_json('json/LIVE_FB_VQA_pad500.json', bs=2, clip_num=3, clip_size=2, item_tfms=CropPad(500))
dls.show_batch()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.vqa.vid_sp2mos import *
from fastai.vision.all import CropPad
dls = VidSP2MOS.from_json('json/LIVE_FB_VQA.json', bs=2, clip_num=3, clip_size=2, item_tfms=CropPad(500))
dls.show_batch()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.vqa.vid_sp2mos import *
# dls = VidSP2MOS.from_json('json/LIVE_FB_VQA_1k.json', bs=1, clip_num=3, clip_size=2, item_tfms=None)
dls = VidSP2MOS.from_json('json/LIVE_FB_VQA_30k.json', bs=1, clip_num=3, clip_size=2, item_tfms=None)
dls.show_batch()

# %%
dls.show_batch()
df = dls.get_df()
df.columns
# %%
self = dls
df = dls.get_df()
[x in df.columns for x in self.vid_p1_label_col]
all([x in df.columns for x in self.vid_p1_rois_col])
dls.get_df().columns
# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


# DataBlock, RegressionBlock, ColReader, RandomSplitter, TensorImage, TensorCategory, TensorBBox, LabeledBBox, PILImage
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt


from . import *
from .vid2mos import *



def create_roi_batch(data):
    xs, rois, ys = [], [], []
    for d in data:
        xs.append(d[0])
        rois.append(d[1])
        ys.append(d[2])

    try:
        xs_ = torch.cat([TensorImage(torch.cat([im[None] for im in x], dim=0))[None] for x in xs], dim=0)
        ys_ = torch.cat([y[None] for y in ys], dim=0)
        rois_ = torch.cat([roi[None] for roi in rois], dim=0)
    except:
        print('Error@create_batch')
        for idx, x in enumerate(xs):
            print(idx, x.size()) # torch.Size([3, 500, 500])
        xs_ = torch.cat([TensorImage(torch.cat([im[None] for im in x], dim=0))[None] for x in xs], dim=0)
        print('xs_ is ok')
        print(xs_)
        raise

    return TensorImage(xs_), TensorCategory(rois_), TensorCategory(ys_)


# https://docs.fast.ai/data.core#TfmdDL.show_batch
def show_vid_roi_batch(dls, b=None, max_n=4, figsize=(30, 6), vertical=False, **kwargs):
    # rb --> rois in a batch
    max_n = min(max_n, dls.bs)
    xb, rb, yb = dls.one_batch() if b is None else b
    if vertical:
        fig, axes = plt.subplots(ncols=max_n, nrows=1, figsize=figsize, **kwargs) # , figsize=(12,6), dpi=120
    else:
        fig, axes = plt.subplots(nrows=max_n, ncols=1, figsize=figsize, **kwargs) # , figsize=(12,6), dpi=120
    if max_n == 1: axes = [axes] # only one item
    for i in range(max_n):
        xs, rs, ys = xb[i], rb[i], yb[i]
        # axes[i].imshow(x.permute(1,2,0).cpu().numpy())
        # axes[i].set_title(f'{ys.item():.02f}')
        rois = rs.cpu().numpy().reshape(-1,4).tolist()
        tbbox = LabeledBBox(TensorBBox(rois), ['video', 'p1', 'p2', 'p3'])
        timg = TensorImage(clip2image(xs).cpu())*255

        tpil = PILImage.create(timg)
        ctx = tpil.show(ax=axes[i])
        tbbox.show(ctx=ctx)

        # ys 0 dim tensor
        lbl = ','.join("%.2f" % x for x in ys.tolist()) if ys.dim() > 0 else "%.2f" % ys
        axes[i].set_title(lbl)
        axes[i].axis('off')

class VidSP2MOS(Vid2MOS):
    pad = None # shift_based on padding
    # if loss_func is None:
    # loss_func = getattr(dls.train_ds, 'loss_func', None)
    def get_df(self):
        # prefix = ['top', 'left', 'bottom', 'right', 'height', 'width']
        df = super().get_df()
        """prepare roi label"""
        if any([x not in df.columns for x in self.roi_col]):
            # unpack position
            for p in ['p1', 'p2', 'p3']:
                df[p] = df[p].apply(literal_eval)
                columns = [x+'_'+p for x in ['left', 'right', 'top', 'bottom', 'start', 'end']]
                df[columns] = pd.DataFrame(df[p].to_list(), columns=columns)
            df['top_vid'] = 0
            df['left_vid'] = 0
            df['bottom_vid'] = df['height'] # df['height_vid']
            df['right_vid'] = df['width'] # df['width_vid']
            df['start_vid'] = 0
            df['end_vid'] = df[self.frame_num_col]

            # shift if needed
            if self.pad:
                top_shift = (self.pad - df['height_vid']) // 2
                left_shift = (self.pad - df['width_vid']) // 2

                for s in ['_vid', '_p1', '_p2', '_p3']:
                    df['top' + s] += top_shift
                    df['left' + s] += left_shift
                    df['bottom' + s] = df['top' + s] + df['height' + s]
                    df['right' + s] = df['left' + s] + df['width' + s]

            df.to_csv(self.path/self.csv_labels, index=False)
        return df

    def get_block(self):
        VideoBlock = partial(MultiClipBlock, clip_size=self.clip_size, clip_num=self.clip_num)
        return DataBlock(
            blocks     = (VideoBlock, RegressionBlock, RegressionBlock),
            getters = [
               ColReader(self.fn_last_frame_col, pref=self.path/self.folder),
               ColReader(self.roi_col),
               ColReader(self.label_col),
            ],
            n_inp = 2,
            splitter   = self.get_splitter(),
            item_tfms  = self.item_tfms,  # RandomCrop(500),
        )

    def create_batch(self, data):
        return create_roi_batch(data)

    def show_crops(self, fn):
        """ only for validate patch coordinates"""
        df = self.get_df()
        row = df[df[self.fn_col] == fn].iloc[0]
        # db = 'ia'
        # if type(row['identifier']) == str else 'yfcc'
        # folder = self.path/self.folder/f"{db}-batch{row['batch']}/{row['identifier']}"
        folder = self.path/self.folder/fn
        p1 = row['left_p1'], row['top_p1'], row['right_p1'], row['bottom_p1']
        tbbox = LabeledBBox(TensorBBox([p1]), ['p1'])

        tpil = PILImage.create(folder/f"image_{1+row['start_p1']:05d}.jpg")
        ctx = tpil.show()
        tbbox.show(ctx=ctx)

        tpil = PILImage.create(folder/f"image_{1+row['start_p2']:05d}.jpg")
        tpil.show()

        p3 = row['left_p3'], row['top_p3'], row['right_p3'], row['bottom_p3']
        tbbox = LabeledBBox(TensorBBox([p3]), ['p3'])

        tpil = PILImage.create(folder/f"image_{1+row['start_p3']:05d}.jpg")
        ctx = tpil.show()
        tbbox.show(ctx=ctx)
        print(row)
        print(folder)


class SP2MOS(VidSP2MOS):
    """Spatial patch to mos"""

    # if loss_func is None:
    # loss_func = getattr(dls.train_ds, 'loss_func', None)

    def get_block(self):
        VideoBlock = partial(MultiClipBlock, clip_size=self.clip_size, clip_num=self.clip_num)
        return DataBlock(
            blocks     = (VideoBlock, RegressionBlock, RegressionBlock),
            getters = [
               ColReader(self.fn_last_frame_col, pref=self.path/self.folder),
               ColReader(self.p1_rois_col),
               ColReader(self.p1_label_col),
            ],
            n_inp = 2,
            splitter   = self.get_splitter(),
            item_tfms  = self.item_tfms,  # RandomCrop(500),
        )


VidSP2MOS.show_batch = show_vid_roi_batch
