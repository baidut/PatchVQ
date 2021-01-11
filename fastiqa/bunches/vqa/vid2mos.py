__all__ = ['Vid2MOS', 'clip2image']

"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.vqa.vid2mos import *
# dls = Vid2MOS.from_json('json/LIVE_VQC.json', bs=3, clip_num=3, clip_size=2)
dls = Vid2MOS.from_json('json/KoNViD.json', bs=3, clip_num=3, clip_size=2)
# dls = Vid2MOS.from_json('json/LIVE_FB_VQA_pad500.json', item_tfms=CropPad(500), bs=3, clip_num=3, clip_size=2)
dls.show_batch()
dls.bs
'_data' in dls.__dict__
del dls.__dict__['_data']
dls.bs = 2
dir(dls)
dls._data.bs
dls = dls.reset(bs=2)
dls.show_batch()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

from . import *

"""
Note here, a video means a collection of jpg files
"""

def clip2image(t, vertical=False):
    # [n, 3, H, W] --> [3, n, H, W] --> [3, n*H, W]
    if vertical:# vertical concat
        return t.transpose(0, 1).reshape(3, -1, t.size(-1))
    else:
        # [n, 3, H, W] --> [3, H, n, W] --> [3, H, n*W]
        return t.transpose(0, 1).transpose(1, 2).reshape(3, t.size(-2), -1)

class Vid2MOS(IqaDataBunch):
    clip_size = 1
    clip_num = 8
    bs = 8
    fn_last_frame_col = "fn_last_frame"
    folder = "jpg"
    frame_num_col = 'frame_number'

    def get_df(self):
        df = super().get_df()
        # add_fn_last_frame col if not exists
        if self.frame_num_col not in df.columns:
            print('add frame_num_col')
            frame_numbers = []
            for folder in df[self.fn_col].tolist():
                n_max = 0
                # .split('.')[0]
                for file in (self.path/self.folder/folder).glob('*.jpg'):
                    n = int(str(file)[:-4].split('_')[-1])
                    if n > n_max:
                        n_max = n;
                frame_numbers.append(n_max)
            df[self.frame_num_col] = frame_numbers
            df.to_csv(self.path/self.csv_labels, index=False)
        if self.fn_last_frame_col not in df.columns:
            print('add fn_last_frame_col')
            df[self.fn_last_frame_col] = df[self.fn_col] + '/image_' + df[self.frame_num_col].astype(str).str.zfill(5)
            df.to_csv(self.path/self.csv_labels, index=False)
        return df

    def get_block(self):
        df = self.get_df()
        VideoBlock = partial(MultiClipBlock, clip_size=self.clip_size, clip_num=self.clip_num)
        return DataBlock(
            blocks     = (VideoBlock, RegressionBlock),
            getters = [
               ColReader(self.fn_last_frame_col, pref=self.path/(self.folder + '/')  ),
               ColReader(self.label_col), # mos_vid
            ],
            item_tfms = self.item_tfms,
            splitter   = self.get_splitter(),
        )

    def show_batch(self, b=None, max_n=4, figsize=(30, 6), vertical=False, **kwargs):
        # rb --> rois in a batch
        dls = self
        max_n = min(max_n, dls.bs)
        xb, yb = dls.one_batch() if b is None else b
        if vertical:
            fig, axes = plt.subplots(ncols=max_n, nrows=1, figsize=figsize, **kwargs) # , figsize=(12,6), dpi=120
        else:
            fig, axes = plt.subplots(nrows=max_n, ncols=1, figsize=figsize, **kwargs) # , figsize=(12,6), dpi=120
        if max_n == 1: axes = [axes] # only one item
        for i in range(max_n):
            xs, ys = xb[i], yb[i]
            # axes[i].imshow(x.permute(1,2,0).cpu().numpy())
            # axes[i].set_title(f'{ys.item():.02f}')
            timg = TensorImage(clip2image(xs).cpu())*255
            tpil = PILImage.create(timg)
            ctx = tpil.show(ax=axes[i])

            # ys 0 dim tensor
            lbl = str(ys.tolist()) # "%.2f" % ys if ys.dim() == 0 else
            axes[i].set_title(lbl)
            axes[i].axis('off')

    def create_batch(self, data):
        # cannot call self in this function
        return create_sequence_batch(data)
        # if self.clip_size > 1 else None

# Vid2MOS.show_batch = show_sequence_batch # old display method
