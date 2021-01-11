__all__ = ['ImRoI2MOS']

"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.iqa.im_roi2mos import *
dls = ImRoI2MOS.from_json('json/LIVE_FB_IQA.json', bs=3)
dls.show_batch()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

from . import *

def show_roi_batch(dls, b=None, max_n=4, figsize=(15, 5), **kwargs):
    # rb --> rois in a batch
    max_n = min(max_n, dls.bs)
    if b is None: b = dls.one_batch()
    xb, rb, yb = b
    fig, axes = plt.subplots(ncols=max_n, nrows=1, figsize=figsize, **kwargs) # , figsize=(12,6), dpi=120
    for i in range(max_n):
        xs, rs, ys = xb[i], rb[i], yb[i]
        # axes[i].imshow(x.permute(1,2,0).cpu().numpy())
        # axes[i].set_title(f'{ys.item():.02f}')
        rois = rs.cpu().numpy().reshape(-1,4).tolist()
        labels = ['image', 'p1', 'p2', 'p3']
        tbbox = LabeledBBox(TensorBBox(rois), labels)
        timg = TensorImage(xs.cpu())*255
        tpil = PILImage.create(timg)
        ctx = tpil.show(ax=axes[i])
        tbbox.show(ctx=ctx)
        axes[i].set_title(','.join("%.2f" % x for x in ys.tolist()) if ys.dim() > 0 else "%.2f" % ys)
        axes[i].axis('off')

class ImRoI2MOS(IqaDataBunch):
    roi_col = ["left_image", "top_image", "right_image", "bottom_image"]
    pad = None
    # if loss_func is None:
    # loss_func = getattr(dls.train_ds, 'loss_func', None)

    # pad the images
    def get_df(self):
        # prefix = ['top', 'left', 'bottom', 'right', 'height', 'width']
        df = super().get_df()
        """prepare roi label"""
        if any([x not in df.columns for x in self.roi_col]):
            if 'height' not in df.columns:
                df['height'] = self.height
            if 'width' not in df.columns:
                df['width'] = self.width

            print('add coordinate information to csv file')
            # unpack position
            # for p in ['p1', 'p2', 'p3']:
            #     df[p] = df[p].apply(literal_eval)
            #     columns = [x+'_'+p for x in ['left', 'right', 'top', 'bottom', 'start', 'end']]
            #     df[columns] = pd.DataFrame(df[p].to_list(), columns=columns)
            df['top_image'] = 0
            df['left_image'] = 0
            df['bottom_image'] = df['height']
            df['right_image'] = df['width']
            df['height_image'] = df['height']
            df['width_image'] = df['width']

            # shift if needed
            if self.pad:
                top_shift = (self.pad - df['height']) // 2
                left_shift = (self.pad - df['width']) // 2

                for s in self.label_suffixes: # by deafult, ""
                    df['top' + s] += top_shift
                    df['left' + s] += left_shift
                    df['bottom' + s] = df['top' + s] + df['height' + s]
                    df['right' + s] = df['left' + s] + df['width' + s]

            # df.to_csv(self.path/self.csv_labels, index=False)
        return df

    def get_block(self):
        getters = [
           ColReader(self.fn_col, pref=self.path/self.folder, suff=self.fn_suffix),
           ColReader(self.roi_col),
           ColReader(self.label_col),
        ]
        return DataBlock(blocks=(ImageBlock, RegressionBlock, RegressionBlock),
                        item_tfms = self.item_tfms,
                        getters=getters, n_inp=2, splitter = self.get_splitter())

ImRoI2MOS.show_batch = show_roi_batch
