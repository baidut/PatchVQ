__all__ = ['Clip', 'ClipBlock', 'MultiClip', 'MultiClipBlock', 'show_sequence_batch', 'takespread', 'create_sequence_batch']

from fastai.vision.all import *
from math import ceil

# https://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
# [x for x in takespread([3,4,5], 5, 1)] # [[3], [4], [4], [5], [5]]
# list(x for x in takespread(list(range(192)),3,2))
def takespread(sequence, num, clip_size=1):
    sequence = list(sequence)
    length = float(len(sequence)-clip_size+1)
    for i in range(num):
        start = int(ceil(i * (length-1) / num))
        yield sequence[start:start+clip_size]

# show_sequence_batch, create_batch

"""fastuple will put the same kind of items together, should use something else
otherwise frame1 all together, then followed by frame2 in a batch
"""

# https://note.nkmk.me/en/python-pillow-concat-images/
def get_concat_h(frames):
    im = frames[0]
    column = len(frames)
    dst = Image.new('RGB', (im.width * column, im.height))
    for x in range(1, column):
        dst.paste(frames[x], (x * im.width, 0))
    return dst

def get_concat_v(frames):
    im = frames[0]
    row = len(frames)
    dst = Image.new('RGB', (im.width, im.height * row))
    for y in range(1, row):
        dst.paste(frames[y], (0, y * im.width))
    return dst


def int2float(o:TensorImage):
    return o.float().div_(255.)


class Clip(fastuple):
    @classmethod
    # first_frame
    def create(cls, fn_last_frame, clip_size=None): # one file name: id
        a = fn_last_frame.rsplit('_', 1)
        base = a[0]
        end_frame = int(a[1].rsplit('.', 1)[0]) # .jpg

        fns = [f'{base}_{n+1:05d}'+'.jpg' for n in range(end_frame-clip_size, end_frame)]
        return cls(tuple(PILImage.create(f) for f in fns))

    def show(self, **kwargs):
        return self[0].show(**kwargs) # show first frame


    @property
    def size(self):
        return self[0].size

# ClipBlock renamed to MultiClipBlock
def ClipBlock(clip_size=8):
    f = partial(Clip.create, clip_size=clip_size)
    return TransformBlock(type_tfms=f, batch_tfms=int2float)

################################################################################

# MultiClipFromLastFrame
class MultiClip(fastuple):
    @classmethod
    def create(cls, fn_last_frame, clip_num=None, clip_size=None): # one file name: id
        a = fn_last_frame.rsplit('_', 1)
        base = a[0]
        n_frames = int(a[1])

        if clip_size is not None and clip_num is None: # get all clips without interval or overlap
            # if last clip has less than clip_size frames?
            # drop last clip
            # sample per second? how to reflect frame rate? --- low frame rate
            raise NotImplementedError
            fns = [f'{base}_{n+1:05d}'+'.jpg' for n in range(n_frames)]
        else:
            fns = []
            for idx in takespread(range(n_frames), clip_num, clip_size):
                fns += [f'{base}_{n+1:05d}'+'.jpg' for n in idx]
        return cls(tuple(PILImage.create(f) for f in fns))

    @property
    def size(self):
        return self[0].size

# ClipBlock renamed to MultiClipBlock
def MultiClipBlock(clip_num=8, clip_size=8):
    f = partial(MultiClip.create, clip_num=clip_num, clip_size=clip_size)
    return TransformBlock(type_tfms=f, batch_tfms=int2float)

def create_sequence_batch(data):
    xs, ys = [], []
    for d in data:
        xs.append(d[0])
        ys.append(d[1])

    try:
        xs_ = torch.cat([TensorImage(torch.cat([im[None] for im in x], dim=0))[None] for x in xs], dim=0)
        ys_ = torch.cat([y[None] for y in ys], dim=0)
    except:
        print('Error@create_sequence_batch')
        for idx, x in enumerate(xs):
            print(idx, x.size()) # torch.Size([3, 500, 500])
        xs_ = torch.cat([TensorImage(torch.cat([im[None] for im in x], dim=0))[None] for x in xs], dim=0)
        print('xs_ is ok')
        print(xs_)
        raise

    return TensorImage(xs_), TensorCategory(ys_)


def show_sequence_batch(dls, max_n=4):
    xb, yb = dls.one_batch()
    max_n = min(dls.bs, max_n)
    ncols = len(xb[0])
    fig, axes = plt.subplots(ncols=ncols, nrows=max_n, figsize=(12,6), dpi=120)
    for i in range(max_n):
        xs, ys = xb[i], yb[i]
        for j, x in enumerate(xs):
            axes[i,j].imshow(x.permute(1,2,0).cpu().numpy())
            axes[i,j].set_title(f'{ys.item():.02f}')
            axes[i,j].axis('off')
