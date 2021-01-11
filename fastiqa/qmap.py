import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .learn import IqaLearner

class QualityMap:
    """Note here the image is fastai image"""
    def __init__(self, mat, img, global_score=0):
        self.mat, self.img = mat, img
        self.global_score = global_score

    # fastai1
    # from fastai.vision import image2np
    # from PIL import Image as PIL_Image
    # @property
    # def pil_image(self):
    #     return PIL_Image.fromarray((255 * image2np(self.img.px)).astype(np.uint8))

    # @property
    def pil_image(self):
        # PIL.Image.frombytes
        # PIL.Image.new
        return PILImage.create((255 * (self.img.px)).astype(np.uint8))

    def plot(self, vmin=0, vmax=100):
        fig, axes = plt.subplots(1, 3, figsize=(12, 8 * 3))

        # fastai image
        self.img.show(axes[0], title='Input image')  # title mos
        self.img.show(axes[1], title=f'Predicted: {self.global_score:.2f}')  # title prediction

        _, H, W = self.img.shape
        h, w = self.mat.shape  # self.mat.size()
        extent = (0, W // w * w, H // h * h, 0)

        axes[1].imshow(self.mat, alpha=0.8, cmap='magma',
                       extent=extent, interpolation='bilinear')
        axes[2].imshow(self.mat, cmap='gray', extent=extent,
                       vmin=vmin, vmax=vmax)
        axes[2].set_title(f'Quality map {h}x{w}')

    def _repr_html_(self):
        self.plot()
        return ''

    def savefig(self, filename):
        plt.savefig(filename, bbox_inches='tight')

    def blend(self, mos_range=(0, 100), alpha=0.8, resample=Image.BILINEAR):
        """qmap.blend().save('qmap.jpg')"""

        def stretch(image, minimum, maximum):
            if maximum is None:
                maximum = image.max()
            if minimum is None:
                minimum = image.min()
            image = (image - minimum) / (maximum - minimum)
            image[image < 0] = 0
            image[image > 1] = 1
            return image

        cm = plt.get_cmap('magma')
        # min-max normalize the image, you can skip this step
        qmap_matrix = self.mat
        if mos_range is not None:
            qmap_matrix = 100*stretch(np.array(qmap_matrix), mos_range[0], mos_range[1])
        qmap_matrix = (np.array(qmap_matrix) * 255 / 100).astype(np.uint8)
        colored_map = cm(qmap_matrix)
        # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
        # But we want to convert to RGB in uint8 and save it:
        heatmap = PIL_Image.fromarray((colored_map[:, :, :3] * 255).astype(np.uint8))
        sz = self.img.shape[-1], self.img.shape[-2]
        heatmap = heatmap.resize(sz, resample=resample)

        return PIL_Image.blend(self.pil_image, heatmap, alpha=alpha)


def predict_quality_map_without_roi_pool(self, img, blk_size=None):
    if blk_size is None:
        blk_size = [4, 4]  # [32, 32]  # very poor

    batch_x = img2patches(img.data, blk_size).cuda()
    batch_y = torch.zeros(1, blk_size[0] * blk_size[1]).cuda()
    predicts = self.pred_batch(batch=[batch_x, batch_y])
    # img.shape 3 x H x W
    h = blk_size[0]  # + int(img.shape[1] % blk_size[0] != 0)
    w = blk_size[1]  # + int(img.shape[2] % blk_size[1] != 0)
    pred = predicts.reshape(h, w)
    return QualityMap(pred, img)

def get_quality_maps(self):
    # TODO predict in batch (according to data.batch_size)
    # get_qmaps
    """
    save all quality maps to quality maps folder
    =======================================
    im = open_image('/media/zq/Seagate/data/TestImages/Noisy Lady.png') #
    learn = e['Pool1RoIModel']
    learn.data = TestImages()
    t, a = learn.predict_quality_map(im, [[8,8]])
    print(a)
    print(t[0].data[0])  # image score
    # im
    plt.savefig('foo.png', bbox_inches='tight')
    :return:
    """
    # cannot do it in parallel (GPU)
    # only generate on valid set
    # must be data, inner_df only contain train samples and it's shuffled
    # must be loaded, since we need to call data.one_item

    # use learn.data.valid_ds with df.fn_col, no need
    # this will not load the database
    # only load the dataframe
    df = self.dls.df[self.dls.df.is_valid]  #.reset_index()
    #for file in df[self.dls.fn_col]
        # img = open_image(self.dls.path/self.dls.folder/file)

    # _, ax = plt.subplots()
    fig, (ax1, ax2) = plt.subplots(2)
    names = df[self.dls.fn_col].tolist()
    for n in range(len(self.dls.valid_ds)):  # add progress bar
        sample = self.dls.valid_ds[n][0]
        qmap = self.predict_quality_map(sample)
        qmap.show(ax=ax2)
        ax1.imshow(sample)
        dir = self.dls.path/'quality_maps'/self.dls.folder
        filename = str(dir/names[n]).split('.')[-2] + '.jpg'
        dir = filename.rsplit('/', 1)[0]  # 'data/CLIVE/quality_maps/Images/trainingImages/t1.jpg'
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(filename, bbox_inches='tight')

def predict_quality_map(self, sample, blk_size=None):
    """
    :param sample: fastai.vision.image.Image  open_image()
    :param blk_size:
    :return:
    if you simply wanna quality map matrix:
    sample = open_image('img.jpg')
    self.model.input_block_rois(blk_size, [sample.shape[-2], sample.shape[-1]])
    t = self.predict(sample)
    """

    if blk_size is None:
        blk_size = [[32, 32]]  #[[8, 8]]

    if not isinstance(blk_size[0], list):
        blk_size = [blk_size]
    # input_block_rois(self, blk_size=None, img_size=[1, 1], batch_size=1, include_image=True):
    # [8, 8] #  [5, 5]
    cuda = self.dls.device.type == 'cuda'
    self.model.input_block_rois(blk_size, [sample.shape[-2], sample.shape[-1]], cuda=cuda)  # self.dls.img_raw_size
    # [768, 1024] [8, 8] is too big learn.data.batch_size

    # predict will first convert image to a batch according to learn.data
    # TODO backup self.dls
    # this is very inefficient
    # data = self.dls
    # self.dls = TestImages()
    # if type(sample) == fastai.vision.core.PILImage:
    #     sample = image2tensor(sample)
    t = self.predict(sample)
    # self.dls = data

    a = t[0].data[1:].reshape(blk_size[0])  # TODO only allow one blk size

    # convert sample to PIL image
    # image = PIL_Image.fromarray((255 * image2np(sample.px)).astype(np.uint8))
    return QualityMap(a, sample, t[0].data[0])


IqaLearner.predict_quality_map_without_roi_pool = predict_quality_map_without_roi_pool
IqaLearner.predict_quality_map = predict_quality_map
IqaLearner.get_quality_maps = get_quality_maps
