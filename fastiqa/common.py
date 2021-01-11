import numpy as np
import torch
from torchvision import transforms

from PIL import Image

# render_output
import matplotlib.pyplot as plt

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


class Transform:
    def __init__(self):
        # normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

        self._train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self._val_transform = transforms.Compose([transforms.ToTensor()])

    @property
    def train_transform(self):
        return self._train_transform

    @property
    def val_transform(self):
        return self._val_transform



def set_up_seed(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def format_output(global_score, local_scores=None):
    if local_scores is None:
        return {"global_score": float(global_score)}
    else:
        return {"global_score": float(global_score), "local_scores": local_scores}


def blend_output(input_image, output, vmin=0, vmax=100, alpha=0.8, resample=Image.BILINEAR):
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
    qmap_matrix = output['local_scores']
    qmap_matrix = 100*stretch(np.array(qmap_matrix), vmin, vmax)
    qmap_matrix = (np.array(qmap_matrix) * 255 / 100).astype(np.uint8)
    colored_map = cm(qmap_matrix)
    # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
    # But we want to convert to RGB in uint8 and save it:
    heatmap = Image.fromarray((colored_map[:, :, :3] * 255).astype(np.uint8))
    sz = input_image.size
    heatmap = heatmap.resize(sz, resample=resample)

    return Image.blend(input_image, heatmap, alpha=alpha)

def render_output(input_image, output, vmin=0, vmax=100, fig=None):
    # QualityMap.plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 8 * 3))

    raw_img = axes[0].imshow(input_image)
    blend_img = axes[1].imshow(input_image, alpha=0.2)

    # _, H, W = input_image.shape # fastai
    W, H = input_image.size # PIL
    h, w = output['local_scores'].shape
    extent = (0, W // w * w, H // h * h, 0)

    blend_mat = axes[1].imshow(output['local_scores'], alpha=0.8, cmap='magma',
                   extent=extent, interpolation='bilinear')
    mat = axes[2].imshow(output['local_scores'], cmap='gray', extent=extent,
                   vmin=vmin, vmax=vmax)

    axes[0].set_title('Input image')
    score = axes[1].set_title(f"Predicted: {output['global_score']:.2f}")
    axes[2].set_title(f'Quality map {h}x{w}')
    return fig, [raw_img, blend_img, blend_mat, mat, score]
