from pathlib import Path

import torch
import numpy as np
# from PIL.Image import Image
from PIL import Image
from torchvision.datasets.folder import default_loader

from .common import Transform, format_output, render_output

import cv2

"""
#######################
# %% show quality map
#######################
%matplotlib inline
from paq2piq.inference_model import *;

file = '/media/zq/Seagate/Git/fastiqa/images/Picture1.jpg'
model = InferenceModel(RoIPoolModel(), 'paq2piq/RoIPoolModel.pth')
image = Image.open(file)
output = model.predict_from_pil_image(image)
render_output(image, output)
# %%

################################
# %% show quality map of a video
################################

# %%

"""

# use cuda if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class InferenceModel:
    blk_size = 20, 20
    categories = 'Bad', 'Poor', 'Fair', 'Good', 'Excellent'

    def __init__(self, model, path_to_model_state: Path):
        self.transform = Transform().val_transform
        model_state = torch.load(path_to_model_state, map_location=lambda storage, loc: storage)
        self.model = model
        self.model.load_state_dict(model_state["model"])
        self.model = self.model.to(device)
        self.model.eval()

    def predict_from_file(self, image_path: Path, render=False):
        image = default_loader(image_path)
        return self.predict(image)

    def predict_from_pil_image(self, image: Image):
        image = image.convert("RGB")
        return self.predict(image)

    def predict_from_cv2_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.predict(image)

    def predict_from_vid_file(self, vid_path: Path):
        im = Image.open(vid_path)
        index = 1
        for frame in ImageSequence.Iterator(im):
            frame.save("frame%d.png" % index)
            index = index + 1
        pass

    @torch.no_grad()
    def predict(self, image):
        image = self.transform(image)
        image = image.unsqueeze_(0)
        image = image.to(device)
        self.model.input_block_rois(self.blk_size, [image.shape[-2], image.shape[-1]], device=device)
        t = self.model(image).data.cpu().numpy()[0]

        local_scores = np.reshape(t[1:], self.blk_size)
        global_score = t[0]

        # normalize the global score
        x_mean, std_left, std_right = 72.59696108881171, 7.798274017370107, 4.118047289170692
        if global_score < x_mean:
            x = x_mean + x_mean*(global_score-x_mean)/(4*std_left)
            if x < 0: x = 0
        elif global_score > x_mean:
            x = x_mean + (100-x_mean)*(global_score-x_mean)/(4*std_right)
            if x > 100: x = 100
        else:
            x = x_mean
        category = self.categories[int(x//20)]
        return {"global_score": global_score,
                "normalized_global_score": x,
                "local_scores": local_scores,
                "category": category}
