# PatchVQ
Patch-VQ: ‘Patching Up’ the Video Quality Problem

***Please email yingzhenqiang at gmail dot com for any questions. Thank you!***

## Demo

Please follow [this](https://colab.research.google.com/drive/1612Y7tiAgiutmt6rXHCjjKgNdJXJ5Hvo) to test the Patch VQ model pretrained on the LSVQ database.
Please follow [this](test_PVQ_on_new_datasets.ipynb) to test our Patch VQ model on your database.

# Note

* Due to the [breaking changes](https://github.com/fastai/fastai/blob/master/CHANGELOG.md#2018) in `fastai 2.0.18 --> 2.1`, the code is incompatible with the latest fastai! Please make sure the following versions are installed: 
  ```
  fastai 2.0.18
  fastcore 1.2.5
  torch 1.6.0
  torchvision 0.7.0
  ```

​	a new version is still work-in-progress. See [here](https://github.com/baidut/fastiqa). 

* To reproduce the results in the paper, make sure to use the pretrained weights provided here: 
  * [RoIPoolModel-fit.10.bs.120.pth](https://github.com/baidut/PatchVQ/releases/download/v0.1/RoIPoolModel-fit.10.bs.120.pth) for PaQ2PiQ (don't use weights from other sources)
  *  [fastai-r3d18_K_200ep.pth](https://github.com/baidut/PatchVQ/releases/download/v0.1/fastai-r3d18_K_200ep.pth) for resnet 3d (don't use pytorch builtin `r3d_18`)

**Common Issues:**

* `NameError: name '_C' is not defined`  --> `pip3 install Cython`
* `RuntimeError: Could not infer dtype of PILImage` --> downgrade: pip install pillow==8.2
* `AttributeError: module 'PIL.Image' has no attribute 'Resampling'` --> check your fastai and fastcore version
* `ImportError: cannot import name 'default_generator' from 'torch._C' (unknown location)` --> `pip install torch==1.6.0 torchvision==0.7.0`

## Download LSVQ database

**Description**

No-reference (NR) perceptual video quality assessment (VQA) is a complex, unsolved, and important problem to social and streaming media applications. Efficient and accurate video quality predictors are needed to monitor and guide the processing of billions of shared, often imperfect, user-generated content (UGC). Unfortunately, current NR models are limited in their prediction capabilities on real-world, "in-the-wild" UGC video data. To advance progress on this problem, we created the largest (by far) subjective video quality dataset, containing 39, 000 real-world distorted videos and 117, 000 space-time localized video patches ("v-patches"), and 5.5M human perceptual quality annotations. Using this, we created two unique NR-VQA models: (a) a local-to-global region-based NR VQA architecture (called PVQ) that learns to predict global video quality and achieves state-of-the-art performance on 3 UGC datasets, and (b) a first-of-a-kind space-time video quality mapping engine (called PVQ Mapper) that helps localize and visualize perceptual distortions in space and time. We will make the new database and prediction models available immediately following the review process.

**Investigators**

* Zhenqiang Ying (<zqying@utexas.edu>) -- Graduate Student, Dept. of ECE, UT Austin
* Maniratnam Mandal (<mmandal@utexas.edu>) -- Graduate Student, Dept. of ECE, UT Austin
* Deepti Ghadiyaram (<deeptigp@fb.com>), Facebook Inc.
* Alan Bovik ([bovik@ece.utexas.edu](mailto:bovik@ece.utexas.edu)) -- Professor, Dept. of ECE, UT Austin

**Download**

**We are making the LSVQ Database available to the research community free of charge. If you use this database in your research, we kindly ask that you reference our papers listed below:**

> - Z. Ying, M. Mandal, D. Ghadiyaram and A.C. Bovik, "Patch-VQ: ‘Patching Up’ the Video Quality Problem," arXiv 2020.[[paper\]](https://arxiv.org/pdf/2011.13544.pdf)
> - Z. Ying, M. Mandal, D. Ghadiyaram and A.C. Bovik, "LIVE Large-Scale Social Video Quality (LSVQ) Database", Online:https://github.com/baidut/PatchVQ, 2020.


**Please fill [THIS FORM ](https://forms.gle/cdxArkHUuuqRCX4Z9) to download our database.**

1. follow '[download_from_internetarchive.ipynb](https://colab.research.google.com/drive/16C4cEe-DRxwMUnMS-PQzjzGPs2HfIX6a)' to download Internet archive videos
2. download YFCC videos from [Box](https://utexas.box.com/s/3x10cuh5m2r85gcjmatgagkpf2ekgqwo) (The password will be sent to your email after you submit the request form.)
3. download label files (coordinates and scores).
    * [labels_test_1080p.csv](https://github.com/baidut/PatchVQ/releases/download/v0.1/labels_test_1080p.csv) 1.05 MB
    * [labels_train_test.csv](https://github.com/baidut/PatchVQ/releases/download/v0.1/labels_train_test.csv) 10.8 MB (`is_test` column denotes if a video is in the train set or the test set )
4. [optional] follow [this](https://drive.google.com/file/d/16B1UvoIE2iCzSt7moXRwfBGpB_nRy5qE/view?usp=sharing) crop patches from videos


**Copyright Notice**

-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2020 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, modify, and distribute this database (the images, the results and the source files) and its documentation for any purpose, provided that the copyright notice in its entirety appear in all copies of this database, and the original source of this database, Laboratory for Image and Video Engineering (LIVE, [http://live.ece.utexas.edu ](http://live.ece.utexas.edu/)) at the University of Texas at Austin (UT Austin, [http://www.utexas.edu ](http://www.utexas.edu/)), is acknowledged in any publication that reports research using this database.

The following papers are to be cited in the bibliography whenever the database is used as:

> - Z. Ying, M. Mandal, D. Ghadiyaram and A.C. Bovik, "Patch-VQ: ‘Patching Up’ the Video Quality Problem," arXiv 2020.[[paper\]](https://arxiv.org/pdf/2011.13544.pdf)
> - Z. Ying, M. Mandal, D. Ghadiyaram and A.C. Bovik, "LIVE Large-Scale Social Video Quality (LSVQ) Database", Online:https://github.com/baidut/PatchVQ, 2020.

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------
