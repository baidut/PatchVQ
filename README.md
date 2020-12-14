# PatchVQ
Patch-VQ: ‘Patching Up’ the Video Quality Problem

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

> - Z. Ying, M. Mandal, D. Ghadiyaram and A.C. Bovik, "Study of 3D Virtual Reality Picture Quality," arXiv 2020.[[paper\]](https://arxiv.org/pdf/2011.13544.pdf)
> - Z. Ying, M. Mandal, D. Ghadiyaram and A.C. Bovik, "LIVE Large-Scale Social Video Quality (LSVQ) Database", Online:https://github.com/baidut/PatchVQ, 2020.

**Please fill [THIS FORM ](https://forms.gle/kmRH2fCuVuLAfruq5) to download our database.**

1. follow '[download_from_internetarchive.ipynb](https://colab.research.google.com/drive/1k_VUBnpZrio2lC-Uxec-WODP8rya-4mV)' to download Internet archive videos
2. download yfcc videos
  * from [Box](https://utexas.box.com/s/3x10cuh5m2r85gcjmatgagkpf2ekgqwo) [Recommended]
  * OR  from [google drive](https://drive.google.com/drive/folders/17HE3KB9hlDDc8H3DTX60MggHinTSZ7Rk)

**Copyright Notice**

-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2020 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, modify, and distribute this database (the images, the results and the source files) and its documentation for any purpose, provided that the copyright notice in its entirety appear in all copies of this database, and the original source of this database, Laboratory for Image and Video Engineering (LIVE, [http://live.ece.utexas.edu ](http://live.ece.utexas.edu/)) at the University of Texas at Austin (UT Austin, [http://www.utexas.edu ](http://www.utexas.edu/)), is acknowledged in any publication that reports research using this database.

The following papers are to be cited in the bibliography whenever the database is used as:

> - Z. Ying, M. Mandal, D. Ghadiyaram and A.C. Bovik, "Study of 3D Virtual Reality Picture Quality," arXiv 2020.[[paper\]](https://arxiv.org/pdf/2011.13544.pdf)
> - Z. Ying, M. Mandal, D. Ghadiyaram and A.C. Bovik, "LIVE Large-Scale Social Video Quality (LSVQ) Database", Online:https://github.com/baidut/PatchVQ, 2020.

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------