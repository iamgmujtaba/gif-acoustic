<b> Client-driven animated GIF generation framework using an acoustic feature </b>

This repository contains the original implementation of the paper **[Client-driven animated GIF generation framework using an acoustic feature](https://doi.org/10.1007/s11042-020-10236-6)**, published at MTAP 2021.

We present a novel methodology to generate animated GIFs images using the computational resources of a client device. The method analyzes an acoustic feature from the climax section of an audio file to estimate the timestamp corresponding to the maximum pitch. Further, it processes a small video segment to generate the GIF instead of processing the entire video. This makes the proposed method computationally efficient, unlike baseline approaches that use entire videos to create GIFs. 


## Prerequisite
- Linux or macOS
- Python 3.6
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/iamgmujtaba/gif-acoustic
cd gif-acoustic
```
- Install [TensorFlow](https://www.tensorflow.org/) and Keras and other dependencies
  - For pip users, please type the command `pip install -r requirements.txt`.


### GTZAN dataset train
- Download the GTZAN dataset [here](http://opihi.cs.uvic.ca/sound/genres.tar.gz)
- Extract the downloaded file in the data folder and the structure should look like this:

```bash
├── data/
   ├── gtzan
      ├── blues
      ├── classical
      .
      .
      .
      ├── rock
```

- Run python train code to train the GTZAN dataset
```bash
python train.py
```


## Experiments
Note: Github does not support animated WebP formats. We have to convert WebP images to GIF to use in Github.

Example of GIFs generated using the proposed method

### YouTube
<img  alt="Maroon 5 Sugar" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Maroon_YouTube.gif" width="260" height="170">  <img  alt="Subeme" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Subeme_YouTube.gif" width="260" height="170">  <img  alt="Happier" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Happier_YouTube.gif" width="260" height="170">

### Baseline
<img  alt="Maroon 5 Sugar" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Maroon_baseline.gif" width="260" height="170">  <img  alt="Subeme" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Subeme_baseline.gif" width="260" height="170">  <img  alt="Happier" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Happier_baseline.gif" width="260" height="170">

### Proposed
<img  alt="Maroon 5 Sugar" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Maroon_proposed.gif" width="260" height="170">  <img  alt="Subeme" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Subeme_proposed.gif" width="260" height="170">  <img  alt="Happier" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Happier_proposed.gif" width="260" height="170">


## Jetson Configuration
Officially, Jetson devices do not support installation. MMAPI or GStreamer can be used. Please use the following guide to install FFmpeg.
[FFmpeg installation on Jetson TX2](https://ghulammujtabakorai.medium.com/ffmpegs-installation-on-the-jetson-tx2-66b5a3f21d02)


## Citation
If you use this code for your research, please cite our paper.
```
@article{mujtaba2021,
  title={Client-driven animated GIF generation framework using an acoustic feature},
  author={Mujtaba, Ghulam and Lee, Sangsoon and Kim, Jaehyoun and Ryu, Eun-Seok},
  journal={Multimedia Tools and Applications},
  year={2021},
  publisher={Springer}}
'''

## License
Copyright (c) 2021, Ghulam Mujtaba. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
