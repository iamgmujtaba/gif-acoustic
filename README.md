<b> Client-driven animated GIF generation framework using an acoustic feature </b>

This repository contains the original implementation of the paper **[ Client-driven animated GIF generation framework using an acoustic feature] ( https://doi.org/10.1007/s11042-020-10236-6)**, published at MTAP 2021.

We present a novel methodology to generate animated GIFs images using the computational resources of a client device. The method analyzes an acoustic feature from the climax section of an audio file to estimate the timestamp corresponding to the maximum pitch. Further, it processes a small video segment to generate the GIF instead of processing the entire video. This makes the proposed method computationally efficient, unlike baseline approaches that use entire videos to create GIFs. 




# Jetson Configuration
Officially, Jetson devices do not support installation. MMAPI or GStreamer can be used. Please use the following guide to install FFmpeg.
[FFmpeg installation on Jetson TX2]: https://ghulammujtabakorai.medium.com/ffmpegs-installation-on-the-jetson-tx2-66b5a3f21d02

# Experiments
Note: Github does not support animated WebP formats. We have to convert WebP images to GIF to use in Github.

Example of GIFs generated using the proposed method

### YouTube
<img  alt="Maroon 5 Sugar" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Maroon_YouTube.gif" width="260" height="170">  <img  alt="Subeme" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Subeme_YouTube.gif" width="260" height="170">  <img  alt="Happier" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Happier_YouTube.gif" width="260" height="170">

### Baseline
<img  alt="Maroon 5 Sugar" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Maroon_baseline.gif" width="260" height="170">  <img  alt="Subeme" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Subeme_baseline.gif" width="260" height="170">  <img  alt="Happier" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Happier_baseline.gif" width="260" height="170">

### Proposed
<img  alt="Maroon 5 Sugar" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Maroon_proposed.gif" width="260" height="170">  <img  alt="Subeme" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Subeme_proposed.gif" width="260" height="170">  <img  alt="Happier" src="https://github.com/iamgmujtaba/gif-acoustic/blob/master/experiments/Happier_proposed.gif" width="260" height="170">
