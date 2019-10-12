PROJECT 1: Implementation EDSR with pytorch from scratch
=============

#### Prerequisite
- Create python 3.6 env with conda\
`conda create -n RP1 python=3.6`
- Unstall dependencies\
`pip install -r requirement.txt`
- Download dataset at \

#### Train
- Train model by the following command:\
`python train.py`

#### Test 
- Test model with the following command:\
`python test.py`

#### Reference

Enhanced Deep Residual Networks for Single Image Super-Resolution

Why choose this paper:
- appear in CVPR 2017 workshop. Best paper award of the NTIRE2017 workshop, and the winners of the NTIRE2017 Challenge on Single Image Super-Resolution

- The standard method that consider one among the state of the art need to be compare with if you propose a new algorithm 


Code github: https://github.com/limbee/NTIRE2017\
Code github: pytorch version - https://github.com/thstkdgus35/EDSR-PyTorch

Target:
- reproduce the EDSR
- incorperated channel attention . 


EDSR:
Weremovethebatchnormalization layers from our network as Nah et al.[19] presented in their image deblurring work. Since batch normalization layers normalizethefeatures, theygetridofrangeﬂexibility from networks by normalizing the features, it is better to remove them
 GPU memory usage is also sufﬁciently reduced since the batch normalization layers consume the same amount of memory as the preceding convolutional layers


them using NVIDIA Titan X GPUs
It takes 8 days and 4 days to train EDSR and MDSR, respectively

Note that geometric self-ensemble is valid only for symmetric downsampling methods such as bicubic downsampling.

Using L1 loss.
300000 updates
residual block 16
EDSR scale 1 for filter size 64  

RCAN:

20 RCAB
10 Resblock
