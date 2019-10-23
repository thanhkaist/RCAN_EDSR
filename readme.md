PROJECT 1: EDSR and RCAN in the comparison
=============

#### Prerequisite
- Create python 3.6 env with conda\
`conda create -n RP1 python=3.6`
- Unstall dependencies\
`pip install -r requirement.txt`
- Download dataset and put to **data** folder. Link for download is below:
https://www.dropbox.com/s/wlslycal91sujdg/SR_data.zip?dl=0

#### Train
- Train model by the following command:\
`$./edsr_train.sh` \
`$./rcan_train.sh`

#### Test 
- Generate test image
`$./gen_eval_images.sh`
- Run matlab and use psnr_ssim_given_2_folder.m for calculating PSNR and SSIM

#### Reference

Enhanced Deep Residual Networks for Single Image Super-Resolution


