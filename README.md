# ARWGAN
This repository is the official PyTorch implementation of ARWGAN: attention-guided robust image watermarking model based on GAN.

## Pretrain
The pre-trained model of [ARWGAN](https://drive.google.com/file/d/1jDpF0LBmuFiy4PNvqaaz7vXyHCbHA4ao/view?usp=drive_link) is avaliable.

## Train
The environmental requirements:
+ Python == 3.7.4; Torch == 1.12.1 + cu102; PIL == 7.2.0  
      python mian.py new -n name -d data-dir -b batch-size -e epochs  -n noise
## Test
Put the pre-trained model into pretrain floder, and you can test ARWGAN by command line as following.

      python test.py -o ./pretrain/options-and-config.pickle -c ./pretrain/ARWGAN.pyt -s data-dir -n noise

## Acknowledgement
The codes are designed based on [HiDDeN](https://github.com/ando-khachatryan/HiDDeN).
