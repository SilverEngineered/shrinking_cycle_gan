# Cycle Gan with Distillation
The code I have added is built on top of https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix, which is the official code release for the Cycle Gan.

The distillation, additional networks added, and modifications to support a teacher network are all my own work and constitute the majority of my contributions to this work.

## To Run:
Download the Dataset: bash ./datasets/download_cyclegan_dataset.sh monet2photo

Download the pretrained weights for the teacher model: bash ./scripts/download_cyclegan_model.sh monet2photo

 To Train: python train.py --dataroot ./datasets/monet2photo --name monet_ex --model cycle_gan
 
 To Test: python test.py --dataroot datasets/monet2photo --name monet_ex