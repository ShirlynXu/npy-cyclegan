set -ex
# python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
python train.py --dataroot /mnt/gcp/ct_contouring/actraining/pytorch-CycleGAN-and-pix2pix/datasets/HN_cbct2ct --name HN_cbct2ct_pix2pix --model pix2pix --netG unet_256 --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
