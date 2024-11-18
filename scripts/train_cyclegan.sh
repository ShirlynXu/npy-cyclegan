set -ex
# python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout
python train.py --dataroot /mnt/data2/pytorch-CycleGAN-and-pix2pix/datasets/HN_cbct2ct --name HN_cbct2ct_cyclegan --model cycle_gan --pool_size 0 --no_dropout --input_nc 1 --output_nc 1 --dataset_mode unaligned --serial_batches
