"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)
sys.path.insert(0,os.path.join(currentdir,'data'))
from options.test_options import TestOptions
from models import create_model
from data_preparation import parse_sample
from systemsetup import DATA_PATH
import numpy as np
from base_dataset import get_ct_transform
import io_utils

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    pt_folders = []
    samples = io_utils.get_list('/home/lewes/ct_contouring/CycleGAN_2d/TCIA_Pelvic_test.CBCT.csv')    
    samples = io_utils.get_list('/home/lewes/ct_contouring/actraining/original_cohort/Hippocampus_L.TCIA-GLIS.MR.csv')        
    for sample in samples:
        pt,for_uid = parse_sample(sample)[0:2]
        pt_folders.append(os.path.join(DATA_PATH,pt,for_uid))
    # pt_folders = ['/home/lewes/data2/binary_data/1.2.246.352.63.3.5691628504634420590.7038749635106406285/1.2.246.352.63.3.5691628504634420590.7038749635106406285']
    # pt_folders = ['/home/lewes/data2/binary_data/5520220620140039/1.2.840.113654.2.382.123.5520220620140039']
    # pt_folders = ['/mnt/gcp/radtruth/2024092208233367/1.3.6.1.4.1.57553.555.20240922200619.524.10025/1.3.6.1.4.1.57553.555.20240922200619.524.10026']
    cbct_list = np.load('/mnt/data2/icon_cohort/cbct_set.npy',allow_pickle=True).item()
    pt_folders = [os.path.join('/mnt/data2/binary_data',tag) for tag in cbct_list]
    for pt_folder in pt_folders:
        cbct_file = os.path.join(pt_folder,'image.bin.gz')
        ori,sp,arr = io_utils.read_image_arr(cbct_file)
        # arr = alg_lib.brain_t1_patch_normalize3(arr) # 0~2000
        synthetic_ct_img = np.zeros(arr.shape)-1000
        for i in range(arr.shape[-1]):
            img_tensor = get_ct_transform(arr[...,i])
            data = {'A':img_tensor,'B':img_tensor,'A_paths':None,'B_paths':None}
            model.set_input(data)  # unpack data from data loader
            ct_slice = model.netG_A(model.real_A)
            slc = ct_slice.cpu().detach().numpy()
            synthetic_ct_img[...,i] = (slc+1)*1250 - 1000
        print(synthetic_ct_img.min(),synthetic_ct_img.max())
        ct_file = os.path.join(pt_folder,'CT.cyclegan.bin.gz')
        # ct_file = os.path.join('/mnt/data2','CT.synthetic.bin.gz')
        print(ct_file)
        io_utils.write_gzip(ct_file,synthetic_ct_img.astype(np.int16),ori,sp)
