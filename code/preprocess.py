import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
import voxelmorph as vxm
import neurite as ne

# Some useful functions first

def extract_slices(vol):
  mid_slices = [np.take(vol, vol.shape[d]//1.8, axis=d) for d in range(3)]
  mid_slices[1] = np.rot90(mid_slices[1], 1)
  mid_slices[2] = np.rot90(mid_slices[2], -1)
  return mid_slices

def Registration(moving_file, label_file, fixed_file, output_file):
  # for fixed_filename in sorted(os.listdir(fixed_file)):
  device, nb_devices = vxm.tf.utils.setup_device(0)
  with tf.device(device):
    for index in range(43,48):
      #fixed_filename = str(index) + '_T1.nii.gz'
      fixed_filename = str(index) + '_ANAT_N4_MNI_fcm.nii.gz'
      for moving_filename in sorted(os.listdir(moving_file)):
        print('moving image: ', moving_filename)
        print('fixed image: ', fixed_filename)
        label_filename = moving_filename[:2] + '_LABELS_MNI.nii.gz'

        moving_img = vxm.py.utils.load_volfile(moving_file + moving_filename)
        label_img = vxm.py.utils.load_volfile(label_file + label_filename)
        fixed_img, fixed_affine = vxm.py.utils.load_volfile(fixed_file + fixed_filename, ret_affine=True)


        pad_amount = ((7, 8), (1, 1), (7, 8))
        moving_img = np.pad(moving_img, pad_amount, 'constant')
        label_img = np.pad(label_img, pad_amount, 'constant')
        #pad_amount = ((32, 32), (32, 32), (32, 32))
        #pad_amount = ((32, 32), (31, 32), (32, 32))
        fixed_img = np.pad(fixed_img, pad_amount, 'constant')

        # print(moving_img.shape)
        # print(label_img.shape)
        # print(fixed_img.shape)
        # quit()

        val_input = [
            moving_img[np.newaxis, ..., np.newaxis],
            fixed_img[np.newaxis, ..., np.newaxis]
        ]

        vol_shape = fixed_img.shape
        nb_features = [
            [16, 32, 32, 32],
            [32, 32, 32, 32, 32, 16, 16]
        ]

        # build vxm network
        vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

        vxm_model.load_weights('./brain_3d.h5')

        val_pred = vxm_model.predict(val_input)

        moved_pred = val_pred[0].squeeze()
        pred_warp = val_pred[1]



        # titlefn = lambda x: ['%s %s' % (x, f) for f in ['saggital', 'axial', 'coronal']]

        # show moving, fixed, and moved volumes
        # ne.plot.slices(extract_slices(moving_img), titles=titlefn('moving'), cmaps=['gray'], width=10)
        # ne.plot.slices(extract_slices(fixed_img), titles=titlefn('fixed'), cmaps=['gray'], width=10)
        # ne.plot.slices(extract_slices(moved_pred), titles=titlefn('moved'), cmaps=['gray'], width=10)
        # print(pred_warp[0,:,:,:].shape)
        # ne.plot.slices(extract_slices(pred_warp[0,:,:,:]), titles=titlefn('moved'), cmaps=['gray'], width=10)

        warp_model = vxm.networks.Transform(vol_shape, interp_method='nearest')
        warped_seg = warp_model.predict([label_img[np.newaxis,...,np.newaxis], pred_warp])

        import matplotlib

        fs_colors = np.load('fs_rgb.npy')
        ccmap = matplotlib.colors.ListedColormap(fs_colors)

        mid_slices_moving = label_img.squeeze()[int(label_img.shape[0]//1.8), ...]
        mid_slices_moved = warped_seg.squeeze()[int(label_img.shape[0]//1.8), ...]

        slices = [mid_slices_moving, mid_slices_moved]

        titles = ['moving', 'moved']
        # the imshow arguments here are simply to maintain freesurfer colors
        # ne.plot.slices(slices, cmaps=[ccmap], imshow_args=[{'vmin':0, 'vmax':255}], titles=titles)


        # save moved image
        vxm.py.utils.save_volfile(moved_pred.squeeze(), output_file+'image/'+ fixed_filename[:2]+'_'+ moving_filename[:2]+ fixed_filename[3:], fixed_affine)
        # save moved label
        vxm.py.utils.save_volfile(warped_seg.squeeze(), output_file+'label/'+ fixed_filename[:2]+'_'+ moving_filename[:2]+ fixed_filename[3:], fixed_affine)

## Registration
#Registration(moving_file='/cis/home/tliu77/Documents/training/skull_stripping/', label_file='/cis/home/tliu77/Documents/training/delineated/manual/',
#             fixed_file='/cis/home/tliu77/Documents/training/skull_stripping_un/', output_file='/cis/home/tliu77/Documents/training/final_test/seg_eval/registration/')

Registration(moving_file='/cis/home/tliu77/Documents/training/skull_stripping/', label_file='/cis/home/tliu77/Documents/training/delineated/manual/',
             fixed_file='/cis/home/tliu77/Documents/training/final_test/seg_eval/skull_stripping/', output_file='/cis/home/tliu77/Documents/training/final_test/seg_eval/registration/')

## plot code
# titlefn = lambda x: ['%s %s' % (x, f) for f in ['saggital', 'axial', 'coronal']]

# moved_img = vxm.py.utils.load_volfile('/cis/home/tliu77/Documents/training/registration/image/001_03_T1.nii.gz')
# ne.plot.slices(extract_slices(moved_img), titles=titlefn('moved'), cmaps=['gray'], width=10)

# import matplotlib

# fs_colors = np.load('fs_rgb.npy')
# ccmap = matplotlib.colors.ListedColormap(fs_colors)
# label_img = vxm.py.utils.load_volfile('/cis/home/tliu77/Documents/training/registration/label/001_03_T1.nii.gz')
# mid_slices_moved = label_img.squeeze()[int(label_img.shape[0]//1.8), ...]

# slices = [mid_slices_moved]

# titles = ['moved']
# # the imshow arguments here are simply to maintain freesurfer colors
# ne.plot.slices(slices, cmaps=[ccmap], imshow_args=[{'vmin':0, 'vmax':255}], titles=titles)


## move file code
# import shutil
# source_dir = '/cis/home/tliu77/Documents/training/skull_stripping_un/'
# dest_dir = '/cis/home/tliu77/Documents/training/skull_stripping_un_mask/'
# for dirpath, dirnames, filenames in os.walk(source_dir):
#     for filename in filenames:
#         if 'mask' in filename:
#             shutil.move(dirpath + filename, dest_dir)
