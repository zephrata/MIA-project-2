import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
import voxelmorph as vxm
import neurite as ne


def extract_slices(vol):
  mid_slices = [np.take(vol, vol.shape[d]//1.8, axis=d) for d in range(3)]
  mid_slices[0] = np.rot90(mid_slices[0], 1)
  mid_slices[1] = np.rot90(mid_slices[1], 1)
  mid_slices[2] = np.rot90(mid_slices[2], -1)
  return mid_slices

## skull stripping
moving_file = '../delineated/volumes/'
moving_filename = '01_ANAT_N4_MNI_fcm.nii.gz'
moving_img = vxm.py.utils.load_volfile(moving_file + moving_filename)

skull_file = '../skull_stripping/'
skull_filename = '01_ANAT_N4_MNI_fcm.nii.gz'
skull_img = moving_img = vxm.py.utils.load_volfile(skull_file + skull_filename)

# fixed_file = '../undelineated/volumes/'
# fixed_filename = '001_T1.nii.gz'
# fixed_img = vxm.py.utils.load_volfile(fixed_file + fixed_filename)

titlefn = lambda x: ['%s %s' % (x, f) for f in ['saggital', 'axial', 'coronal']]

# show moving, fixed, and moved volumes
# ne.plot.slices(extract_slices(moving_img), titles=titlefn('moving'), cmaps=['gray'], width=10)
# ne.plot.slices(extract_slices(skull_img), titles=titlefn('skull stripping'), cmaps=['gray'], width=10)
# ne.plot.slices(extract_slices(moved_pred), titles=titlefn('moved'), cmaps=['gray'], width=10)

import matplotlib
fs_colors = np.load('fs_rgb.npy')
ccmap = matplotlib.colors.ListedColormap(fs_colors)

## registration
# moving_file = '../delineated/manual/'
# moving_filename = '02_LABELS_MNI.nii.gz'
# moving_img = vxm.py.utils.load_volfile(moving_file + moving_filename)

# fixed_file = '../registration/label/'
# fixed_filename = '001_02_T1.nii.gz'
# fixed_img = vxm.py.utils.load_volfile(fixed_file + fixed_filename)

# mid_slices_moving = np.rot90(moving_img.squeeze()[int(moving_img.shape[0]//1.8), ...], 1)
# mid_slices_fixed = np.rot90(fixed_img.squeeze()[int(fixed_img.shape[0]//1.8), ...], 1)

# slices = [mid_slices_moving, mid_slices_fixed]

# titles = ['moving', 'fixed']
# # the imshow arguments here are simply to maintain freesurfer colors
# ne.plot.slices(slices, cmaps=[ccmap], imshow_args=[{'vmin':0, 'vmax':255}], titles=titles)

## label fusion
skull_file = '../labelfusion/'
skull_filename = '001_T1.nii.gz'
label_fusion_img = vxm.py.utils.load_volfile(skull_file + skull_filename)
mid_slices_label_fusion = np.rot90(label_fusion_img.squeeze()[int(label_fusion_img.shape[0]//1.8), ...], 1)

slices = [mid_slices_label_fusion]

titles = ['label fusion']
# the imshow arguments here are simply to maintain freesurfer colors
ne.plot.slices(slices, cmaps=[ccmap], imshow_args=[{'vmin':0, 'vmax':255}], titles=titles)