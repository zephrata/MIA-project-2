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


label_file = '../final_test/seg_eval/registration/label/'

for fixed_index in range(43, 48):
    label_img_set = []
    for moving_index in range(1, 41):
        label_filename = str(fixed_index).zfill(2) + '_' + str(moving_index).zfill(2) + 'ANAT_N4_MNI_fcm.nii.gz'
        print(label_filename)
        label_img_set.append(vxm.py.utils.load_volfile(label_file + label_filename))
    print('begin label fusion:', str(fixed_index).zfill(3))
    label_img_set = np.array(label_img_set)

    label_fusion_img = np.zeros(label_img_set.shape[1:])
    for i in range(label_img_set.shape[1]):
        for j in range(label_img_set.shape[2]):
            for k in range(label_img_set.shape[3]):
                label_fusion_img[i,j,k] = np.bincount(label_img_set[:,i,j,k].astype(int)).argmax()

    output_file = '../final_test/seg_eval/segmentation/'
    vxm.py.utils.save_volfile(label_fusion_img.squeeze(), output_file+ str(fixed_index).zfill(2) + '_ANAT_N4_MNI_fcm.nii.gz')

    # import matplotlib
    # fs_colors = np.load('fs_rgb.npy')
    # ccmap = matplotlib.colors.ListedColormap(fs_colors)

    # origin_img = vxm.py.utils.load_volfile(label_file + '001_01_T1.nii.gz')
    # mid_slices_moving = label_fusion_img.squeeze()[int(label_fusion_img.shape[0]//1.8), ...]
    # mid_slices_moved = origin_img.squeeze()[int(origin_img.shape[0]//1.8), ...]

    # slices = [mid_slices_moving, mid_slices_moved]

    # titles = ['moving', 'moved']
    # the imshow arguments here are simply to maintain freesurfer colors
    # ne.plot.slices(slices, cmaps=[ccmap], imshow_args=[{'vmin':0, 'vmax':255}], titles=titles)

    # output_file = '../labelfusion/'
    # vxm.py.utils.save_volfile(label_fusion_img.squeeze(), output_file+ str(fixed_index) + '_T1.nii.gz')
    # np.save(output_file+ '001_T1.npy', label_fusion_img)

# img = nib.load('../delineated/manual/'+ '01_LABELS_MNI.nii.gz')
# print(img.shape)
# OrthoSlicer3D(img.dataobj).show()

# img = nib.load('../labelfusion/'+ '001_T1.nii.gz')
# print(img.dataobj[32:-32, 32:-32, 32:-32].shape)
# OrthoSlicer3D(img.dataobj[32:-32, 32:-32, 14:-50]).show()
# print(img.shape)

# img = nib.load('../delineated/volumes/'+ '01_ANAT_N4_MNI_fcm.nii.gz')
# # print(img.shape)
# OrthoSlicer3D(img.dataobj).show()

# img = nib.load('../'+ '01_ANAT_N4_MNI_fcm.nii.gz')
# # print(img.shape)
# OrthoSlicer3D(img.dataobj).show()

# label_img_set = label_img_set.astype(int)
# print(label_img_set[:,100,100,100])
# print(np.bincount(label_img_set[:,100,100,100]).argmax())
# for i in label_img_set.shape[1]:
#     for j in label_img_set.shape[2]:
#         for k in label_img_set.shape[3]:
            
