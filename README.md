# MIA-project-2

This is the instruction for proejct II

## Task A: Multi-Atlas Brain Segmentation

### Skull Stripping
First we have to remove the skull in the MRI brain image. Here we use a pretrained U-Net image segmentation architecture [HD-BET](https://github.com/MIC-DKFZ/HD-BET) to remove the skull.

The general idea of U-Net is to continuously downsampling, and then upsampling back to add up the results of the same layer, and finally output the probability of the segmented image.
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/tree/main/pics/unet.png" width="700px">
</div>

The original result is 
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/tree/main/pics/original.png" width="700px">
</div>
The skull stripped result is 
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/tree/main/pics/skull_stripping.png" width="700px">
</div>

### Registration
After getting the skull stripped images from both label images (e.x. /delineated/volumes/01_ANAT_N4_MNI_fcm.nii.gz) and unlabeled images (e.x. /undelineated/volumes/001_T1.nii.gz), we have to do registration on each unlabeled image to the label images.
Here we use the package [voxelmorph](https://github.com/voxelmorph/voxelmorph) to find a transformation field and apply the transformation to the segementation label. After doing the registration, we can have segmentation label images for our unlabel images on each label images.

The reulst for the registration from 01_ANAT_N4_MNI_fcm.nii.gz to 001_T1.nii.gz:
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/tree/main/pics/registration_01_001.png" width="700px">
</div>

The reulst for the registration from 02_ANAT_N4_MNI_fcm.nii.gz to 001_T1.nii.gz:
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/tree/main/pics/registration_02_001.png" width="700px">
</div>

### Label Fusion
After registration, we have got 40 labeled images for each unlabel images, we need find a way to combine the labels into a single segmentation map. Here we use the Majority voting method: the label of the piexls are the most frequency labels in that pixel values among 40 registrationed atlases.

The label fusion result is 
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/tree/main/pics/labelfusion_001.png" width="700px">
</div>

## Task B: Age Prediction
First we want to use 3D-convolution neural network to predict age. However, since the gradient of the neural network will disappear as the depth increases (the derivative of F(x) will be multiplied in each layer, and the final number will be close to 0), this problem can be solved by adding a layer of shortcut, so that the value of the derivative is constantly greater than 1. The following figure shows the residual block of resnet:
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/tree/main/picsresnet.png" width="700px">
</div>


