# MIA-project-2

This is the instruction for proejct II

## Task A: Multi-Atlas Brain Segmentation

### Skull Stripping

```bash
cd HD-BET/HD_BET
hd-bet -i INPUT_FOLDER -o OUTPUT_FOLDER
```

First we have to remove the skull in the MRI brain image. Here we use a pretrained U-Net image segmentation architecture [HD-BET](https://github.com/MIC-DKFZ/HD-BET) to remove the skull.

The general idea of U-Net is to continuously downsampling, and then upsampling back to add up the results of the same layer, and finally output the probability of the segmented image.
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/blob/main/pics/unet.png" width="700px">
</div>

The original result is 
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/blob/main/pics/original.png" width="700px">
</div>
The skull stripped result is 
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/blob/main/pics/skull_stripping.png" width="700px">
</div>

### Registration

```bash
cd code
python registration.py
```

After getting the skull stripped images from both label images (e.x. /delineated/volumes/01_ANAT_N4_MNI_fcm.nii.gz) and unlabeled images (e.x. /undelineated/volumes/001_T1.nii.gz), we have to do registration on each unlabeled image to the label images.
Here we use the package [voxelmorph](https://github.com/voxelmorph/voxelmorph) to find a transformation field and apply the transformation to the segementation label. After doing the registration, we can have segmentation label images for our unlabel images on each label images.

The reulst for the registration from 01_ANAT_N4_MNI_fcm.nii.gz to 001_T1.nii.gz:
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/blob/main/pics/registration_01_001.png" width="700px">
</div>

The reulst for the registration from 02_ANAT_N4_MNI_fcm.nii.gz to 001_T1.nii.gz:
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/blob/main/pics/registration_02_001.png" width="700px">
</div>

### Label Fusion
```bash
cd code
python labelfusion.py
```
After registration, we have got 40 labeled images for each unlabel images, we need find a way to combine the labels into a single segmentation map. Here we use the Majority voting method: the label of the piexls are the most frequency labels in that pixel values among 40 registrationed atlases.

The label fusion result is 
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/blob/main/pics/labelfusion_001.png" width="700px">
</div>

## Task B: Age Prediction
Running the training set spilted in a ratio of 9:1 with validation set.
```bash
cd code
python train.py
```

Running the whole training set and evaluate on validation set.
```bash
cd code
python trainall.py
```

First we want to use 3D-convolution neural network to predict age. However, since the gradient of the neural network will disappear as the depth increases (the derivative of F(x) will be multiplied in each layer, and the final number will be close to 0), this problem can be solved by adding a layer of shortcut, so that the value of the derivative is constantly greater than 1. The following figure shows the residual block of resnet:
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/blob/main/pics/resnet.png" width="300px">
</div>

Since it is a regression prediction problem, we use MSE (Mean squared error) as the loss function, Adam as the optimizer, I built resnet10, resnet18, resnet34. The final network structure is as follows.
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/blob/main/pics/resnet_para.png" width="500px">
</div>

The result with resnet 10 (the validation set is spilted over training set with a ratio of 9:1):
<div align=center>
<img src = "https://github.com/zephrata/MIA-project-2/blob/main/pics/loss_historyGPU_num_0-GPU_no_1-batch_size_8-epoch_40-pretrain_False-model_depth_10-lr_0.0001-flag_w_o_label.png" width="500px">
</div>

