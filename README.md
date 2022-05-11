# MIA-project-2

This is the instruction for proejct II

## Task A: Multi-Atlas Brain Segmentation

### Skull Stripping
First we have to remove the skull in the MRI brain image. Here we use a pretrained U-Net image segmentation architecture [HD-BET](https://github.com/MIC-DKFZ/HD-BET) to remove the skull.

The general idea of U-Net is to continuously downsampling, and then upsampling back to add up the results of the same layer, and finally output the probability of the segmented image.
<div align=center>
<img src = "https://github.com/TIGEERR/MIA-project-2/blob/main/pics/unet.png" width="700px">
</div>
</br>
The final result is 


### Registration

### Label Fusion
