"""
METRICS.PY
==========

Functions for calculatings super-resolution metrics.
"""

# Imports.
import numpy as np
import tensorflow as tf


def batch_psnr(val_images_pred, val_images_true):
    """
    Given the validation images, calculates the PSNR for each image.
    
    Args:
        val_images_pred (list, numpy.array): Validation images, predicted
         by the model.
        
        val_images_true (list, numpy.array): Validation images, true
         high-resolution images.
        
    Returns:
        (list): A list of the PSNR values and floats.
    """
    psnr_list = []
    for im_pred, im_true in zip(val_images_pred, val_images_true):
        psnr_i = tf.image.psnr(im_pred/im_pred.max(), im_true/im_true.max(), 1.0).numpy()
        psnr_list.append(round(psnr_i, 3))
    return psnr_list


def batch_ssim(val_images_pred, val_images_true):
    """
    Given the validation images, calculates the SSIM for each image.
    
    Args:
        val_images_pred (list, numpy.array): Validation images, predicted
         by the model.
        
        val_images_true (list, numpy.array): Validation images, true
         high-resolution images.
        
    Returns:
        (list): A list of the PSNR values and floats.
    """
    ssim_list = []
    for im_pred, im_true in zip(val_images_pred, val_images_true):
        ssim_i = tf.image.ssim(np.float32(im_pred/im_pred.max()), np.float32(im_true/im_true.max()), 1.0).numpy()
        ssim_list.append(round(ssim_i, 3))
    return ssim_list
