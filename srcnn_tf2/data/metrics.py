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
        psnr_list.append(tf.image.psnr(im_pred/im_pred.max(), im_true/im_true.max(), 1.0).numpy())
    return psnr_list
