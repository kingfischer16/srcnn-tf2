"""
PREPROCESSING.PY
================

Functions dedicated to importing and processing image data
for training and testing.
"""

# Imports.
import os
import numpy as np
import glob
from imageio import imread
from PIL import Image


def get_random_patch(image_as_array, patch_size):
    """
    Returns a random patch of the image of size 'patch_size'.
    
    Args:
        image_as_array (numpy_array): The input image as an array,
         assumed to be 3-dimensional with the third dimension being
         color channels.
        
        patch_size (tuple, list): The (width, height) of the patch to
         return.
    
    Returns:
        (numpy.array): The patch of the image.
    """
    x_start = np.random.randint(low=0,
                                high=image_as_array.shape[1] - patch_size[0] + 1)
    y_start = np.random.randint(low=0,
                                high=image_as_array.shape[0] - patch_size[1] + 1)
    return image_as_array[y_start:y_start+patch_size[1], x_start:x_start+patch_size[0], :]


def create_training_patches(images, patch_size, patches_per_image=1):
    """
    Returns a batch of image patches, given a batch of images.

    Args:
        images (list, numpy.array): Batch of images.
        
        patch_size (tuple, list): The (width, height) of the patch to
         return.
        
        patches_per_image (int): Number of random patches to
         generate from each image in the input batch. Default is 1.
    
    Returns:
        (numpy.array): Batch of image patches.
    """
    image_patches = []
    for im in images:
        for i in range(patches_per_image):
            image_patches.append(get_random_patch(im, patch_size))
    return np.array(image_patches)


def scale_batch(images, output_image_size):
    """
    Scales and returns a batch of images.

    Args:
        images (list, numpy.array): Batch of input images.

        output_image_size (list, tuple): The size of the output
         image.
        
    Returns:
        (numpy.array): Batch of scaled images.
    """
    scaled_images = []
    for im in images:
        pil_image = Image\
                    .fromarray(np.uint8(im*255))\
                    .resize(size=output_image_size, resample=Image.BICUBIC)
        s_image = np.array(pil_image)
        scaled_images.append(s_image)
    return np.array(scaled_images) / 255.0


def create_xy_data(images, scale):
    """

    """
    return None


def import_from_file(location):
    """
    Imports all PNG images in a file location and returns
    them as a numpy.array.

    Args:
        location (str): File folder location with images.
    
    Returns:
        (numpy.array): Batch of images as a numpy.array, scaled to [0,1].
    """
    image_data = []
    for image in glob.glob(location + '*.png'):
        im_array = imread(image)
        image_data.append(im_array)
    return np.array(image_data) / 255.0
