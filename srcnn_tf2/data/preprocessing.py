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
from scipy.ndimage import rotate, gaussian_filter
from itertools import permutations


def center_crop(images, remove_edge):
    """
    Return a set of images cropped to the center pixels defined
    by 'patch_size'. This is required since the SRCNN algorithm
    described in the original paper uses no padding (to avoid
    border effects). This function is used to crop the "y_true"
    data before computing loss (i.e. just before passing into 
    the 'fit' method.)
    
    Args:
        images (numpy.array): The images to crop.
        
        remove_edge (int): The number of pixels to remove from
         the edge of the image.
        
    Returns:
        (numpy.array): The cropped image batch.
    """
    images_cropped = []
    for img in images:
        img_size = img.shape[:2]
        x_start = remove_edge
        x_end = img_size[0] - remove_edge
        y_start = remove_edge
        y_end = img_size[1] - remove_edge
        images_cropped.append(img[x_start:x_end, y_start:y_end, :])
    return np.array(images_cropped)


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


def create_training_patches(images, patch_size, patches_per_image=1, patch_stride=None):
    """
    Returns a batch of image patches, given a batch of images.

    Args:
        images (list, numpy.array): Batch of images.
        
        patch_size (tuple, list): The (width, height) of the patch to
         return.
        
        patches_per_image (int): Number of random patches to
         generate from each image in the input batch. Default is 1.
        
        patch_stride (int): Stride to use in strided patching. Default
         is None, which does not use strided patching. If integer is passed
         then strided patching will be used regardless of what is passed
         to 'patches_per_image'.
    
    Returns:
        (numpy.array): Batch of image patches.
    """
    image_patches = []
    for im in images:
        if patch_stride is None:
            for i in range(patches_per_image):
                image_patches.append(get_random_patch(im, patch_size))
        else:
            image_patches += list(get_stride_patches(im, patch_size, patch_stride, 2))
    return np.array(image_patches)


def get_stride_patches(image_as_array, patch_size, stride=14, min_stride=1):
    """
    Extracts a number of sub-images from the input image given the
    number of pixels per stride and the patch size. The maximum number of
    patches will be extracted that can fit into the image without violating
    the minimum stride.
    
    Args:
        image_as_array (numpy.array): The input image as an array,
         assumed to be 3-dimensional with the third dimension being
         color channels.
        
        patch_size (tuple, list): The (width, height) of the patch to
         return.
        
        stride (int): Number of pixels to move the patch size.
        
        min_stride (int): The minimum difference between the last patch
         location and the current one. Must be smaller than 'stride'.
        
    Returns:
        (numpy.array): Batch of image patches.
    """
    if stride <= min_stride:
        raise ValueError(f"'stride' must be greater than 'min_stride': {stride} is not greater than {min_stride}")
    
    img_size = image_as_array.shape[:2]
    images = []
    i_start_last, j_start_last = 0, 0
    xstride, ystride = stride, stride
    
    for i_start in range(0, img_size[0] - xstride + 1, xstride):
        if ((i_start - i_start_last) < min_stride) & (i_start > 0):
            continue
        if i_start+patch_size[0] >= img_size[0] + 1:
            if (i_start + patch_size[0]) - (img_size[0] + 1) < min_stride:
                i_start -= (i_start + patch_size[0]) - (img_size[0])
            else:
                continue
        for j_start in range(0, img_size[1] - ystride + 1, ystride):
            if ((j_start - j_start_last) < min_stride) & (j_start > 0):
                continue
            if j_start+patch_size[1] >= img_size[1] + 1:
                if (j_start + patch_size[1]) - (img_size[1] + 1) < min_stride:
                    j_start -= (j_start + patch_size[1]) - (img_size[1])
                else:
                    continue
            images.append(
                image_as_array[i_start:i_start+patch_size[0], j_start:j_start+patch_size[1], :]
            )
    
    return np.array(images)
        

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


def import_from_file(location, image_formats=['png']):
    """
    Imports all PNG images in a file location and returns
    them as a numpy.array.

    Args:
        location (str): File folder location with images. Searches
         all subfolders for images.
        
        image_formats (list): List of image format extensions to read
         into the dataset.
    
    Returns:
        (numpy.array): Batch of images as a numpy.array, scaled to [0,1].
    """
    image_data = []
    for (folder, subfolders, files) in os.walk(location):
        if len(files) > 0:
            for f in files:
                if any([f.lower().endswith('.'+ext)] for ext in image_formats):
                    image_data.append(imread(folder+'/'+f))
    return np.array(image_data) / 255.0


def create_xy_patches(location_or_images, scale, patch_size=(60,60),
                      patches_per_image=1, patch_stride=None, blur_kernel=None,
                      rotations=[0], swap_channels=False, image_formats=['png']):
    """
    Returns the x and y training data from file. Automatically
    extracts patches, and scales these to create the x (low-res)
    and y (high-res truth) datasets.
    
    Args:
        location_or_images (str, nuiimpy.array): Either a string indicating 
         the file folder location with images (searches all subfolders
         for images), or a numpy.array containing the images.
        
        scale (int): Scaling factor by which to reduce the iamges to
         form the x data. Must divide evenly into the dimensions passed
         to 'patch_size'.
        
        patch_size (tuple): Size of patches to take from each image. Value for
         'scale' must divide evenly into 'patch_size'.
        
        patches_per_image (int): Number of random patches to
         generate from each image in the input batch. Default is 1.
        
        patch_stride (int): Stride to use in strided patching. Default
         is None, which does not use strided patching. If integer is passed
         then strided patching will be used regardless of what is passed
         to 'patches_per_image'.
        
        blur_kernel (int): Applies a Gaussian blur of size 'blur_kernel' after
         downscaling. If negative, applies the Gaussian blur before scaling.
         Default is None, which applies no blur.
        
        rotations (list): A list of integers of rotations (in degrees) to
         perform on each, preferably multiples of 90, i.e. [0, 90, 180, 270].
         Default is just 0 degrees (unrotated).
        
        swap_channels (bool): If True, returns 6 images per image, one
         for every possible arrangement of the RGB channels in
         the image. Default is False, implementing no channel swapping.
        
        image_formats (list): List of image format extensions to read
         into the dataset. Unused if images are passed to this
         function as an array.
    
    Returns:
        (numpy.array, numpy.array): x and y training data.
    """
    # Check if 'scale' divides into 'patch_size' evenly.
    if (patch_size[0] % scale != 0) | (patch_size[1] % scale != 0):
        raise ValueError(f"""Value for 'scale' must divide evenly into 'patch_size'"""
                        f""". '{scale}' does not divide into '{patch_size}'.""")
    x_image_size = (int(patch_size[0]/scale), int(patch_size[1]/scale))
    if isinstance(location_or_images, str):
        y_data_raw = import_from_file(location_or_images, image_formats)
    else:
        y_data_raw = location_or_images
    
    # Implement rotations.
    y_data_rotated = []
    for y_img in y_data_raw:
        y_data_rotated += [rotate(y_img, r) for r in rotations]
    
    # Implement channel permutation swap.
    if swap_channels:
        y_data = []
        channel_combos = list(permutations([0,1,2], 3))
        for y_img in y_data_rotated:
            y_data += [y_img[:, :, p] for p in channel_combos]
    else:
        y_data = y_data_rotated
    
    y_data = np.array(y_data)
    
    # Get random or strided patches.
    if patch_stride is None:
        y_data = create_training_patches(y_data, patch_size, patches_per_image)
    else:
        y_data = create_training_patches(y_data, patch_size, patch_stride=patch_stride)
    # Blur is applied before scaling, as in paper, if desired.
    if blur_kernel is None:
        x_data = scale_batch(y_data, x_image_size)
    elif blur_kernel > 0:
        x_data = gaussian_blur(scale_batch(y_data, x_image_size), blur_kernel)
    elif blur_kernel < 0:
        x_data = scale_batch(gaussian_blur(y_data, -blur_kernel), x_image_size)
    return x_data, y_data


def create_xy_data(file_location, scale, target_size=(60,60),
                   rotations=[0], swap_channels=False, image_formats=['png']):
    """
    Returns the x and y training data from file. Automatically
    scales the images to uniform resolution, then scales these to create
    the x (low-res) and y (high-res truth) datasets.
    
    Args:
        file_location (str): File folder location with images. Searches
         all subfolders for images.
        
        scale (int): Scaling factor by which to reduce the iamges to
         form the x data. Must divide evenly into the dimensions passed
         to 'target_size'.
        
        target_size (tuple): Size of target image (y data). Value for
         'scale' must divide evenly into 'target_size'.
        
        rotations (list): A list of integers of rotations (in degrees) to
         perform on each, preferably multiples of 90, i.e. [0, 90, 180, 270].
         Default is just 0 degrees (unrotated).
        
        swap_channels (bool): If True, returns 6 images per image, one
         for every possible arrangement of the RGB channels in
         the image. Default is False, implementing no channel swapping.
        
        image_formats (list): List of image format extensions to read
         into the dataset.
    
    Returns:
        (numpy.array, numpy.array): x and y training data.
    """
    # Check if 'scale' divides into 'patch_size' evenly.
    if (target_size[0] % scale != 0) | (target_size[1] % scale != 0):
        raise ValueError(f"""Value for 'scale' must divide evenly into 'patch_size'"""
                        f""". '{scale}' does not divide into '{target_size}'.""")
    x_image_size = (int(target_size[0]/scale), int(target_size[1]/scale))
    y_data_raw = import_from_file(file_location, image_formats)
    y_data_raw = scale_batch(y_data_raw, target_size)
    
    y_data_rotated = []
    for y_img in y_data_raw:
        y_data_rotated += [rotate(y_img, r) for r in rotations]
    
    if swap_channels:
        y_data = []
        channel_combos = list(permutations([0,1,2], 3))
        for y_img in y_data_rotated:
            y_data += [y_img[:, :, p] for p in channel_combos]
    else:
        y_data = y_data_rotated
    y_data = np.array(y_data)
    
    x_data = scale_batch(y_data, x_image_size)
    return x_data, y_data


def gaussian_blur(images, kernel):
    """
    Returns the image batch with the gaussian blur applied.
    
    Args:
        images (list, numpy.array): Batch of input images.
        
        kernel (int): Standard deviation for Gaussian kernel.
         The standard deviations of the Gaussian filter are given
         for each axis as a sequence, or as a single number, in
         which case it is equal for all axes.
    
    Returns:
        (list, numpy.array): Batch of filtered images.
    """
    images_out = []
    for img in images:
        images_out.append(gaussian_filter(img, [kernel, kernel, 0]))
    return np.array(images_out)
