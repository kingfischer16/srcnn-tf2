"""
PLOTTING.PY
===========

Functions for plotting and comparing super resolution results.
"""

# Imports.
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def subsample_image(input_image, zoom_box_coords):
    """
    Crops the input image to the coordinates described in the
    'zoom_box_coords' argument.

    Args:
        input_image (numpy.array): The input image.

        zoom_box_coords (tuple, list): Coordinates corresponding to the first
         (low-resolution) image. Coordinates are described and ordered as
         follows: (x, y, width in pixels, height in pixels), where 'x' and 'y'
         describe the top-left of the box.
         Default is None, which draws no box and shows no zoomed images in the
         row below.
    
    Returns:
        (numpy.array) The cropped image.

    Notes:
        Code adapted from:
            https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    """
    start_x, start_y = zoom_box_coords[0:2]
    return input_image[start_y:start_y+zoom_box_coords[3], start_x:start_x+zoom_box_coords[2]]


def n_compare(im_list, label_list, figsize=(12,8), zoom_box_coord=None,
                zoom_box_color='red'):
    """
    Function to compare multple images side by side with custom zoom box.
    
    Assumes that the following images are higher resolution versions of the
    first image upscaled by some scaling factor, and that the scaling factor
    is the same for all images. The scaling factor will be inferred from the
    first (low-resolution) and second (high-resolution).

    Args:
        im_list (list): A list of numpy arrays containing the image data. The
         list must contain a minimum of 2 images.

        label_list (list): A list of labels for the images. Must be the
         same length as 'im_list'.
        
        figsize (tuple): Figure width and height in inches. Preserved image
         aspect ratio, maps to smallest dimension.
        
        zoom_box_coord (tuple, list): Coordinates corresponding to the first
         (low-resolution) image. Coordinates are described and ordered as
         follows: (x, y, width in pixels, height in pixels), where 'x' and 'y'
         describe the top-left of the box.
         Default is None, which draws no box and shows no zoomed images in the
         row below.
        
        zoom_box_color (string): A matplotlib string for the box color,
         default color is 'red'.
    
    Returns:
        None. Images are displayed.
    
    Raises:
        ValueError: If length of 'im_list' and 'label_list' are not identical.
    """
    # Error checks.
    # =============
    # 1. Matching lengths of image and lebal lists.
    if len(im_list) != len(label_list):
        raise ValueError(f"""List length mismatch: {len(im_list)} != {len(label_list)}. """
                         """Length of arguments 'im_list' and 'label_list' must be the same.""")

    # 2. At least 2 images are present in the input list.
    if len(im_list) < 2:
        raise ValueError(f"Input list to argument 'im_list' must contain 2 or more items.")
    
    # Infer values for plotting.
    n_rows = 1 if zoom_box_coord is None else 2
    n_cols = len(im_list)
    scaling_factor_x = im_list[1].shape[1] / im_list[0].shape[1]
    scaling_factor_y = im_list[1].shape[0] / im_list[0].shape[0]
    
    # Arrange and display images.
    # ===========================
    _, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    for i, im_i in enumerate(im_list):
        if n_rows == 1:
            # 1. If only a single row (no zoom).
            ax[i].imshow(im_i)
            ax[i].set_title(label_list[i])
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
        else:
            # 2. If 2 rows are to be plotted (zoom is applied).
            ax[0][i].imshow(im_i)
            ax[0][i].set_title(label_list[i])
            ax[0][i].get_xaxis().set_visible(False)
            ax[0][i].get_yaxis().set_visible(False)

            # 2.1. Handle box coordinates and scaling thereof.
            if i == 0:
                xy = (zoom_box_coord[0], zoom_box_coord[1])
                rect = patches.Rectangle(xy,
                                         zoom_box_coord[2],
                                         zoom_box_coord[3],
                                         linewidth=1,edgecolor='red',facecolor='none')
                new_zoom_coords = (xy[0], xy[1], zoom_box_coord[2], zoom_box_coord[3])
            else:
                xy = (int(zoom_box_coord[0]*scaling_factor_x), int(zoom_box_coord[1]*scaling_factor_y))
                rect = patches.Rectangle(xy,
                                         int(zoom_box_coord[2]*scaling_factor_x),
                                         int(zoom_box_coord[3]*scaling_factor_y),
                                         linewidth=1,edgecolor='red',facecolor='none')
                new_zoom_coords = (xy[0],
                                   xy[1],
                                   int(zoom_box_coord[2]*scaling_factor_x),
                                   int(zoom_box_coord[3]*scaling_factor_y)
                                  )
            ax[0][i].add_patch(rect)
            
            # 2.2. Draw zoomed images.
            ax[1][i].imshow(subsample_image(im_i, (new_zoom_coords)))
            ax[1][i].get_xaxis().set_visible(False)
            ax[1][i].get_yaxis().set_visible(False)
    
    plt.tight_layout()
    plt.show()
