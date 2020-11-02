# srcnn-tf2
*Implementation and experimentation of the SRCNN model in TensorFlow 2.0*

---

## Purpose
This repository contains an implementation of the SRCNN super resolution algorithm [1] implemented in TensorFlow 2.0 (v2.3.1). This library is used to construct, train, and evaluate the SRCNN model, as well as to explore variations on architecture and implementation.

## Model
### Model architecture
The base SRCNN model consists of upscaling using bicubic interpolation three main convolutional layers [1]:
 1. A layer for mapping low-resolution patches as feature vectors, consisting of *n<sub>1</sub>* x *c* (where *c* is the number of channels in the image) filters of size *f<sub>1</sub>* x *f<sub>1</sub>*.
 2. A non-linear mapping layer consisting of *n<sub>2</sub>* filters of size 1 x 1. This layer maps each low-resolution patch onto another high-resolution patch in the following layer. Can be more that one layer to increase non-linearity.
 3. A reconstruction layer consisting of *c* x *n<sub>2</sub>* filters of size *f<sub>3</sub>* x *f<sub>3</sub>*, where the layers are aggregated to product the final image.

Typical values for the parameters are:
 * *n<sub>1</sub>*: 64
 * *n<sub>2</sub>*: 32
 * *f<sub>1</sub>*: 9
 * *f<sub>3</sub>*: 5

### Loss function
The authors of [1] use the **mean squared error, MSE**, as a loss function and note that this favors a high PSNR measurement on performance testing. While PSNR corresponds to closely matching pixel values, it does not necessarily help create images with high perceptual image quality (e.g. satisfying to human vision).

Gains in perceptual quality can be made by using content or texture loss, or a weighted combination of these. This will be added in the future.

## Data
This work beings by using the same training and benchmarking data used in the original paper.

### Datasets
 * **T91**
    - Training data
    - 91 images of varying resolution featuring plants, animals, humans, vehicles, and buildings
    - Source: https://www.kaggle.com/ll01dm/t91-image-dataset
* **Set5**
    - Evaluation data for scaling factors of 2, 3 and 4
    - 5 images (baby, bird, butterfly, head, woman), varying resolution
    - Source: https://www.kaggle.com/ll01dm/set-5-14-super-resolution-dataset
* **Set14**
    - Evaluation data for scaling factor 3
    - 14 images, varying resolution, humans, animals, plants
    - Source: https://www.kaggle.com/ll01dm/set-5-14-super-resolution-dataset
* **CIFAR-10**
    - 60,000 images, 32 x 32 pixels
    - 10 image classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - Used to evaulate image quality by human judgement. The images is small and there is no high-resolution version to compare, so this is just for "eyeballing" on fresh images.

### Preprocessing
Training is done only on the T91 datset, but these images require preprocessing.
#### Image patch generation
We will train on a series of 32 x 32 image patches pulled from the T91 dataset by stepping the patch window across each image by a stride of 14 pixels, which will give roughly 90,000 images. These images are downscaled and them blurred using a Gaussian kernel. Upscaling will happen in the model class.
#### Data augmentation
**Rotation** - Each 32 x 32 pixel image is rotated by 90 degrees to give 4 images per input image.

**Channel swap** - Swapping the red, green, and blue channels of each image, creating 6 images per input image.

## Performance
We will use the following to gauge the performance of upscaling:
 * **PSNR** - Peak signal-to-noise ratio, https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
 * **SSIM** - Structural similarity index ratio, https://en.wikipedia.org/wiki/Structural_similarity

## Model exploration
After creating and benchmarking the basic SRCNN model described in [1] we will explore the effect of changing several factors in an attempt to maximize the effectiveness of the model. The following will be altered:
 * Number of filters in layers 1 and 3
 * Size of filters in layers 1 and 3
 * Number of non-linear layers
 * Loss functions: content/texture/hyrbid loss
 * Batch normalization and dropout
 * Ensembling: multiple trained SRCNN models with fused prediction output
 * Self-ensembling: Predict on separate 90 degree rotations of the same input image and combine the predictions

## Attributions
### References
 1. C. Dong, C. C. Loy, K. He, X. Tang, "Learning a deep convolutional network for image super-resolution," in ECCV, 2014.

### Github projects
This repository is built on the work of the following projects:
 * https://github.com/tegg89/SRCNN-Tensorflow
 * https://github.com/liliumao/Tensorflow-srcnn
