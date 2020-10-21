# srcnn-tf2
*Implementation and experimentation of the SRCNN model in TensorFlow 2.0*

---

## Purpose
This repository contains an implementation of the SRCNN super resolution algorithm [1] implemented in TensorFlow 2.0 (v2.3.1).

## Model
### Model architecture

### Loss function

## Data
This work beings by using the same training and benchmarking data used in the original paper.
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

## Attributions
### References
 1. C. Dong, C. C. Loy, K. He, X. Tang, "Learning a deep convolutional network for image super-resolution," in ECCV, 2014.

### Github projects
This repository is built on the work of the following projects:
 * https://github.com/tegg89/SRCNN-Tensorflow
 * https://github.com/liliumao/Tensorflow-srcnn
