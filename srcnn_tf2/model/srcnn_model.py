"""
SRCNN Model - TensorFlow 2.0
============================

Code herein shall return a TensorFlow model, built using the keras API.
The base model is that of the SRCNN architecture found in [1]. Additional
code shall allow changes to be easily made to the basic model.

References
----------
 1. C. Dong, C. C. Loy, K. He, X. Tang, "Learning a deep convolutional network for image super-resolution," in ECCV, 2014.
"""

# Imports.
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, layers
from tensorflow import keras
from ..data.preprocessing import scale_batch, center_crop, gaussian_blur
from ..data.plotting import n_compare
from ..data.metrics import batch_psnr, batch_ssim
from time import time


class SRCNN:
    """
    SRCNN model class.
    """
    def __init__(self, num_channels=3, f1=9, f3=5, n1=64, n2=32, nlin_layers=1,
                 activation='relu', optimizer='adam', loss='mse', metrics=['accuracy'],
                 padding='valid', batch_norm=False):
        """
        Constructor.
        
        Args:
            num_channels (int): Number of channels in the image data.
            
            f1 (int): First filter dimension.
            
            f3 (int): Third filter dimension.
            
            n1 (int): Number of filters in the first layer.
            
            n2 (int): Number of filters in the second layer.
            
            nlin_layers (int): Number of middle (non-linear) layers.
            
            activation (str, function): Activation function.
            
            optimzer (str, function): Optimizer to use.
            
            loss (str, function): Loss function to use.
            
            metrics (list): List of metric functions to display.
            
            padding (str): One of 'valid' (default here and in SRCNN paper)
             or 'same'.
            
            batch_norm (bool): If True, applies batch normalization to the output
             of all but the last layer.
        """
        self.num_channels = num_channels
        self.f1 = f1
        self.f3 = f3
        self.n1 = n1
        self.n2 = n2
        self.nlin_layers = nlin_layers
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.padding = padding
        self.batch_norm = batch_norm
        
        self.make_model()
    
    def get_crop_size(self):
        """
        Calculates the size to which the output should be cropped. Used if
        padding is 'valid'
        
        """
        reduce_by = (self.f1 + self.f3 - 2) // 2
        return reduce_by
    
    def make_model(self):
        """
        Build and compile the model.
        """
        # Input layer.
        # SRCNN is a pre-upscaling model. This will happen during training and prediction.
        i = layers.Input(shape=(None, None, self.num_channels))

        # Convolutional layers.
        x = layers.Conv2D(filters=self.n1,
                          kernel_size=self.f1,
                          activation=self.activation,
                          padding=self.padding)(i)
        if self.batch_norm:
            x = layers.BatchNormalization()(x)
        for j in range(self.nlin_layers):
            x = layers.Conv2D(filters=self.n2,
                              kernel_size=1,
                              activation=self.activation,
                              padding=self.padding)(x)
            if self.batch_norm:
                x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=self.num_channels,
                          kernel_size=self.f3,
                          activation=self.activation,
                          padding=self.padding)(x)
        self.model = keras.Model(i, x)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
    
    def fit(self, xdata, ydata, epochs, batch_size, validation_split=0.0, verbose=1):
        """
        Fits the model.
        """
        scale_x = ydata.shape[1] / xdata.shape[1]
        scale_y = ydata.shape[2] / xdata.shape[2]
        if ((scale_x != scale_y) | (ydata.shape[1] % xdata.shape[1] != 0) | 
            (ydata.shape[2] % xdata.shape[2] != 0)):
            raise ValueError("Y-data must be scaled from the X-data by the same integer factor on both axes.")
        self.scale = int(scale_x)
        x_scaled = scale_batch(xdata, ydata.shape[1:3])
        start_time = time()
        self.result = self.model.fit(x_scaled,
                                     ydata if self.padding=='same' else center_crop(ydata, self.get_crop_size()),
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     validation_split=validation_split,
                                     verbose=verbose)
        exec_time = time() - start_time
        print(f"{epochs} epochs completed in {exec_time//60} minutes {exec_time%60:.2f} seconds, approx. {exec_time/epochs:.2f} seconds per epoch.")
    
    def summary(self):
        """
        Get the model summary.
        """
        print(self.model.summary())
    
    def predict(self, ximages):
        """
        Predict images.
        """
        if len(ximages.shape) != 4:
            raise ValueError(f"""Input images must be in 4-dimensional """
                             """vector including batch number, found: {ximages.shape}.""")
        x_scaled_data = scale_batch(ximages,
                                    (ximages.shape[2]*self.scale, ximages.shape[1]*self.scale))
        y_pred = self.model.predict(x_scaled_data)
        return np.array([yp for yp in y_pred])
    
    def plot_training(self, figsize=(12, 8), plot_vars=['accuracy']):
        """
        Plots the training
        """
        _, ax = plt.subplots(1, 1, figsize=figsize)
        for plot_var in plot_vars:
            ax.plot(self.result.history[plot_var], label=plot_var)
        plt.legend()
        plt.show()
        
    def benchmark(self, test_images, metric='psnr', return_metrics=False):
        """
        Benchmark the trained model against some test images. Test images
        can be different sizes. If model uses 'valid' padding, the metric
        will be calculated on a center-cropped version of the test images.
        
        Args:
            test_images (list): A list of images as numpy.arrays, may be of
             different shapes.
            
            metric (str): The metric to return. Must be one of: 'psnr', 'ssim'.
            
            return_metrics (bool): If True, the metrics are returned as a list
             and plotting is suppressed. Use this option to benchmark many models.
            
        Returns:
            (list, list): None by default, optionally the metrics as a list for
             both the model predictions and the bicubic upscaling.
        """
        # Modify the input data so it can be evenly divided by the model scaling factor.
        # Create x_test data for prediction.
        print("-------------------------------------------------------------------")
        print("Starting model benchmark...")
        print(f"\n\t1. Scaling test images to divide evenly by model scaling factor: {self.scale}")
        print("\n\t2. Downscaling and blurring test images for prediction input.")
        y_true = []
        x_test = []
        for img in test_images:
            # Modify y-image first.
            x_shape = (img.shape[0]//self.scale) * self.scale
            y_shape = (img.shape[1]//self.scale) * self.scale
            y_img = img[:x_shape, :y_shape, :]

            # Scale and blur x-image.
            out_shape = (y_img.shape[1]//self.scale, y_img.shape[0]//self.scale)
            x_scale = scale_batch(np.array([y_img]), out_shape)
            x_blur = gaussian_blur(x_scale, 1)[0]
            x_test.append(x_blur/x_blur.max())
            
            # Crop y-images if needed.
            if self.padding == 'valid':
                y_img_crop = center_crop([y_img], self.get_crop_size())[0]
                y_true.append(y_img_crop/y_img_crop.max())
            else:
                y_true.append(y_img/y_img.max())

        # Get predicted images.
        print("\n\t3. Predicting images using model.")
        y_pred = []
        for ximg in x_test:
            predicted_image = self.predict(np.array([ximg]))[0]
            y_pred.append(predicted_image/predicted_image.max())

        # Calculate metrics.
        print(f"\n\t4. Calculating metric: {metric.upper()}")
        if metric == 'psnr':
            m_suff = 'dB'
            metric_list = batch_psnr(y_pred, y_true)
        elif metric == 'ssim':
            m_suff = ''
            metric_list = batch_ssim(y_pred, y_true)
        else:
            raise ValueError(f"Value '{metric}' passed to argument 'metric' is not valid.")

        # Print results:
        print(f"\n\t5. {metric.upper()} results:")
        for i, m in enumerate(metric_list):
            print(f"\t\t5.{i+1}. {metric.upper()}: {m:.1f}")
        print(f"\tAverage {metric.upper()}: {np.mean(metric_list):.3f}")

        # Plot results.
        print(f"\n\t6. Plotting {metric.upper()} results:")
        scaled_metric_list = []
        for x_in, y_p, y_t, m in zip(x_test, y_pred, y_true, metric_list):
            if self.padding == 'valid':
                x_in = center_crop([x_in], self.get_crop_size()//self.scale)[0]
            x_scale = scale_batch(np.array([x_in]), (y_t.shape[1], y_t.shape[0]))[0]
            
            # Calculate the metric on the interpolated result for comparison.
            if metric == 'psnr':
                scaled_metric = batch_psnr([x_scale], [y_t])[0]
            elif metric == 'ssim':
                scaled_metric = batch_ssim([x_scale], [y_t])[0]
            else:
                raise ValueError(f"Value '{metric}' passed to argument 'metric' is not valid.")
            scaled_metric_list.append(scaled_metric)
            
            if not return_metrics:
                n_compare(
                    im_list=[x_in, x_scale, y_p, y_t],
                    label_list=[f'X Input - {x_in.shape[1]} x {x_in.shape[0]}',
                                f'X Scaled, {metric.upper()}: {scaled_metric:.1f} {m_suff} - {x_scale.shape[1]} x {x_scale.shape[0]}',
                                f'Y Predicted, {metric.upper()}: {m:.1f} {m_suff} - {y_p.shape[1]} x {y_p.shape[0]}',
                                f'Y True - {y_t.shape[1]} x {y_t.shape[0]}'],
                    figsize=(24,12),
                    zoom_box_coord=(x_in.shape[1]//2, x_in.shape[0]//2, x_in.shape[1]//4, x_in.shape[0]//4)
                )
        
        # Return metrics if desired.
        if return_metrics:
            return metric_list, scaled_metric_list
