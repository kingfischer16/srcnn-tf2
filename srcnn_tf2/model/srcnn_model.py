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
from ..data.preprocessing import scale_batch


class SRCNN:
    """
    SRCNN model class.
    """
    def __init__(self, num_channels=3, f1=9, f3=5, n1=64, n2=32, nlin_layers=1,
                 activation='relu', optimizer='adam', loss='mse', metrics=['accuracy']):
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
        self.make_model()
    
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
                          padding='same')(i)
        for j in range(self.nlin_layers):
            x = layers.Conv2D(filters=self.n2,
                              kernel_size=1,
                              activation=self.activation,
                              padding='same')(x)
        x = layers.Conv2D(filters=self.num_channels,
                          kernel_size=self.f3,
                          activation=self.activation,
                          padding='same')(x)
        self.model = keras.Model(i, x)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
    
    def fit(self, xdata, ydata, epochs, batch_size, validation_split=0.0):
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
        self.result = self.model.fit(x_scaled, ydata, epochs=epochs, batch_size=batch_size,
                                      validation_split=validation_split)
    
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
                                    (ximages.shape[1]*self.scale, ximages.shape[2]*self.scale))
        return self.model.predict(x_scaled_data)
    
    def plot_training(self, figsize=(12, 8), plot_vars=['accuracy']):
        """
        Plots the training
        """
        _, ax = plt.subplots(1, 1, figsize=figsize)
        for plot_var in plot_vars:
            ax.plot(self.result.history[plot_var], label=plot_var)
        plt.legend()
        plt.show()
        
