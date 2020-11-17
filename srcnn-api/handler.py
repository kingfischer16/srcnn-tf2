"""
The handler function for using pretrained models to upscale
images. The predictor class is included in the handler file
for convenience (there's not much to it) and makes use of
the 'tflite_runtime' for predicting from a saved model.
"""
# Imports.
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image


#
class SRCNNDeploy:
    """
    Model class for deploying a pre-trained SRCNN model e.g. to AWS Lambda API.
    
    This class is built around the expectation that it will have to upscale
    one image at a time.
    """
    def __init__(self, model_location, scaling_factor, edge_pad):
        """
        Constructor.
        
        Args:
            model_location (str): The file location and model name to load.
            
            scaling_factor (int): The scaling factor of the model must be known
             in order to apply pre-upscaling. This is not discerable from the model
             but should be contained in the model name.
            
            edge_pad (int): How many pixels around the edge of the image to zero-pad.
             Required because the model uses a 'valid' padding for training to avoid
             training in edge effects, to images to enhance must have zero padding
             applied after upscaling.
        """
        # Setup model interpreter and parameters.
        self.interpreter = tflite.Interpreter(model_path=model_location)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.scaling_factor = scaling_factor
        self.edge_pad = edge_pad
    
    def _load_image(self):
        """
        Loads the image to scale. Image is read in as a Pillow Image class.
        """
        self.image = Image.open(self.image_path)
    
    def _bicubic_upscale(self):
        """
        Performs the pre-upscaling using bicubic interpolation on the input image.
        """
        output_image_size = (int(self.image.size[0]*self.scaling_factor), int(self.image.size[1]*self.scaling_factor))
        self.upscaled_image = self.image.resize(size=output_image_size, resample=Image.BICUBIC)
        self.upscaled_image = np.array(self.upscaled_image) / 255.0
    
    def _pad_image(self):
        """
        Applied zero padding to the edge of the image.
        """
        self.padded_image = np.pad(self.upscaled_image, ((self.edge_pad, self.edge_pad), (self.edge_pad, self.edge_pad), (0, 0)))
    
    def enhance(self, image_path, return_path=None):
        """
        Enhances the image using the model provided.
        """
        # Process input image into numpy array.
        self.image_path = image_path
        self._load_image()
        self._bicubic_upscale()
        self._pad_image()
        x_image = np.array([self.padded_image])

        # Set 'x_image' as input data and predict.
        self.interpreter.set_tensor(self.input_details[0]['name'], x_image)
        self.interpreter.invoke()
        output_details = self.interpreter.get_output_details()[0]
        pred_image = self.interpreter.get_tensor(output_details['index'])
        if return_path is not None:
            new_im = Image.fromarray(pred_image[0])
            new_im.save(return_path)
        else:
            return pred_image[0]

def image_upscale(event, context):
    model_path = 'lite_models/20201107_basic_2x_srcnn.tflite'
    srcnn_model = SRCNNDeploy(model_location=model_path, scaling_factor=2, edge_pad=6)
    print(f"Building SRCNN model from path: {model_path}")
    print(srcnn_model)
    print(srcnn_model.scaling_factor, srcnn_model.edge_pad)
    return 'Function complete.'
