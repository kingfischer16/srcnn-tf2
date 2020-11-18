"""
The handler function for using pretrained models to upscale
images. The predictor class is included in the handler file
for convenience (there's not much to it) and makes use of
the 'tflite_runtime' for predicting from a saved model.
"""
# Imports.
import io
import boto3
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image


# Setting client.
s3 = boto3.client('s3')


class SRCNNDeployLite:
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
        self.input_details = self.interpreter.get_input_details()
        self.scaling_factor = scaling_factor
        self.edge_pad = edge_pad
    
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
    
    def enhance(self, image):
        """
        Enhances the image using the model provided.
        """
        # Process input image into numpy array.
        self.image = image
        self._bicubic_upscale()
        self._pad_image()
        x_image = np.array([self.padded_image], dtype=np.float32)
        im_size = x_image.shape
        
        # Resize input tensor and allocate.
        self.interpreter.resize_tensor_input(
            self.input_details[0]['index'],
            [im_size[0], im_size[1], im_size[2], 3],
            strict=True)
        self.interpreter.allocate_tensors()

        # Set 'x_image' as input data and predict.
        self.interpreter.set_tensor(self.input_details[0]['index'], x_image)
        self.interpreter.invoke()
        output_details = self.interpreter.get_output_details()[0]
        pred_image = self.interpreter.get_tensor(output_details['index'])
        return pred_image[0]


def get_s3_image(bucket, key):
    """
    Retrieves the image from S3.
    """
    response = s3.get_object(Bucket=bucket, Key=key)
    imagecontent = response['Body'].read()
    
    image_file = io.BytesIO(imagecontent)
    img = Image.open(image_file)
    return img


def upload_to_s3(bucket, key, image):
    """
    Writes the thumbnail image to s3.
    """
    out_image = io.BytesIO()
    image.save(out_image, 'PNG')
    out_image.seek(0)

    response =s3.put_object(
        ACL='private',
        Body=out_image,
        Bucket=bucket,
        ContentType='image/png',
        Key=key
    )
    print(response)

    url = f'{s3.meta.endpoint_url}/{bucket}/{key}'
    return url

def image_upscale(image):
    """
    Upscale the image given the input image and scaling factor.
    """
    model_path = 'lite_models/20201107_basic_2x_srcnn.tflite'
    srcnn_model = SRCNNDeployLite(model_location=model_path, scaling_factor=2, edge_pad=6)
    output_arr = srcnn_model.enhance(image)
    output_image = Image.fromarray(output_arr)
    return output_image

def image_upscaler(event, context):
    """
    Handler method.
    """
    # parse event, just for understanding what is in the event
    print(event)
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    # only create a thumbnail on non-thumbnail images
    if not key.endswith("_upscaled.png"):
        # get the image
        image = get_s3_image(bucket, key)
        # upscale image
        upscaled_image = image_upscale(image)
        # upload the file
        upscaled_key = key.rsplit('.', 1)[0] + '_upscaled.png'
        url = upload_to_s3(bucket, upscaled_key, upscaled_image)
        return url
