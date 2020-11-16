import tflite_runtime.interpreter as tflite
import numpy as np

from PIL import Image


def image_upscale(event, context):
    interpreter = tflite.interpreter
    print(interpreter)
