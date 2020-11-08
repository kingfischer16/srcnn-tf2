"""
SRCNN_UPSCALE_API.PY
====================

Script to run on AWS Lambda to use a pre-trained SRCNN
model for image upscaling.
"""

# Imports.
import numpy as np
from PIL import Image
from ..srcnn_tf2.model.srcnn_model import SRCNNDeploy
