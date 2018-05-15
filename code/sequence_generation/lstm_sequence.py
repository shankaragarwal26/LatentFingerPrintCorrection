from __future__ import print_function

import os
import sys
from math import ceil, floor

import numpy as np

#
from PIL import Image

sys.path.append("/Users/shankaragarwal/LP/LatentFingerPrintCorrection/fingerprint_python")
# sys.path.append("/home/ubuntu/code/fingerprint_python")

from constants import constants as fpconst
from preprocessor import get_images as get_images
from encoder_cnn import encoder
import tensorflow as tf

IMAGE_HEIGHT = fpconst.region_height_size
IMAGE_WIDTH = fpconst.region_width_size

