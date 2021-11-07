import tensorflow as tf
import tensorflow_text as tft
from tensorflow.keras import layers as tfnn

import numpy as np


try:
    from ..archs.transformer_arch import Encoder
except ImportError:
    Encoder = None
    raise ImportError('Encoder' + ' non importé')

try:
    from ..archs.transformer_arch import Decoder
except ImportError:
    Decoder = None
    raise ImportError('Decoder' + ' non importé')


class Transformer(tf.keras.Mode):
    
    def __init__(self, ):












































