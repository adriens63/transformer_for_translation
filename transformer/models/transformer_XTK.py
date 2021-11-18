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

try:
    from ..archs.transformer_arch import TokenizerBlock
except ImportError:
    TokenizerBlock = None
    raise ImportError('TokenizerBlock' + ' non importé')





class Transformer(tf.keras.Model):
    
    def __init__(self, tokenizer_a, 
                 tokenizer_b,
                 n_enc,
                 n_enc_blocks, 
                 n_heads, 
                 n_layers_ff, 
                 hidden_dim_ff, 
                 dropout,
                 n_dec,
                 n_dec_blocks):
        super(Transformer, self).__init__()
        
        self.T = TokenizerBlock(tokenizer_a, tokenizer_b)
        self.E = [Encoder(n_enc_blocks, n_heads, n_layers_ff, hidden_dim_ff, dropout) for _ in range(n_enc)]
        self.D = [Decoder(n_dec_blocks, n_heads, n_layers_ff, hidden_dim_ff, dropout) for _ in range(n_dec)]
        
    def call(self, x):
        











































