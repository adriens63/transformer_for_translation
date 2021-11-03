import tensorflow as tf
from tensorflow.python.keras.layers.embeddings import Embedding
import tensorflow_datasets as tfds
import tensorflow_text as tft
from tensorflow.keras import layers as tfnn

import numpy as np




# Globales
AUTOTUNE = tf.data.AUTOTUNE

# Hp globaux
BATCH_SIZE = 32
BUFFER_SIZE = 20000
EMBEDDING_DIM = 512
MAX_SEQ_LENGTH = 60
INPUT_VOCAB_SIZE = 8500
TARGET_VOCAB_SIZE = 8000

# Les autres ne sont pas globaux, et devront être passés en param : num_heads, dff



# Actual code
class TokenizerBlock:
    
    def __init__(self, tokenizer_a, tokenizer_b):
        """les deux tokenizers doivent avoir la methode .tokenize qui renvoie le ragged tensor des mots tokenisé

        Args:
            tokenizer_a ([type]): [description]
            tokenizer_b ([type]): [description]
        """
        self.tokenizer_a = tokenizer_a
        self.tokenizer_b = tokenizer_b
    
    def prepare_dataset(self, dataset):
        
        def twofold_tokenization(a, b):
            """
            Tokenize en tf.RaggedTensor, puis fait un padding jusqu'à la longueur max en les convertissant en tf.Tensor

            Args:
                a ([type]): [description]
                b ([type]): [description]
            """
            a = self.tokenizer_a.tokenize(a)
            a = a.to_tensor()
            
            b = self.tokenizer_b.tokenize(b)
            b = b.to_tensor()
        
            return a, b
        
        return (dataset
                .cache()
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .map(twofold_tokenization, num_parallel_calls = AUTOTUNE)
                .prefetch(AUTOTUNE))

        

class EmbeddingBlock(tfnn.Layer):
    def __init__(self):
        
        super(EmbeddingBlock, self).__init__()
    
    
    @staticmethod    
    def inside_sine_cosine(pos, i):
        temp = 1 / np.power(10000, (2 * (i // 2)) / np.float32(EMBEDDING_DIM) )
        return pos * temp
    
            
        





































































