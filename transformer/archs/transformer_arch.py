import tensorflow as tf
from tensorflow.python.keras.layers.embeddings import Embedding
import tensorflow_datasets as tfds
import tensorflow_text as tft
from tensorflow.keras import layers as tfnn

import numpy as np

from math import ceil




# Globales
AUTOTUNE = tf.data.AUTOTUNE

# Hp globaux
BATCH_SIZE = 32
BUFFER_SIZE = 20000
EMBEDDING_DIM = 512
MAX_SEQ_LENGTH = 200
INPUT_VOCAB_SIZE = 8500
TARGET_VOCAB_SIZE = 8000

# Les autres ne sont pas globaux, et devront être passés en param : num_heads, dff
n_heads = 8


## Blocks

class TokenizerBlock:
    
    def __init__(self, tokenizer_a, tokenizer_b):
        """les deux tokenizers doivent avoir la methode .tokenize qui renvoie le ragged tensor des mots tokenisés

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

        

class EmbeddingBlock_0(tfnn.Layer):
    def __init__(self):
        
        super(EmbeddingBlock_0, self).__init__()
    
    
    @staticmethod    
    def inside_sine_cosine(pos, i):
        temp = 1 / np.power(10000, (2 * (i // 2)) / np.float32(EMBEDDING_DIM) )
        return pos * temp
    
    def position_embedding(self):
        temp = tf.range(start = 0, limit = MAX_SEQ_LENGTH)
        temp = tf.expand_dims(temp, axis = -1)
        temp = tf.tile(temp, tf.constant([1, EMBEDDING_DIM]))
        temp = temp.numpy()
        
        out = np.zeros(shape = (MAX_SEQ_LENGTH, EMBEDDING_DIM), dtype = np.float32)
        #out[:, ::2] = np.sin(temp)
        return self.inside_sine_cosine()
        





class EmbeddingBlock(tfnn.Layer):
    
    def __init__(self):
        
        super(EmbeddingBlock, self).__init__()
        self.token_embedding = tfnn.Embedding(input_dim = INPUT_VOCAB_SIZE, output_dim = EMBEDDING_DIM, )

    @staticmethod
    def vector_angles(pos, i):
        temp = 1 / tf.math.pow(10000, (2 * (i//2)) / np.float32(EMBEDDING_DIM))
        return tf.linalg.matmul(pos, temp) # multiplie shape [seq_length, 1] par [1, EMBEDDING_DIM]
    
    def position_embedding(self, seq_length):
        """creer la matrices des positions embedding d'UNE sequence, donc on obtient la shape
        [seq_length, EMBEDDING_DIM], reste à la tile dans le call pour obtenir [BATCH_SIZE, seq_length, EMBEDDING_DIM]

        Args:
            seq_length ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        embedding_idx = tf.range(start = 0, limit = EMBEDDING_DIM, dtype = tf.float32)
        embedding_idx = tf.expand_dims(embedding_idx, 0) # shape [1, EMBEDDING_DIM]

        pos = tf.range(start = 0, limit = seq_length, dtype = tf.float32)
        pos = tf.expand_dims(pos, -1) # shape [seq_length, 1]
        
        angles = self.vector_angles(pos, embedding_idx)
        
        cond = tf.constant([True, False])
        cond = tf.expand_dims(cond, -1)
        cond = tf.tile(cond, tf.constant([ceil(EMBEDDING_DIM // 2) , seq_length]))
        cond = tf.transpose(cond) # shape [seq_length, EMBEDDING_DIM]
        
        return tf.where(cond, tf.math.sin(angles), tf.math.cos(angles)) # [seq_length, EMBEDDING_DIM]
        
    def call(self, tok):
        
        tok = self.token_embedding(tok)
        out = self.position_embedding(tok.get_shape()[-2]) # seq_length est à la -2e position
        
        out = tf.expand_dims(out, axis = 0) # il faudrait trouver un moyen de cache ces position_embedding, meme si elles changent à chaque seq car seq_length change, il faudrait les compute avec MAX_SEQ_LENGTH puis tronquer
        out = tf.tile(out, tf.constant([BATCH_SIZE, 1, 1]))
        
        out = out + tok # rectfié -> il n'ont pas la meme shape, peut etre erreur
        
        return out
        


class AttentionBlock(tfnn.Layer):
    
    def __init__(self, n_heads):
        
        self.n_heads = n_heads
        super(AttentionBlock, self).__init__()
        
        self.padding_mask = self.generate_padding_mask()
        
        initializer = tf.keras.initializers.GlorotUniform()
        
        self.w_q = tf.Variable(initial_value = initializer(shape = [EMBEDDING_DIM, EMBEDDING_DIM]), trainable = True, dtype = tf.float32)
        self.b_q = tf.Variable(initial_value = initializer(shape = (EMBEDDING_DIM,)), trainable = True, dtype = tf.float32) 
        
        self.w_k = tf.Variable(initial_value = initializer(shape = [EMBEDDING_DIM, EMBEDDING_DIM]), trainable = True, dtype = tf.float32)
        self.b_k = tf.Variable(initial_value = initializer(shape = (EMBEDDING_DIM,)), trainable = True, dtype = tf.float32) 
        
        self.w_v = tf.Variable(initial_value = initializer(shape = [EMBEDDING_DIM, EMBEDDING_DIM]), trainable = True, dtype = tf.float32)
        self.b_v = tf.Variable(initial_value = initializer(shape = (EMBEDDING_DIM,)), trainable = True, dtype = tf.float32) 
        
        self.w   = tf.Variable(initial_value = initializer(shape = [EMBEDDING_DIM, EMBEDDING_DIM]), trainable = True, dtype = tf.float32)
        self.b   = tf.Variable(initial_value = initializer(shape = (EMBEDDING_DIM,)), trainable = True, dtype = tf.float32)

    def generate_padding_mask(self, seq):
        """fait en sorte que le model ne prenne pas le padding comme une input
        les 0 (token associé au padding) sont repérés par des 1 dans le mask
        les autres token sont des 0 du mask

        Args:
            seq ([type]): un batch de sequences de tokens
        """
        msk = tf.cast(tf.equal(seq, 0), tf.float32)
        msk = msk[:, tf.newaxis, tf.newaxis,:]
        
        return msk
    
    def generate_not_look_ahead_mask(self, seq_length):
        """Pareil, il y a des 0 là ou le modèle peut regarder, et des 1 ailleurs

        Args:
            seq_length ([type]): [description]
        """
        inp = tf.ones([seq_length, seq_length]) # ne marche que pour une matrice pour une meme seq, pas pour une seq en langue a , et l'autre en b
                                            #garde tous les 1 du bas
        msk = 1 - tf.linalg.band_part(inp, num_lower = -1, num_upper = 0)
        
        return msk 
        
    def scale_dot_product_with_mask(self, q, k, v, mask):
        """[summary]

        Args:
            q ([type]): [description]
            k ([type]): [description]
            v ([type]): [description]
            mask ([type]): [description]
        """
        embedding_dim = tf.cast(EMBEDDING_DIM, tf.float32)
        
        out = tf.linalg.matmul(q, k, transpose_b = True)
        out = out / tf.math.sqrt(embedding_dim)
        
        if mask is not None:
            out = out + mask * (- 1e9) # met un 'poids' de 'moins l'infini' aux endroits du mask ayant un 1 ie les interdits ou <pad>
        
        att = tf.nn.softmax(out, axis = -1)
        out = tf.linalg.matmul(att, v)
        
        return out, att
    
    
    def split_embedding_into_heads(self, x):
        """x a pour shape [b, seq_length, embedding_dim]
            ressort x comme [b, n_heads, seq_length, embedding_dim]
        Args:
            x ([type]): [description]
        """                                     #on met -1 ici, qui ajoute une dimension, de sorte que le nombre de composantes du tenseur reste constant
        out = tf.reshape(x, shape = [BATCH_SIZE, -1, self.n_heads, EMBEDDING_DIM])
        out = tf.transpose(out, perm = [0, 2, 1, 3])
        
        return out
    
    
    def call(self, q, k, v, mask):
        
        # les inp passent par les memes layers pour chaque head
        q = tf.linalg.matmul(q, self.w_q) + self.b_q
        k = tf.linalg.matmul(k, self.w_k) + self.b_k
        v = tf.linalg.matmul(v, self.w_v) + self.b_v
        
        # split 
        q = self.split_embedding_into_heads(q)
        k = self.split_embedding_into_heads(k)
        v = self.split_embedding_into_heads(v)
        
        # formule de l'attention
        out, att = self.scale_dot_product_with_mask(q, k, v, mask) # out shape : [b, nh, s, e] ; att shape : [b, nh, s, s]
        
        # un-split de l'out
        out = tf.transpose(out, perm = [0, 2, 1, 3]) # out shape : [b, s, nh, e]
        out = tf.reshape(out, shape = [BATCH_SIZE, -1, EMBEDDING_DIM])
        
        out = tf.linalg.matmul(out, self.w) + self.b
        
        return out, att
    


class FeedForwardBlock(tfnn.Layer):
    
    def __init__(self, n_layers, hidden_dim, output_dim):
        """[summary]

        Args:
            n_layers ([type]): [description]
            hidden_dim (int): n_units des layers hidden
            output_dim ([type]): [description]
        """
        self.layers = []
        
        for _ in range(n_layers):
            self.layers.append(tfnn.Dense(hidden_dim, activation = 'relu'))
        self.last_layer = tfnn.Dense(output_dim, activation = 'linear') # juste la pour ajuster la dimension et que ce soit digerable par la suite du model
        
        
    def call(self, x):
        
        out = x
        
        for l in self.layers:
            out = l(out)
            
        out = self.last_layer(out)
        
        return out








        






























































