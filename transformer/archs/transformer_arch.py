import tensorflow as tf
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
EPS_LAYERNORM = 1e-6

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
        
        #TODO : faire en sorte que la pos embedding ne soit qu'une troncature, evite de le recalculer
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
        
        assert EMBEDDING_DIM % self.n_heads == 0
                
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
        
    def scale_dot_product_with_mask(self, q, k, v, msk):
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
        
        if msk is not None:
            out = out + msk * (- 1e9) # met un 'poids' de 'moins l'infini' aux endroits du mask ayant un 1 ie les interdits ou <pad>
        
        att = tf.nn.softmax(out, axis = -1)
        out = tf.linalg.matmul(att, v)
        
        return out, att
    
    
    def split_embedding_into_heads(self, x):
        """x a pour shape [b, seq_length, embedding_dim]
            ressort x comme [b, n_heads, seq_length_new, embedding_dim]
        Args:
            x ([type]): [description]
        """                                     #on met -1 ici, qui ajoute une dimension, de sorte que le nombre de composantes du tenseur reste constant
        out = tf.reshape(x, shape = [BATCH_SIZE, -1, self.n_heads, EMBEDDING_DIM // self.n_heads])
        out = tf.transpose(out, perm = [0, 2, 1, 3])
        
        return out
    
    
    def call(self, q, k, v, msk):
        
        # les inp passent par les memes layers pour chaque head
        q = tf.linalg.matmul(q, self.w_q) + self.b_q
        k = tf.linalg.matmul(k, self.w_k) + self.b_k
        v = tf.linalg.matmul(v, self.w_v) + self.b_v
        
        # split 
        q = self.split_embedding_into_heads(q)
        k = self.split_embedding_into_heads(k)
        v = self.split_embedding_into_heads(v)
        
        # formule de l'attention
        out, att = self.scale_dot_product_with_mask(q, k, v, msk) # out shape : [b, nh, s, e] ; att shape : [b, nh, s, s]
        
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
        
        for _ in range(n_layers - 1):
            self.layers.append(tfnn.Dense(hidden_dim, activation = 'relu'))
        self.last_layer = tfnn.Dense(output_dim, activation = 'linear') # juste la pour ajuster la dimension et que ce soit digerable par la suite du model
        
        
    def call(self, x):
        
        out = x
        
        for l in self.layers:
            out = l(out)
            
        out = self.last_layer(out)
        
        return out


class EncoderBlock(tfnn.Layer):
    
    def __init__(self, n_heads = n_heads, n_layers_ff = 2, hidden_dim_ff = 2048, dropout = 0.1):
        """On suppose les sequences tokenizées et emmbeded avant le call

        Args:
            n_heads ([type], optional): [description]. Defaults to n_heads.
            n_layers_ff (int, optional): [description]. Defaults to 2.
            hidden_dim_ff (int, optional): [description]. Defaults to 2048.
            dropout (float, optional): [description]. Defaults to 0.1.
        """
        
        self.A = AttentionBlock(n_heads = n_heads)
        self.D = tfnn.Dropout(dropout)
        self.N = tfnn.LayerNormalization(epsilon = EPS_LAYERNORM)
        
        self.F = FeedForwardBlock(n_layers = n_layers_ff, hidden_dim = hidden_dim_ff, output_dim = EMBEDDING_DIM)
        self.N_ = tfnn.LayerNormalization(epsilon = EPS_LAYERNORM)
        self.D_ = tfnn.Dropout(dropout)
        
    def call(self, x, msk, training):
        
        out, _ = self.A(x, x, x, msk)
        
        out = self.D(out, training = training)
        out = out + x
        out = self.N(out)
        
        
        out_ = self.F(out)
        
        out_ = self.D_(out_, training = training)
        out = out_ + out
        out = self.N_(out)
        
        return out



class DecoderBlock(tfnn.Layer):
    
    def __init__(self, 
                 n_heads = n_heads, 
                 n_layers_ff = 2, 
                 hidden_dim_ff = 2048, 
                 dropout = 0.1):
        
        self.A = AttentionBlock(n_heads)
        self.D = tfnn.Dropout(dropout)
        self.N = tfnn.LayerNormalization(epsilon = EPS_LAYERNORM)
        
        self.A_1 = AttentionBlock(n_heads)
        self.D_1 = tfnn.Dropout(dropout)
        self.N_1 = tfnn.LayerNormalization(epsilon = EPS_LAYERNORM)
        
        self.F = FeedForwardBlock(n_layers_ff, hidden_dim_ff, output_dim = EMBEDDING_DIM)
        self.D_2 = tfnn.Dropout(dropout)
        self.N_2 = tfnn.LayerNormalization(epsilon = EPS_LAYERNORM)
        
    def call(self, x, last_enc_x, not_look_ahead_msk, padding_msk, training):
        
        out, att = self.A(x, x, x, padding_msk)
        
        out = self.D(out)
        out = out + x
        out = self.N(out)
        
        
        out_, att_1 = self.A_1(last_enc_x, last_enc_x, out, not_look_ahead_msk)
        
        out_ = self.D_1(out_)
        out = out + out_
        out = self.N_1(out)
        
        
        out_ = self.F(out)
        
        out_ = self.D_2(out_)
        out = out + out_
        out = self.N_2(out)
        
        return out, att, att_1



class Encoder(tfnn.Layer):
    
    def __init__(self, n_enc_blocks, 
                 n_heads, 
                 n_layers_ff = 2, 
                 hidden_dim_ff = 2048,
                 dropout = 0.1):

        self.EBD = EmbeddingBlock()
        self.ECD = [EncoderBlock(n_heads, n_layers_ff, hidden_dim_ff, dropout) for _ in range(n_enc_blocks)]
        
        self.D = tfnn.Dropout(dropout)
        
        def call(self, tok, msk, training):
            """les seq sont supposées tokenisées

            Args:
                tok ([type]): [description]
                msk ([type]): [description]
                training ([type]): [description]
            """
            
            out = self.EMB(tok)
            
            out = self.D(out)
            
            for l in self.ECD:
                out = l(out)
            
            return out



class Decoder(tfnn.Layer):
    
    def __init__(self, n_dec_blocks, 
                n_heads, 
                n_layers_ff = 2, 
                hidden_dim_ff = 2048,
                dropout = 0.1):               
            
        self.E = EmbeddingBlock()
        self.DCD = [DecoderBlock(n_heads, n_layers_ff, hidden_dim_ff, dropout) for _ in range(n_dec_blocks)]
        
        self.D = tfnn.Dropout(dropout)
        
    def call(self, tok, last_enc_x, not_look_ahead_msk, padding_msk, training):
        
        out = self.E(tok)
        
        acc = 0
        
        att_w = {} # stockage des attentions weights
        
        for l in self.DCD:
            
            acc += 1

            out, att, att_1 = l(out)
            
            att_w[f'attention_weights_du_block_{acc}_sous_block_0'] = att 
            att_w[f'attention_weights_du_block_{acc}_sous_block_1'] = att_1 

        return out, att_w

    
                
            
                           
            
            


        
        
         
        




        






























































