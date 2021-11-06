import unittest

import tensorflow_datasets as tfds
import tensorflow_text as tft
import tensorflow as tf
import numpy as np


try:
    import transformer_arch as arch
except ImportError:
    cc = None
    raise ImportError('transformer_arch', ' non importé')





# Globales
# Hp globaux
BATCH_SIZE = arch.BATCH_SIZE
BUFFER_SIZE = arch.BUFFER_SIZE
EMBEDDING_DIM = arch.EMBEDDING_DIM
MAX_SEQ_LENGTH = arch.MAX_SEQ_LENGTH
INPUT_VOCAB_SIZE = arch.INPUT_VOCAB_SIZE
TARGET_VOCAB_SIZE = arch.TARGET_VOCAB_SIZE

n_heads = arch.n_heads



# ************************* chargement données ***************************

## ds
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

## tokenizer
model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir='.', cache_subdir='', extract=True)

tokenizers = tf.saved_model.load(model_name)





# ************************* tests TokenizerBlock ***********************





t_b = arch.TokenizerBlock(tokenizers.pt, tokenizers.en)

ds = t_b.prepare_dataset(train_examples)


# tests

class TestTokenizerBlock_0(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """va run ça avant de faire les tests
        """
        cls.t_b = arch.TokenizerBlock(tokenizers.pt, tokenizers.en)
        cls.ds = cls.t_b.prepare_dataset(train_examples)

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self):
        pass
    
    def tearDown(self) -> None:
        pass
        
    def test_prepare_dataset(self):
        
        for pt, en in self.ds.take(1):
            print(pt)
            print()
            
            print(en)
            print()
            
            self.assertEqual(pt.shape.as_list()[0], BATCH_SIZE)
            self.assertEqual(len(pt.shape.as_list()), 2)



class TestTokenizerBlock(tf.test.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """va run ça avant de faire les tests
        """
        cls.T = arch.TokenizerBlock(tokenizers.pt, tokenizers.en)
        cls.ds = cls.T.prepare_dataset(train_examples)

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self):
        pass
    
    def tearDown(self) -> None:
        pass
        
    def test_prepare_dataset(self):
        
        for pt, en in self.ds.take(1):
            print(pt)
            print()
            
            print(en)
            print()
            
            self.assertEqual(pt.shape.as_list()[0], BATCH_SIZE)
            self.assertEqual(len(pt.shape.as_list()), 2)
            self.assertDTypeEqual(pt, np.int64)
            
            self.assertEqual(en.shape.as_list()[0], BATCH_SIZE)
            self.assertEqual(len(en.shape.as_list()), 2)
            self.assertDTypeEqual(en, np.int64)





# ************************* tests EmbeddingBlock ***********************





class TestEmbeddingBlock(tf.test.TestCase):
    
    @classmethod
    def setUpClass(cls) -> None:
        
        cls.T = arch.TokenizerBlock(tokenizers.pt, tokenizers.en)
        cls.ds = cls.T.prepare_dataset(train_examples)
        
        cls.E = arch.EmbeddingBlock()
        
        return super().setUpClass()
    
    @classmethod
    def tearDownClass(cls) -> None:
        pass
    
    #@unittest.expectedFailure
    def test_call(self):
        
        pt, en = next(ds.__iter__())
        
        x = self.E(pt)
        
        self.assertEqual(x.shape.as_list()[0], BATCH_SIZE)
        self.assertEqual(len(x.shape.as_list()), 3) # shape de x : [b, s, e]
        self.assertDTypeEqual(x, np.float32)





# ************************* tests AttentionBlock ***********************






class TestAttentionBlock(tf.test.TestCase):
    
    @classmethod
    def setUpClass(cls) -> None:
        
        cls.T = arch.TokenizerBlock(tokenizers.pt, tokenizers.en)
        cls.ds = cls.T.prepare_dataset(train_examples)
        
        cls.E = arch.EmbeddingBlock()
        
        cls.A = arch.AttentionBlock(n_heads = n_heads)
        
        cls.seq_length = 8
        cls.seq = tf.constant([[5, 3, 0, 0], [2, 3, 3, 0], [2, 4, 7, 0]])

        
        return super().setUpClass()
    
    @classmethod
    def tearDownClass(cls) -> None:
        pass
    
    def test_padding_mask(self):
        
        xpected = tf.constant([[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
        self.assertEqual(self.A.generate_padding_mask(self.seq), xpected)
    
    def test_not_look_ahead_mask(self):
        
        xpected = tf.constant( [[0., 1., 1., 1., 1., 1.],
                                [0., 0., 1., 1., 1., 1.],
                                [0., 0., 0., 1., 1., 1.],
                                [0., 0., 0., 0., 1., 1.],
                                [0., 0., 0., 0., 0., 1.],
                                [0., 0., 0., 0., 0., 0.]])
        self.assertEqual(self.A.generate_not_look_ahead_mask(self.seq_length), xpected)
    
    def test_attention_formula(self):
        
        q = tf.random.uniform(shape = [self.seq_length, EMBEDDING_DIM])
        k = q
        v = q

        msk = self.A.generate_padding_mask(self.seq)
        _msk = self.A.generate_not_look_ahead_mask(self.seq_length)
        
        out, att = self.A.scale_dot_product_with_mask(q, k, v, msk)
        _out, _att = self.A.scale_dot_product_with_mask(q, k, v, _msk)
        
        self.assertShapeEqual(out [self.seq_length, EMBEDDING_DIM])
        self.assertShapeEqual(_out, [self.seq_length, EMBEDDING_DIM])
        
        self.assertShapeEqual(att, [self.seq_length, self.seq_length])
        self.assertShapeEqual(_att, [self.seq_length, self.seq_length])
        
        # TODO :essayer de print pour voir les 'moins l'infini'
        
    def test_split_into_heads(self):

        q = tf.random.uniform(shape = [BATCH_SIZE, self.seq_length, EMBEDDING_DIM])

        self.assertShapeEqual(self.A.split_embedding_into_heads(q), tf.constant([BATCH_SIZE, n_heads, self.seq_length, EMBEDDING_DIM]))

    # @unittest.expectedFailure
    # def test_call(self):
        
    #     pt, en = next(ds.__iter__())
        
    #     x = self.E(pt)
        
    #     self.assertEqual(x.shape.as_list()[0], BATCH_SIZE)
    #     self.assertEqual(len(x.shape.as_list()), 3) # shape de x : [b, s, e]
    #     self.assertDTypeEqual(x, np.float32)









    
    













# uncomment ca pour faire marcher le debugger

if __name__ == '__main__':
    unittest.main()
