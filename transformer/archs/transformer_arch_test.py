import unittest

import tensorflow_datasets as tfds
import tensorflow_text as tft
import tensorflow as tf
import numpy as np

try:
    import transformer_arch as arch
except ImportError:
    cc = None
    print('transformer_arch', ' non importé')





# Globales
# Hp globaux
BATCH_SIZE = arch.BATCH_SIZE
BUFFER_SIZE = arch.BUFFER_SIZE
EMBEDDING_DIM = arch.EMBEDDING_DIM
MAX_SEQ_LENGTH = arch.MAX_SEQ_LENGTH
INPUT_VOCAB_SIZE = arch.INPUT_VOCAB_SIZE
TARGET_VOCAB_SIZE = arch.TARGET_VOCAB_SIZE





# chargement données

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





# tests TokenizerBlock





t_b = arch.TokenizerBlock(tokenizers.pt, tokenizers.en)

ds = t_b.prepare_dataset(train_examples)


# tests

class Test_TokenizerBlock(unittest.TestCase):
    
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





# tests EmbeddingBlock





# print

    
    















if __name__ == '__main__':
    unittest.main()
