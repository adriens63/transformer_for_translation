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





# creation instances

# TokenizerBlock
t_b = arch.TokenizerBlock(tokenizers.pt, tokenizers.en)





# prints
ds = t_b.prepare_dataset(train_examples)
print()

print(ds)
print()

print('n_batches :', ds.cardinality())
print()

for pt, en in ds.take(1):
    print(pt)
    print()
    print(en)
    print()




# tests

class Test_TokenizerBlock_Test(unittest.TestCase):
    
    def test_prepare_dataset(self):
        self.assertEqual(tf.shape(t_b.prepare_dataset(train_examples).take(1)), BATCH_SIZE)
    
    # def test_increment(self):
    #     self.assertEqual(3, 4)

    
    












# t_b = TokenizerBlock(tokenizers.pt, tokenizers.en)

# ds = t_b.prepare_dataset(train_examples)


# for pt, en in ds.take(1):
#     print(pt)
#     print()
#     print(en)
    

# test = EmbeddingBlock()
# print(test.position_embedding())

# print('finished')



if __name__ == '__main__':
    unittest.main()
