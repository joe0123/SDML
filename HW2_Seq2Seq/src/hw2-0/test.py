import os
import sys
import tensorflow as tf
import numpy as np
from keras.models import load_model

from functions import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])

BATCH = 128

if __name__ == '__main__':
    texts = []
    with open(sys.argv[1], "r") as f:
        for line in f.readlines():
            texts.append(line.strip('\n'))
    char_index = load_dict(sys.argv[2], texts)
    seq_len, encoder_input_data, decoder_input_data, decoder_target_data = data_preprocessing(char_index, texts)

    model = load_model(sys.argv[3])
    index_char = {i: c for c, i in char_index.items()}
    out = np.argmax(model.predict([encoder_input_data, decoder_input_data], batch_size=BATCH), axis=-1)
    for seq in out:
        print('<SOS>', end='', flush=True)
        for text in seq:
            char = index_char[text]
            if char == "<EOS>":
                print("<EOS>", flush=True, end='')
                break
            print(char.lower(), flush=True, end='')
        print("\n")
