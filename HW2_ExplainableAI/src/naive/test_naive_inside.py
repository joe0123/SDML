import os
import sys
import tensorflow as tf
import numpy as np
import keras
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import Flatten
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.models import Model
from keras.models import load_model

from functions import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*1)])

HID_DIM = 128


if __name__ == '__main__':
    texts = []
    with open(sys.argv[1], "r") as f:
        for line in f.readlines():
            texts.append(line.strip('\n'))
    char_index = load_dict(sys.argv[2], texts)
    voc_size = len(char_index)
    encoder_seq_len, encoder_input_data, decoder_seq_len, ctrl_seq_len, ctrl_sig_data = test_data_inside(char_index, texts)
    print(encoder_input_data[:3], ctrl_sig_data[:3])

    encoder_input = Input(shape=(encoder_seq_len, ))
    embedding = Embedding(input_dim = voc_size, output_dim = voc_size, weights = [np.eye(voc_size)], trainable=False)
    encoder_embedding = embedding(encoder_input)
    encoder_gru = Bidirectional(GRU(HID_DIM, return_state=True, dropout=0.1))
    encoder_output, forward_h, backward_h = encoder_gru(encoder_embedding)
    state_h = Concatenate()([forward_h, backward_h])

    decoder_input = Input(shape=(decoder_seq_len, ))
    decoder_embedding = embedding(decoder_input)
    decoder_gru = GRU(2 * HID_DIM, return_sequences=True, return_state=True, dropout=0.1)
    decoder_output, _ = decoder_gru(decoder_embedding, initial_state=state_h)
    decoder_dense = Dense(voc_size, activation='softmax')
    decoder_output = decoder_dense(decoder_output)
    
    model = Model([encoder_input, decoder_input], decoder_output)
    model.load_weights(sys.argv[3])
 
    encoder = Model(encoder_input, state_h)
    print(encoder.summary(), flush=True)

    decoder_input = Input(shape=(1, ))
    decoder_state_input = Input(shape=(HID_DIM*2, ))
    decoder_embedding = embedding(decoder_input)
    decoder_output, decoder_state = decoder_gru(decoder_embedding, initial_state=decoder_state_input)
    decoder_output = decoder_dense(decoder_output)
    decoder = Model([decoder_input, decoder_state_input], [decoder_output, decoder_state])
    print(decoder.summary(), flush=True)

    index_char = {i: c for c, i in char_index.items()}
    for seq_index in range(len(encoder_input_data)):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = '<SOS>'

        states_value = encoder.predict(input_seq)
        new_input = [char_index['<SOS>']]
        
        while True:
            prob_out, states_value = decoder.predict([new_input, states_value])
            out = np.argmax(prob_out, axis=-1)
            char = index_char[int(np.asscalar(out))]
            if(char == '<EOS>' or len(decoded_sentence) >= decoder_seq_len - 1):
                break
            decoded_sentence += char
            new_input = out
        decoded_sentence += '<EOS>'
        print(decoded_sentence, flush=True)
    
