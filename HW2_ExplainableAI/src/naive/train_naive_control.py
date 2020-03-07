import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import RepeatVector
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from functions import *

HID_DIM = 128
BATCH = 128
EPOCH = 500

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*6)])


np.random.seed(871024)
tf.random.set_seed(871024)


if __name__ == "__main__":
    char_index = {'<UNK>': 0, '<SOS>': 1, '<EOS>': 2}
    texts = []
    with open(sys.argv[1], "r") as f:
        for line in f.readlines():
            texts.append(line.strip('\n'))

    char_index = load_dict(sys.argv[2], texts)
    voc_size = len(char_index)
    encoder_seq_len, encoder_input_data, decoder_seq_len, decoder_input_data, decoder_target_data, ctrl_seq_len, ctrl_sig_data = train_data_control(char_index, texts)
    print(encoder_input_data[:3], decoder_input_data[:3], ctrl_sig_data[:3])
    
    encoder_input = Input(shape=(encoder_seq_len, ))
    embedding = Embedding(input_dim = voc_size, output_dim = voc_size, weights = [np.eye(voc_size)], trainable=False)
    encoder_embedding = embedding(encoder_input)
    encoder_output, forward_h, backward_h = Bidirectional(GRU(HID_DIM, return_state=True, dropout=0.1))(encoder_embedding)
    state_h = Concatenate()([forward_h, backward_h])

    decoder_input = Input(shape=(decoder_seq_len, ))
    decoder_embedding = embedding(decoder_input)
    decoder_output, _ = GRU(2 * HID_DIM, return_sequences=True, return_state=True, dropout=0.1)(decoder_embedding, initial_state=state_h)
    decoder_output = Dense(voc_size, activation='softmax')(decoder_output)
    
    model = Model([encoder_input, decoder_input], decoder_output)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer=Adam(1e-3))
    print(model.summary(), flush=True)

    checkpoint = [EarlyStopping(monitor='val_acc', patience=5),
            ModelCheckpoint(sys.argv[3], monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
            ReduceLROnPlateau(monitor='val_acc', factor=0.25, patience=3, verbose=0, mode='max', min_delta=0.001)]
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=EPOCH, batch_size=BATCH, callbacks=checkpoint, validation_split=0.2)
    

