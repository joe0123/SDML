#reference: https://tinyurl.com/y5pf27on
#reference: https://tinyurl.com/yytlgbz5
import os
import sys
import tensorflow as tf
import keras.backend as K
import pandas as pd
import numpy as np
import keras
from keras.models import load_model
from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sklearn

pretrained_path = sys.argv[1]
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')
our_checkpoint = os.path.join(pretrained_path, 'checkpoint.hdf5')
sys.path.append(pretrained_path)
from pretrain_bert import data

def gpu_init(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*10)])

def init():
    global SEQ_LEN
    SEQ_LEN = 256



def first_model():
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=SEQ_LEN, training=True, trainable=False)
    model.load_weights(our_checkpoint)

    inputs = model.inputs[:2]
    cls = model.layers[-6].output
    dropout0 = keras.layers.Dropout(0.5, name='Dropout0')(cls)
    output = keras.layers.Dense(4, activation='sigmoid', name='Output')(dropout0)
    model = keras.models.Model(inputs, output)
    model.compile(loss='binary_crossentropy', metrics = ['acc'], optimizer=Adam(1e-4))  #multilabel: sigmoid + binary_crossentropy
    return model

def second_model(first_checkpoint):
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=SEQ_LEN, training=True, trainable=True)

    inputs = model.inputs[:2]
    cls = model.layers[-6].output
    dropout0 = keras.layers.Dropout(0.5, name='Dropout0')(cls)
    output = keras.layers.Dense(4, activation='sigmoid', name='Output')(dropout0)
    model = keras.models.Model(inputs, output)
    model.load_weights(first_checkpoint)
    model.compile(loss='binary_crossentropy', metrics = ['acc'], optimizer=Adam(2e-6))  #multilabel: sigmoid + binary_crossentropy
    return model


if __name__ == "__main__":
    init()
    gpu_init('0')
##### data preprocessing #####
    train_data = data(sys.argv[2])
    train_x, train_y = train_data.data_to_input(token_dict=load_vocabulary(vocab_path), seq_len=SEQ_LEN, y=True)
    print(len(train_x), len(train_y))
# split validation data
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
    train_x = [train_x] + [np.zeros_like(train_x)]
    val_x = [val_x] + [np.zeros_like(val_x)]
##### construct bert + simple #####
# construct model1
    model = first_model()
    print(model.summary(), flush=True)
# start training1
    checkpoint = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=10), 
            keras.callbacks.ModelCheckpoint('weight1.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
    history = model.fit(train_x, train_y, batch_size=25, epochs=1000, verbose=2, validation_data=(val_x, val_y), callbacks=checkpoint)

# construct model2
    model = second_model('weight1.hdf5')
    print(model.summary(), flush=True)
# start training2
    checkpoint = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=2), 
            keras.callbacks.ModelCheckpoint('weight2.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
    history = model.fit(train_x, train_y, batch_size=10, epochs=50, verbose=2, validation_data=(val_x, val_y), callbacks=checkpoint)

##### evaluating 2 #####
    prob_result = model.predict(val_x)
#pick threshold
    best_score = -1
    best_indices = []
    for th0 in np.arange(0.21, 0.61, 0.03):
        result = np.where(prob_result[:,0:1] >= th0, 1, 0)
        for th1 in np.arange(0.60, 0.20, -0.03):
            result = np.concatenate((result, np.where(prob_result[:,1:2] >= th1, 1, 0)), axis=1)
            for th2 in np.arange(0.21, 0.61, 0.03):
                result = np.concatenate((result, np.where(prob_result[:, 2:3] >= th2, 1, 0)), axis=1)
                for th3 in np.arange(0.60, 0.20, -0.03):
                    result = np.concatenate((result, np.where(prob_result[:, 3:] >= th3, 1, 0)), axis=1)
                    score = f1_score(val_y, result, average='micro')
                    if score >= best_score:
                        best_score = score
                        best_indices = [th0, th1, th2, th3]
                        print(best_indices, best_score)
                    result = np.delete(result, -1, 1)
                result = np.delete(result, -1, 1)
            result = np.delete(result, -1, 1)
        result = np.delete(result, -1, 1)
    print(best_indices, best_score)


