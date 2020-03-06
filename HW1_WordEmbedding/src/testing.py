import os
import numpy as np
import sys
import tensorflow as tf
import keras.backend as K
import keras
from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import Tokenizer
from keras.optimizers import Adam


pretrained_path = sys.argv[1]
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

sys.path.append(pretrained_path)
from pretrain_bert import data

def gpu_init(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*10)])

def init():
    global SEQ_LEN
    SEQ_LEN = 256


def finetuning_model(checkpoint):
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=SEQ_LEN, training=True, trainable=False)

    inputs = model.inputs[:2]
    cls = model.layers[-6].output
    dropout0 = keras.layers.Dropout(0.5, name='Dropout0')(cls)
    output = keras.layers.Dense(4, activation='sigmoid', name='Output')(dropout0)
    model = keras.models.Model(inputs, output)
    model.load_weights(checkpoint)
    return model


if __name__ == '__main__':
    init()
    gpu_init('0')

    test_data = data(sys.argv[2])
    test_x = test_data.data_to_input(token_dict=load_vocabulary(vocab_path), seq_len=SEQ_LEN, y=False)
    test_x = [test_x] + [np.zeros_like(test_x)]
    model = finetuning_model(sys.argv[3])
    prob_result = model.predict(test_x)
    
    thresholds = [float(i) for i in sys.argv[4:8]]
    result = np.where(prob_result[:,0:1] >= thresholds[0], 1, 0)
    result = np.concatenate((result, np.where(prob_result[:,1:2] >= thresholds[1], 1, 0)), axis=1)
    result = np.concatenate((result, np.where(prob_result[:, 2:3] >= thresholds[2], 1, 0)), axis=1)
    result = np.concatenate((result, np.where(prob_result[:, 3:] >= thresholds[3], 1, 0)), axis=1)

    for i in range(len(result)):
        print(int(result[i][3]), int(result[i][1]), int(result[i][0]), int(result[i][2]), sep=',', end='\n')
    
