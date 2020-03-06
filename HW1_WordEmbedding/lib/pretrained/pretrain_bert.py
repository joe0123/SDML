import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras_bert import compile_model, gen_batch_inputs, get_model, load_trained_model_from_checkpoint
from keras_bert import load_vocabulary, Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import re
import random
from math import ceil


def gpu_init(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*10)])

def init():
    global pretrained_path, config_path, checkpoint_path, vocab_path    
    pretrained_path = sys.argv[1]
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')

    global SEQ_LEN, BATCH_SIZE
    SEQ_LEN = 256
    BATCH_SIZE = 10

class bert():
    def __init__(self, config, checkpoint, st_pairs, token_dict, token_list, training, trainable, seq_len):
        self.st_pairs = st_pairs
        self.token_dict = token_dict
        self.token_list = token_list
        self.seq_len = seq_len
        self.model = load_trained_model_from_checkpoint(config, checkpoint, training=training, trainable=trainable, seq_len = self.seq_len)
        self.model.compile(optimizer=keras.optimizers.Adam(2e-5), loss=keras.losses.sparse_categorical_crossentropy)
        print(self.model.summary(), flush=True)
    
    def _train_generator(self, steps):
        while True:
            random.shuffle(self.st_pairs)
            for i in range(steps):
                if (i + 1) * BATCH_SIZE <= len(self.st_pairs):
                    yield gen_batch_inputs(sentence_pairs=self.st_pairs[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], 
                            token_dict=self.token_dict, token_list=self.token_list, seq_len=self.seq_len)
                else:
                    yield gen_batch_inputs(sentence_pairs=self.st_pairs[i*BATCH_SIZE:],
                            token_dict=self.token_dict, token_list=self.token_list, seq_len=self.seq_len)

    def _valid_generator(self):
        while True:
            valid_sts = [self.st_pairs[i] for i in random.choices([j for j in range(len(self.st_pairs))], k=BATCH_SIZE)]
            yield gen_batch_inputs(sentence_pairs=valid_sts, token_dict=self.token_dict, 
                    token_list=self.token_list, seq_len=self.seq_len)


    def pretrain(self, checkpointfile):
        steps = ceil(float(len(self.st_pairs)) / float(BATCH_SIZE))
        history = self.model.fit_generator(generator=self._train_generator(steps), steps_per_epoch=steps, epochs=100, 
                validation_data=self._valid_generator(), validation_steps=2000,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_MLM_loss', patience=5),
                keras.callbacks.ModelCheckpoint(checkpointfile, monitor='val_MLM_loss', verbose=1, save_best_only=True, mode='min')])
        return history

class data():
    def __init__(self, path):
        self.df = pd.read_csv(path)

    def data_for_pretrain(self, st_pairs, token_dict):
        tokenizer = Tokenizer(token_dict)
        for i in range(len(self.df)):
            sts = self.df.iloc[i, 2].split('$$$')
            for j in range(len(sts)):
                d = r'\s*[-,\./;\'`\[\]<>\?:"\{\}\~!@%#\$\^&\(\)-=\_\+\s]\s*'
                words = tokenizer.tokenize(' '.join(re.split(d, sts[j])))[1:-1]
                if j != 0:
                    st_pairs[-1].append(words)
                st_pairs.append([words])
            if len(st_pairs[-1]) == 1:
                st_pairs.pop()
        token_list = list(token_dict.keys())
        return st_pairs, token_dict, list(token_dict.keys())

    def data_to_input(self, token_dict, seq_len, y=True):
        tokenizer = Tokenizer(token_dict)
        indices, targets = [], []
        count = 0
        for i in range(len(self.df)):
            d = r'\s*[-,\./;\'`\[\]<>\?:"\{\}\~!@%#\$\^&\(\)-=\_\+\s]\s*'
            index, segment = tokenizer.encode(' '.join(re.split(d, self.df.iloc[i, 2])))
            if len(index) > seq_len:
                count += 1
            if len(index) > seq_len:
                indices.append(index[0:int(1*seq_len/4)] + index[int(-3*seq_len/4):])
            else:
                title_index, title_segment = tokenizer.encode(' '.join(re.split(d, self.df.iloc[i, 1])))
                blanks = seq_len - len(index)
                if blanks <= len(title_index) - 2:
                    indices.append(title_index[:blanks+1] + index[1:])
                else:
                    index = title_index[:-1] + index[1:]
                    blanks = seq_len - len(index)
                    indices.append(index + [0]*blanks)
            if y:
                targets.append(tuple(self.df.iloc[i, 6].split(' ')))
        if y:
            return np.array(indices), MultiLabelBinarizer().fit_transform(targets)
        else:
            return np.array(indices)



if __name__ == '__main__':
    init()
    gpu_init('0')

    st_pairs, token_dict, token_list = data(sys.argv[2]).data_for_pretrain([], load_vocabulary(vocab_path))
    st_pairs, token_dict, token_list = data(sys.argv[3]).data_for_pretrain(st_pairs, token_dict)
    model = bert(config=config_path, checkpoint=checkpoint_path, training=True, trainable=True, seq_len=SEQ_LEN, 
            st_pairs=st_pairs, token_dict=token_dict, token_list=token_list)
    model.pretrain(checkpointfile='./checkpoint.hdf5')
