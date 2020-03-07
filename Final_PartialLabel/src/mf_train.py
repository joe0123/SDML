import os
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks.callbacks import EarlyStopping
from keras.callbacks.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
args = parser.parse_args()

from data import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*6)])


np.random.seed(1114)
tf.random.set_seed(1114)

BATCH = 512
EPOCH = 300
LR = 1e-3

if __name__ == '__main__':
    ratings_df = pd.read_csv("{_dir}/data/ratings_{_dir}.csv".format(_dir=args.dir))
    items_df = pd.read_csv("{_dir}/data/items_{_dir}.csv".format(_dir=args.dir))
    train_user, user_dict, train_item, item_dict, _, train_y = df_to_np(ratings_df, items_df)
    # WARNING: Keras split first then shuffle, so we have to shufflue by ourselves first
    train_user, train_item, train_y = shuffle(train_user, train_item, train_y)

    user_input = Input(shape=(1, ))
    item_input = Input(shape=(1, ))
    user_embedding = Embedding(input_dim=len(user_dict), output_dim=256)(user_input)
    item_embedding = Embedding(input_dim=len(item_dict), output_dim=384, name="item_embedding")(item_input)
    all_input = Concatenate()([user_embedding, item_embedding])

    NN = Flatten()(all_input)
    NN = Dropout(0.2)(NN)
    NN = Dense(128)(NN)
    NN = Dropout(0.3)(NN)
    NN = Dense(256)(NN)
    NN = Dropout(0.3)(NN)
    NN = Dense(512)(NN)
    NN = Dropout(0.5)(NN)
    output = Dense(1, activation='relu')(NN)
    
    model = Model([user_input, item_input], output)
    print(model.summary())
    model.compile(loss="mae", optimizer=Adam(LR))
    
    checkpoint = [EarlyStopping(monitor="val_loss", patience=30),
                ModelCheckpoint("{_dir}/mf_model.hdf5".format(_dir=args.dir), monitor="val_loss", verbose=1, save_best_only=True, mode="min")]
    model.fit([train_user, train_item], train_y, validation_split=0.2, batch_size=BATCH, epochs=EPOCH, callbacks=checkpoint)

