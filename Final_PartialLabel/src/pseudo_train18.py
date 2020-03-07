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
from keras.regularizers import l1_l2
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks.callbacks import EarlyStopping
from keras.callbacks.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.utils.class_weight import compute_class_weight
import concurrent.futures

from loss import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--pseudo', type=str, required=True, choices=["random", "conf"])
parser.add_argument('--feature', type=str, required=True)
args = parser.parse_args()

from data import *

global x

np.random.seed(1114)
tf.random.set_seed(1114)

BATCH = 128
EPOCH = 1000
LR = 2e-4

def train_mf_model(train_y, checkpoint):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*5)])
    
    yn_weight = compute_class_weight('balanced', np.unique(train_y), train_y)
    yn_weight = dict(enumerate(yn_weight))
    print(yn_weight)
    
    train_x, train_y = shuffle(x, train_y)

    # Build model
    feature_input = Input(shape=[train_x.shape[1], ])
    nn = Dense(256)(feature_input)
    nn = Dropout(0.3)(nn)
    nn = Dense(512)(nn)
    nn = Dropout(0.3)(nn)
    nn = Dense(1024)(nn)
    output = Dense(1, activation="sigmoid")(nn)

    model = Model(feature_input, output)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=LR), metrics=["acc"])

    # Train
    model.fit(train_x, train_y, batch_size=BATCH, epochs=EPOCH, 
            validation_split=0.2, callbacks=checkpoint, class_weight=yn_weight)

def train_sup_model(train_y, checkpoint):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*5)])
    
    yn_weight = compute_class_weight('balanced', np.unique(train_y), train_y)
    yn_weight = dict(enumerate(yn_weight))
    print(yn_weight)
    
    train_x, train_y = shuffle(x, train_y)

    # Build model
    feature_input = Input(shape=[train_x.shape[1], ])
    nn = Dense(32, kernel_regularizer=l1_l2(1e-3, 1e-3))(feature_input)
    nn = Dropout(0.3)(nn)
    nn = Dense(64, kernel_regularizer=l1_l2(1e-3, 1e-3))(nn)
    output = Dense(1, activation="sigmoid")(nn)

    model = Model(feature_input, output)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=LR), metrics=["acc"])

    # Train
    model.fit(train_x, train_y, batch_size=BATCH, epochs=EPOCH, 
            validation_split=0.2, callbacks=checkpoint, class_weight=yn_weight)

if __name__ == '__main__':
# Data processing
    ratings_df = pd.read_csv("{_dir}/data/ratings_{_dir}.csv".format(_dir=args.dir))
    items_df = pd.read_csv("{_dir}/data/items_{_dir}.csv".format(_dir=args.dir))
    _, user_dict, rated_item, item_dict, train_y, _ = df_to_np(ratings_df, items_df)
    print("number of genre in valid items =", np.sum(train_y, axis=0))
    print("average number of labels =", np.sum(train_y) / train_y.shape[0])

    mf = np.memmap("{_dir}/mf.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), len(user_dict)))
    mf = scale(mf, axis=0)

    try:
        supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
    except:
        supmatrix_df = pd.read_csv("{_dir}/data/sup_{_dir}.csv".format(_dir=args.dir))
        supmatrix_save(supmatrix_df, item_dict, "{_dir}/supmatrix.dat".format(_dir=args.dir))
        supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))

    if(args.feature == "mf"):
        x = mf[np.unique(np.sort(rated_item))]
    elif(args.feature == "sup"):
        x = supmatrix[np.unique(np.sort(rated_item))]
    else:
        exit()

    if(args.pseudo == "random"):
        train_y = build_random_pseudo(supmatrix, train_y, item_dict)
    elif(args.pseudo == "conf"):
        train_y = build_conf_pseudo(supmatrix, train_y, item_dict)
    else:
        exit()
    tmp = np.memmap("{_dir}/{pseudo}_pseudo/{feature}_y.dat".format(_dir=args.dir, pseudo=args.pseudo, feature=args.feature), dtype='float32', mode='w+', shape=(len(item_dict), 18))
    tmp[:] = train_y[:]

    print("number of pseudo genre in valid items =", np.sum(train_y, axis=0))
    print("average number of pseudo-labels =", np.sum(train_y) / train_y.shape[0])
    
    for label in range(train_y.shape[1]):
        checkpoint = [EarlyStopping(monitor="val_loss", patience=100), ModelCheckpoint("{_dir}/{pseudo}_pseudo/0{feature}_model{label}.hdf5".format(_dir=args.dir, pseudo=args.pseudo, feature=args.feature, label=label), monitor="val_loss", verbose=1, save_best_only=True, mode="min")]
        print("\n\n\n\n\nlabel =", label)
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            if(args.feature == "mf"):
                future = executor.submit(train_mf_model, train_y[:, label], checkpoint)
            elif(args.feature == "sup"):
                future = executor.submit(train_sup_model, train_y[:, label], checkpoint)
                print(future.exception())
            else:
                exit()
