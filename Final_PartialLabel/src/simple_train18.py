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
parser.add_argument('--feature', type=str, required=True)
args = parser.parse_args()

from data import *

global x

np.random.seed(1114)
tf.random.set_seed(1114)

BATCH = 128
EPOCH = 1000
LR = 1e-5

def train_label_model(train_y, checkpoint):
    index, train_x, train_y = shuffle(range(len(x)), x, train_y)
    print(index[-200:])
    yn_weight = compute_class_weight('balanced', np.unique(train_y), train_y)
    yn_weight = dict(enumerate(yn_weight))
    print(yn_weight)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*5)])

    # Build model
    feature_input = Input(shape=(train_x.shape[1], ))
    nn = Dense(256)(feature_input)
    nn = Dropout(0.5)(nn)
    nn = Dense(512)(nn)
    nn = Dropout(0.5)(nn)
    nn = Dense(1024)(nn)
    output = Dense(1, activation="sigmoid")(nn)

    model = Model(feature_input, output)
    #model.compile(loss="binary_crossentropy", optimizer=SGD(lr=LR, momentum=0.01, decay=1e-6), metrics=["acc"])
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=LR), metrics=["acc"])

    # Train
    model.fit(train_x, train_y, batch_size=BATCH, epochs=EPOCH, 
            validation_split=0.2, callbacks=checkpoint, class_weight=yn_weight)

if __name__ == '__main__':
# Data processing
    ratings_df = pd.read_csv("{_dir}/data/ratings_{_dir}.csv".format(_dir=args.dir))
    items_df = pd.read_csv("{_dir}/data/items_{_dir}.csv".format(_dir=args.dir))
    _, user_dict, rated_item, item_dict, train_y, _ = df_to_np(ratings_df, items_df)

    features = np.memmap("{_dir}/{feature}.dat".format(_dir=args.dir, feature=args.feature), dtype='float32', mode='r', shape=(len(item_dict), len(user_dict)))
    features = scale(features, axis=0)
    #features = np.where(features > 0, 1, 0)

    x = features[np.unique(np.sort(rated_item))]
    print(x)

    for label in range(train_y.shape[1]):
        checkpoint = [EarlyStopping(monitor="val_loss", patience=30), ModelCheckpoint("{_dir}/simple/{feature}_model{label}.hdf5".format(_dir=args.dir, feature=args.feature, label=label), monitor="val_loss", verbose=1, save_best_only=True, mode="min")]
        print("\n\n\n\n\nlabel =", label)
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(train_label_model, train_y[:, label], checkpoint)
            print(future.exception())
