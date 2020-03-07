import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*6)])

np.random.seed(1114)
tf.random.set_seed(1114)
 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
args = parser.parse_args()

from data import *


if __name__ == '__main__':
    ratings_df = pd.read_csv("{_dir}/data/ratings_{_dir}.csv".format(_dir=args.dir))
    items_df = pd.read_csv("{_dir}/data/items_{_dir}.csv".format(_dir=args.dir))
    train_user, user_dict, train_item, item_dict, _, train_y = df_to_np(ratings_df, items_df)
    
    model = load_model("{_dir}/mf_model.hdf5".format(_dir=args.dir))
    print(model.summary(), flush=True)

    # mf matrix: (item * user)
    mf = np.memmap("{_dir}/mf1.dat".format(_dir=args.dir), dtype='float32', mode='w+', shape=(len(item_dict), len(user_dict)))
    for i in range(len(train_user)):
        mf[train_item[i]][train_user[i]] = train_y[i]
    for i in range(mf.shape[0]):
        predict_user = []
        for j in range(mf.shape[1]):
            if mf[i][j] == 0:
                predict_user.append(j)
        if(len(predict_user) == 0):
            continue
        predict_item = [i] * len(predict_user)
        predict_y = model.predict_on_batch([predict_user, predict_item])
        for j in range(len(predict_user)):
            mf[i][predict_user[j]] = predict_y[j]
        print(mf[i], len(mf[i]) - len(mf[i].nonzero()[0]))

        if(i % 1000 == 0):
            print(str(i) + "-th is done!", flush=True)
            print(np.mean(np.var(mf[:i], axis=0)))
