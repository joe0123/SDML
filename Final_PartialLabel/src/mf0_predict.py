import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from scipy.stats.mstats import gmean

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
    train_user, user_dict, train_item, item_dict, true_genre, train_y = df_to_np(ratings_df, items_df)
    model = load_model("{_dir}/mf_model.hdf5".format(_dir=args.dir))
    print(model.summary(), flush=True)

    # mf matrix: (item * user)
    mf = np.memmap("{_dir}/mf1.dat".format(_dir=args.dir), dtype='float32', mode='w+', shape=(len(item_dict), len(user_dict)))
    try:
        supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
    except:
        supmatrix_df = pd.read_csv("{_dir}/data/sup_{_dir}.csv".format(_dir=args.dir))
        supmatrix_save(supmatrix_df, item_dict, "{_dir}/supmatrix.dat".format(_dir=args.dir))
        supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
    soft_genre = (supmatrix[np.unique(np.sort(train_item))] + true_genre).clip(0, 1)
    genre_ratio = np.sum(soft_genre, axis=0) / soft_genre.shape[0]
    print(genre_ratio)

    user_select = np.zeros((len(user_dict), soft_genre.shape[1]))
    for i in range(len(train_user)):
        mf[train_item[i]][train_user[i]] = train_y[i]
    for i in range(user_select.shape[0]):
        tmp = np.sum(soft_genre[mf[:, i].nonzero()], axis=0) / len(mf[:, i].nonzero()[0])
        tmp /= genre_ratio
        for j in range(len(tmp)):
            if tmp[j] <= 1:
                user_select[i][j] = (tmp[j] - 1) * 0.5 + 0.5
            else:
                user_select[i][j] = (tmp[j] - 1) / (1 / genre_ratio[j] - 1) * 0.5 + 0.5
    np.seterr(invalid="ignore")
    probs = np.nan_to_num(np.dot(soft_genre, (user_select).T) / np.sum(soft_genre, axis=1).reshape(soft_genre.shape[0], 1), nan=0.5)
    skip_probs = (1 - probs) * 0.2
    print(np.var(skip_probs))
    
    for i in range(mf.shape[0]):
        predict_user = []
        for j in range(mf.shape[1]):
            if mf[i][j] == 0 and np.random.choice(2, 1, p=[skip_probs[i][j], 1 - skip_probs[i][j]]):
                predict_user.append(j)
        if(len(predict_user) == 0):
            continue
        predict_item = [i] * len(predict_user)
        predict_y = model.predict_on_batch([predict_user, predict_item])
        for j in range(len(predict_user)):
            mf[i][predict_user[j]] = predict_y[j]
        print(mf[i], len(mf[i]) - len(mf[i].nonzero()[0]), flush=True)

        if(i % 1000 == 0):
            print(str(i) + "-th is done!", flush=True)
            print(np.mean(np.var(mf[:i], axis=0)))
