import os
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import scale
import concurrent.futures

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
args = parser.parse_args()

from data import *



np.random.seed(1114)
tf.random.set_seed(1114)

BATCH = 384


def predict_label_model(model_name, predict_x, true_y, train_y, genre_rate):
# Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*3)])

# Load model
    model = load_model(model_name)

# Predict
    predict_prob = model.predict(predict_x, batch_size=BATCH).flatten()
    print("max =", np.max(predict_prob, axis=0))
    print("min =", np.min(predict_prob, axis=0))
    print("mean =", np.mean(predict_prob, axis=0))
    print("50% =", np.percentile(predict_prob, 50, axis=0))
    print(predict_prob[:10])
    print(train_y[:10])
    
    predict_y = np.copy(true_y)
    for i in np.arange(0.02, 0.21, 0.02):
        threshold = np.sort(predict_prob)[-int(genre_rate * i * predict_y.shape[0])]
        tmp = np.where(predict_prob >= threshold, predict_prob, 0)
        tmp = np.where(tmp * 0.9 > train_y, 1, 0)
        tmp = np.bitwise_or(tmp.astype(int), true_y.astype(int))

        ambig_ratio = (np.sum(tmp) - np.sum(true_y)) / (genre_rate * i * tmp.shape[0])
        print(i, threshold, ambig_ratio)
        if(ambig_ratio <= 0.2):
            predict_y = tmp

    return predict_y


if __name__ == '__main__':
# Data processing
    ratings_df = pd.read_csv("{_dir}/data/ratings_{_dir}.csv".format(_dir=args.dir))
    items_df = pd.read_csv("{_dir}/data/items_{_dir}.csv".format(_dir=args.dir))
    _, user_dict, rated_item, item_dict, true_y, _ = df_to_np(ratings_df, items_df)
    mf = np.memmap("{_dir}/mf.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), len(user_dict)))
    mf = scale(mf, axis=0)
    predict_x = mf[np.unique(np.sort(rated_item))]
    print(np.sum(true_y) / true_y.shape[0])

    try:
        supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
    except:
        supmatrix_df = pd.read_csv("{_dir}/data/sup_{_dir}.csv".format(_dir=args.dir))
        supmatrix_save(supmatrix_df, item_dict, "{_dir}/supmatrix.dat".format(_dir=args.dir))
        supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
    train_y = (supmatrix[np.unique(np.sort(rated_item))] + true_y).clip(0, 1)

    genre_rate = np.sum(true_y, axis=0) / true_y.shape[0]
    predict_y = np.zeros(true_y.shape)
    for label in range(true_y.shape[1]):
        print("\n\n\nlabel =", label)
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(predict_label_model, "{_dir}/supsoft/mf_model{label}.hdf5".format(_dir=args.dir, label=label), predict_x, true_y[:, label], train_y[:, label], genre_rate[label])
            predict_y[:, label] = future.result()
 
    print(np.sum(predict_y) / predict_y.shape[0])
    print(np.sum(predict_y, axis=0) - np.sum(true_y, axis=0))

    for i in items_df.index:
        itemid = items_df.loc[i, "itemId"]
        if(itemid in item_dict.keys()):
            tmp = ""
            for j in range(predict_y.shape[1]):
                if predict_y[item_dict[itemid]][j]:
                    if tmp == "":
                        tmp += str(j)
                    else:
                        tmp += " " + str(j)
                else:
                    continue
            items_df.loc[i, "genres"] = tmp
        else:
            continue

    items_df.to_csv("{_dir}/supsoft/mf18_result.csv".format(_dir=args.dir), index=False)
