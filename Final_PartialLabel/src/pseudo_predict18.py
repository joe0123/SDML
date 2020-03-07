import os
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import scale
import concurrent.futures
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--pseudo', type=str, required=True, choices=["random", "conf"])
parser.add_argument('--feature', type=str, required=True)
args = parser.parse_args()

from data import *



np.random.seed(1114)
tf.random.set_seed(1114)

BATCH = 384

global predict_x

def predict_mf_model(model_name, true_y, train_y, label):
# Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    plt.subplot(3,1,1)
    plt.hist(predict_prob[(predict_prob * true_y).nonzero()], bins=np.arange(0, 1.01, 0.05))
    plt.subplot(3,1,2)
    plt.hist(predict_prob[(predict_prob * (1 - true_y) * train_y).nonzero()], bins=np.arange(0, 1.01, 0.05))
    plt.subplot(3,1,3)
    plt.hist(predict_prob[(predict_prob * (1 - true_y) * (1 - train_y)).nonzero()], bins=np.arange(0, 1.01, 0.05))
    plt.savefig("{}.jpg".format(label))
    
    predict_y = np.copy(true_y)
    for i in np.arange(0.03, 0.4, 0.03):
        threshold = np.sort(predict_prob)[-int(i * np.sum(true_y))]
        tmp = np.where(predict_prob >= threshold, 1, 0)
         
        ambig_ratio = np.sum(tmp * (1 - train_y)) / int(i * np.sum(train_y))
        print(i, threshold, ambig_ratio)
        if(ambig_ratio <= 0.15):
            predict_y = np.bitwise_or(tmp.astype(int), true_y.astype(int))
    return predict_y

def predict_sup_model(model_name, true_y, train_y, label):
# Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    #plt.subplot(3,1,1)
    #plt.hist(predict_prob[(predict_prob * true_y).nonzero()], bins=np.arange(0, 1.01, 0.05))
    #plt.subplot(3,1,2)
    #plt.hist(predict_prob[(predict_prob * (1 - true_y) * train_y).nonzero()], bins=np.arange(0, 1.01, 0.05))
    #plt.subplot(3,1,3)
    #plt.hist(predict_prob[(predict_prob * (1 - true_y) * (1 - train_y)).nonzero()], bins=np.arange(0, 1.01, 0.05))
    #plt.savefig("{}.jpg".format(label))
    
    predict_y = np.copy(true_y)
    for i in np.arange(0.03, 1.06, 0.03):
        threshold = np.sort(predict_prob)[-int(i * np.sum(train_y))]
        tmp = np.where(predict_prob >= threshold, 1, 0)

        ambig_ratio = np.sum(tmp * (1 - train_y)) / int(i * np.sum(train_y))
        print(i, threshold, ambig_ratio)
        if(ambig_ratio <= 0.08):
            predict_y = np.bitwise_or(tmp.astype(int), true_y.astype(int))
    return predict_y




if __name__ == '__main__':
# Data processing
    ratings_df = pd.read_csv("{_dir}/data/ratings_{_dir}.csv".format(_dir=args.dir))
    items_df = pd.read_csv("{_dir}/data/items_{_dir}.csv".format(_dir=args.dir))
    _, user_dict, rated_item, item_dict, true_y, _ = df_to_np(ratings_df, items_df)
    mf = np.memmap("{_dir}/mf.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), len(user_dict)))
    mf = scale(mf, axis=0)
   
    try:
        supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
    except:
        supmatrix_df = pd.read_csv("{_dir}/data/sup_{_dir}.csv".format(_dir=args.dir))
        supmatrix_save(supmatrix_df, item_dict, "{_dir}/supmatrix.dat".format(_dir=args.dir))
        supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
    if(args.feature == "mf"):
        predict_x = mf[np.unique(np.sort(rated_item))]
    elif(args.feature == "sup"):
        predict_x = supmatrix[np.unique(np.sort(rated_item))]
    else:
        exit()

    train_y = np.memmap("{_dir}/{pseudo}_pseudo/{feature}_y.dat".format(_dir=args.dir, pseudo=args.pseudo, feature=args.feature), dtype='float32', mode='r', shape=(len(item_dict), 18))
    print("number of pseudo genre in valid items =", np.sum(train_y, axis=0))
    print("average number of pseudo-labels =", np.sum(train_y) / train_y.shape[0])
    #a = supmatrix * train_y * (1 - true_y)
    #print(np.max(a[a.nonzero()]), np.min(a[a.nonzero()]))
    
    predict_y = np.zeros(true_y.shape)
    for label in range(true_y.shape[1]):
        print("\n\n\nlabel =", label)
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            if(args.feature == "mf"):
                future = executor.submit(predict_mf_model, "{_dir}/{pseudo}_pseudo/{feature}_model{label}.hdf5".format(_dir=args.dir, pseudo=args.pseudo, feature=args.feature, label=label), true_y[:, label], train_y[:, label], label)
            elif(args.feature == "sup"):
                future = executor.submit(predict_sup_model, "{_dir}/{pseudo}_pseudo/0{feature}_model{label}.hdf5".format(_dir=args.dir, pseudo=args.pseudo, feature=args.feature, label=label), true_y[:, label], train_y[:, label], label)
            else:
                continue
            predict_y[:, label] = future.result()
    print(np.sum(true_y) / true_y.shape[0])
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

    items_df.to_csv("{_dir}/{pseudo}_pseudo/{feature}_result18.csv".format(_dir=args.dir, pseudo=args.pseudo, feature=args.feature), index=False)
