import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt


def build_dict(nonuniq_list):
    dictionary = {}
    for i in nonuniq_list:
        if i not in dictionary.keys():
            dictionary[i] = len(dictionary)
    return dictionary


def df_to_np(ratings_df, items_df, count_floor=100., std_floor=0.9):
# Find qualified users, items and their data
    user_count = ratings_df.groupby("userId")["rating"].count()
    user_std = ratings_df.groupby("userId")["rating"].std()
    user_stats = pd.DataFrame({"rating_count": user_count, "rating_std": user_std})
    user_candid = user_stats[(user_stats.rating_count >= count_floor) & (user_stats.rating_std >= std_floor)].index.tolist()
    item_candid = items_df["itemId"].values.tolist()
    data_df = ratings_df[(ratings_df.userId.isin(user_candid)) & (ratings_df.itemId.isin(item_candid))]

# Build dict (real_ID:model_ID)
    user_dict = build_dict(np.unique(np.sort(data_df["userId"].values)).tolist())
    item_dict = build_dict(np.unique(np.sort(data_df["itemId"].values)).tolist())

# Make data
    user_data = [user_dict[i] for i in data_df["userId"].values.tolist()]
    item_data = [item_dict[i] for i in data_df["itemId"].values.tolist()]
    rating_data = data_df["rating"].values
    
    item_genre = items_df["genres"].values.tolist()
    genre_data = np.zeros((len(item_dict), 18), dtype=float)
    
    for i in range(len(item_genre)):
        if item_candid[i] in item_dict.keys():
            tmp = str(item_genre[i]).split(" ")
            one_hot = [0] * 18
            for j in tmp:
                if j == "nan":
                    continue
                else:
                    one_hot[int(j)] = 1
            genre_data[item_dict[item_candid[i]]] = np.array(one_hot)
        else:
            continue
    return user_data, user_dict, item_data, item_dict, genre_data, rating_data


def supmatrix_save(df, item_dict, train_y, filename):
    item_candid = df["itemId"].values.tolist()
    
    supmatrix = np.memmap(filename, dtype='float32', mode='w+', shape=(len(item_dict), 18))
    for i in range(len(item_candid)):
        if item_candid[i] in item_dict.keys():
            tmp = df.iloc[i].values.tolist()[1:]
            if(np.max(tmp) != 0):
                supmatrix[item_dict[item_candid[i]]] = tmp / np.max(tmp)
        else:
            continue


def build_random_pseudo(supmatrix, train_y, item_dict):
    #0 prior = 0.06
    #1 prior = np.sum(train_y, axis=0) / np.sum(1 - train_y, axis=0) * 3 / 7
    prior = np.sum(train_y) / np.sum(1 - train_y) * 5 / 5
    print(prior)
    probs = prior * supmatrix
    pseudo_y = np.copy(train_y)
    for i in range(pseudo_y.shape[0]):
        for j in range(train_y.shape[1]):
            if pseudo_y[i][j] == 0:
                pseudo_y[i][j] = np.random.choice(2, 1, p=[1 - probs[i][j], probs[i][j]])
                #3 pseudo_y[i][j] = np.random.choice(2, 1, p=[1 - prior, prior])
            else:
                continue
    tmp = supmatrix[(supmatrix * pseudo_y * (1 - train_y)).nonzero()]
    print(np.max(tmp), np.min(tmp))
    return pseudo_y

def build_conf_pseudo(supmatrix, train_y):
    tmp = supmatrix * train_y
    threshold = np.min(np.where(tmp == 0, np.inf, tmp), axis=0).reshape(train_y.shape[1], 1)
    pseudo_y = np.where(supmatrix.T >= threshold, 1, 0).T
    return pseudo_y



def embed_save(modelname, filename):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*2)])
    
    embed_matrix = np.memmap(filename, dtype='float32', mode='w+', shape=(len(item_dict), 384))
    model = load_model(modelname)
    embed_matrix[:] = model.get_layer("item_embedding").get_weights()[0][:]

    return


def plot_tsne(x, y, prefix):
    for label in range(y.shape[1]):
        x_P = x[np.argwhere(y[:, label] == 1).flatten()]
        x_U = x[np.argwhere(y[:, label] == 0).flatten()]
        plt.scatter(x_U[:, 0], x_U[:, 1], c="gray", edgecolor="none", s=1, alpha=0.5)
        plt.scatter(x_P[:, 0], x_P[:, 1], c="green", edgecolor="none", s=1.2, alpha=1)
        plt.xlabel("Component_1")
        plt.ylabel("Component_2")
        plt.savefig("{prefix}{label}.jpg".format(prefix=prefix, label=label), dpi=300)
        plt.clf()

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument("--random_pseudo", action='store_true')
    parser.add_argument("--conf_pseudo", action='store_true')
    parser.add_argument("--tsne", type=int, default=0)
    args = parser.parse_args()

    ratings_df = pd.read_csv("{_dir}/data/ratings_{_dir}.csv".format(_dir=args.dir))
    print(ratings_df.nunique())
    print("rating statistic:")
    print(ratings_df.groupby('rating')['userId'].count())

    user_count = ratings_df.groupby('userId')['rating'].count()
    user_std = ratings_df.groupby('userId')['rating'].std()
    item_count = ratings_df.groupby('itemId')['rating'].count()
    item_std = ratings_df.groupby('itemId')['rating'].std()
    print("mean of user_rating count =", user_count.mean())
    print("mean of user_rating std =", user_std.mean())
    print("mean of item_rating count =", item_count.mean())
    print("mean of item_rating std =", item_std.mean())
    
    #items_df = pd.read_csv("{_dir}/data/items_{_dir}.csv".format(_dir=args.dir))
    items_df = pd.read_csv("{_dir}/simple/rating0_result18.csv".format(_dir=args.dir))
    #items_df = pd.read_csv("random/pseudo/mf18_result.csv".format(_dir=args.dir))
    user_data, user_dict, item_data, item_dict, genre_data, rating_data = df_to_np(ratings_df, items_df)
    print("valid user number =", len(user_dict))
    print("valid item number =", len(item_dict))
    print("valid rating number =", len(rating_data))
    print("number of genre in valid items =", np.sum(genre_data, axis=0))
    print("average number of labels =", np.sum(genre_data) / genre_data.shape[0])
    
    if(args.random_pseudo):
        try:
            supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
        except:
            supmatrix_df = pd.read_csv("{_dir}/data/sup_{_dir}.csv".format(_dir=args.dir))
            supmatrix_save(supmatrix_df, item_dict, genre_data, "{_dir}/supmatrix.dat".format(_dir=args.dir))
            supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
        pseudo_genre_data = build_random_pseudo(supmatrix, genre_data, item_dict)
        print("number of random pseudo genre in valid items =", np.sum(pseudo_genre_data, axis=0))
        print("average number of random pseudo-labels =", np.sum(pseudo_genre_data) / pseudo_genre_data.shape[0])
    
    if(args.conf_pseudo):
        try:
            supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
        except:
            supmatrix_df = pd.read_csv("{_dir}/data/sup_{_dir}.csv".format(_dir=args.dir))
            supmatrix_save(supmatrix_df, item_dict, genre_data, "{_dir}/supmatrix.dat".format(_dir=args.dir))
            supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
        pseudo_genre_data = build_conf_pseudo(supmatrix, genre_data)
        print("number of conf pseudo genre in valid items =", np.sum(pseudo_genre_data, axis=0))
        print("average number of conf pseudo-labels =", np.sum(pseudo_genre_data) / pseudo_genre_data.shape[0])

    if(args.tsne & 1):
        mf = np.memmap("{_dir}/rating.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), len(user_dict)))
        mf = scale(mf, axis=0)
        tsne = TSNE(n_components=2, perplexity=150, verbose=1, n_jobs=8)
        tsne_result = tsne.fit_transform(mf[np.unique(np.sort(item_data))])
        plot_tsne(tsne_result, genre_data, "./random/tsne/mf")
     
    if(args.tsne & 2):
        try:
            embed_matrix = np.memmap("{_dir}/embedding.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 384))
        except:
            embed_save("{_dir}/mf_model.hdf5".format(_dir=args.dir), "{_dir}/embedding.dat".format(_dir=args.dir))
            embed_matrix = np.memmap("{_dir}/embedding.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 384))
        tsne = TSNE(n_components=2, perplexity=75, verbose=1, n_jobs=8)
        tsne_result = tsne.fit_transform(embed_matrix[np.unique(np.sort(item_data))])
        plot_tsne(tsne_result, genre_data, "./random/tsne/embed")

    if(args.tsne & 4):
        try:
            supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
        except:
            supmatrix_save(supmatrix_df, item_dict, genre_data, "{_dir}/supmatrix.dat".format(_dir=args.dir))
            supmatrix = np.memmap("{_dir}/supmatrix.dat".format(_dir=args.dir), dtype='float32', mode='r', shape=(len(item_dict), 18))
        tsne = TSNE(n_components=2, perplexity=30, verbose=1, n_jobs=8)
        tsne_result = tsne.fit_transform(supmatrix[np.unique(np.sort(item_data))])
        plot_tsne(tsne_result, genre_data, "./random/tsne/sup")
