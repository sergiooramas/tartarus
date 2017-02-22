import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, normalize
import sys

import common

FACT = 'pmi'  # nmf/pmi_wl/pmi_wp/pmi_wlp
DIM = 200
DATASET = 'MSDmm'
WINDOW = 1
NSAMPLES = 'all' #all
MAX_N_SCALER = 300000
N_PATCHES = 3


def scale(X, scaler=None, max_N=MAX_N_SCALER):
    shape = X.shape
    X.shape = (shape[0], shape[2] * shape[3])
    if not scaler:
        scaler = StandardScaler()
        N = pd.np.min([len(X), max_N])  # Limit the number of patches to fit
        scaler.fit(X[:N])
    X = scaler.transform(X)
    X.shape = shape
    return X, scaler


def load_X(args):
    data_path = '../data/patches_%s_%s/' % (DATASET, args.window)
    progress_update = 1

    data_files = glob.glob(os.path.join(data_path, "*.npy"))
    #songs_in = set(open(common.DATASETS_DIR+'/trainset_%s.tsv' %
    #                    (args.dataset)).read().splitlines())
    if len(data_files) == 0:
        raise ValueError("Error: Empty directory %s" % data_path)

    index_factors = set(open(common.DATASETS_DIR+'/items_index_train_'+DATASET+'.tsv').read().splitlines())

    data_files_in = []
    for file in data_files:
        filename = file[file.rfind('/')+1:-4]
        item_id, npatch = filename.split('_')
        if int(npatch) < args.npatches and item_id in index_factors:
            data_files_in.append(file)

    all_X = []
    songs_dataset = []
    X_mbatch = np.load(data_files_in[0])
    X = np.zeros((len(data_files_in),1,X_mbatch.shape[0],X_mbatch.shape[1]))
    for i, data_file in enumerate(data_files_in):
        song_id = data_file[data_file.rfind('/')+1:data_file.rfind('_')]
        X_mbatch = np.load(data_file)
        X[i,0,:,:] = X_mbatch
        #if len(all_Y) == 0:
        #    plt.imshow(X_mbatch,interpolation='nearest',aspect='equal')
        #    plt.show()
        #all_X.append(X_mbatch.reshape(-1,X_mbatch.shape[0],X_mbatch.shape[1]))
        songs_dataset.append(song_id)
        if i % progress_update == 0:
            sys.stdout.write("\rLoading Data: %.2f%%" % (100 * i / float(len(data_files_in))))
            sys.stdout.flush()
    sys.stdout.write("\rLoading Data: 100%")
    sys.stdout.flush()
    print "X data loaded"
    output_suffix_X = '%s_%sx%s' % (args.dataset,args.npatches,args.window)
    scaler_file=common.DATASETS_DIR+'/train_data/scaler_%s.pk' % output_suffix_X
    X,scaler = scale(X)
    pickle.dump(scaler,open(scaler_file,'wb'))
    X_file = common.DATASETS_DIR+'/train_data/X_train_'+output_suffix_X
    np.save(X_file,X)
    fw=open(common.DATASETS_DIR+'/train_data/index_train_'+output_suffix_X+'.tsv','w')
    fw.write("\n".join(songs_dataset))


def load_Y(args):
    progress_update = 1

    output_suffix_X = '%s_%sx%s' % (args.dataset,args.npatches,args.window)
    index_X=open(common.DATASETS_DIR+'/train_data/index_train_'+output_suffix_X+'.tsv').read().splitlines()
    song_factors=np.load(common.DATASETS_DIR+'/item_factors_%s_%s_%s.npy' % (args.fact,args.dim,args.dataset))
    song_index=open(common.DATASETS_DIR+'/items_index_%s.tsv' % (args.dataset)).read().splitlines()
    #print common.DATASETS_DIR+'/song_factors_%s_%s_%s.npy' % (args.fact,args.dim,args.dataset)
    print len(song_index)
    inv_song_index = dict()
    for i,song_id in enumerate(song_index):
        inv_song_index[song_id] = i

    # Read all data into memory (this might need to change if data too large)
    all_Y = []
    songs_dataset = []
    Y = np.zeros((len(index_X), int(args.dim)))
    for i, song_id in enumerate(index_X):
        # all_Y.append(song_factors[inv_song_index[song_id]])
        Y[i, :] = song_factors[inv_song_index[song_id]]
        if i % progress_update == 0:
            sys.stdout.write("\rLoading Data: %.2f%%" %
                             (100 * i / float(len(index_X))))
            sys.stdout.flush()
    sys.stdout.write("\rLoading Data: 100%")
    sys.stdout.flush()
    print "Y data loaded"
    output_suffix_Y = '%s_%s_%s_%sx%s' % (args.fact, args.dim, args.dataset,
                                          args.npatches, args.window)
    normalize(Y, copy=False)
    Y_file = common.DATASETS_DIR+'/train_data/Y_train_'+output_suffix_Y
    np.save(Y_file, Y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains the model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d',
                        '--dataset',
                        dest="dataset",
                        type=str,
                        help='Dataset name',
                        default=DATASET)
    parser.add_argument('-f',
                        '--fact',
                        dest="fact",
                        type=str,
                        help='Factorization method',
                        default=FACT)
    parser.add_argument('-dim',
                        '--dim',
                        dest="dim",
                        type=str,
                        help='Factors dimensions',
                        default=DIM)
    parser.add_argument('-w',
                        '--window',
                        dest="window",
                        type=str,
                        help='Patches window size in seconds',
                        default=WINDOW)
    parser.add_argument('-np',
                        '--npatches',
                        dest="npatches",
                        type=str,
                        help='Number of patches',
                        default=N_PATCHES)
    parser.add_argument('-x',
                        '--loadx',
                        dest="loadX",
                        help='Load X',
                        action='store_true',
                        default=False)
    parser.add_argument('-y',
                        '--loady',
                        dest="loadY",
                        help='Load Y',
                        action='store_true',
                        default=False)
    parser.add_argument('-all',
                        '--all',
                        dest="all_data",
                        help='All data, test and train set together',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    if args.loadX:
        load_X(args)
    if args.loadY:
        load_Y(args)
