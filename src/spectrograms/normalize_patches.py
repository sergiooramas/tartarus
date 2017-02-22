from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd
import sys
sys.path.insert(0, '../')
import common
import h5py
import pickle

MAX_N_SCALER=300000

def scale(X, scaler=None, max_N=MAX_N_SCALER):
    shape = X.shape
    X.shape = (shape[0], shape[2] * shape[3])
    if not scaler:
        scaler = StandardScaler()
        N = min([len(X), max_N])
        scaler.fit(X[:N])
    X = scaler.transform(X)
    X.shape = shape
    return X, scaler


DATASET_NAME = "MSD-AG-S"
WINDOW = 15

hdf5_file = common.PATCHES_DIR+"/patches_%s_%s.hdf5" % (DATASET_NAME,WINDOW)
f = h5py.File(hdf5_file,"r+")
block_step = 10000
size = f['targets'].shape[0]
scaler = None
for i in range(0,size,block_step):
    x_block = f['features'][i:min(size,i+block_step)]
    x_norm, scaler = scale(x_block,scaler)
    print i
    f['features'][i:min(size,i+block_step)] = x_norm

scaler_file=common.DATASETS_DIR+'/train_data/scaler_%s_%s.pk' % (DATASET_NAME,WINDOW)
pickle.dump(scaler,open(scaler_file,'wb'))
