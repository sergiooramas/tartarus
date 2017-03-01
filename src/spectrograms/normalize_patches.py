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


DATASET_NAME = "MSD"
WINDOW = 15
N_PATCHES = 1

hdf5_file = common.PATCHES_DIR+"/patches_%s_%sx%s.hdf5" % (DATASET_NAME,N_PATCHES,WINDOW)
hdf5_newfile = common.PATCHES_DIR+"/patches_%s_%sx%s_norm.hdf5" % (DATASET_NAME,N_PATCHES,WINDOW)
print hdf5_file
f = h5py.File(hdf5_file,"r")
fw = h5py.File(hdf5_newfile,"w")
x_dset = fw.create_dataset("features", f['features'].shape, dtype='f')
y_dset = fw.create_dataset("targets", f['targets'].shape, dtype='f')
i_dset = fw.create_dataset("index", f['index'].shape, dtype='S18')
block_step = 10000
size = f['targets'].shape[0]
scaler = None
for i in range(0,size,block_step):
    x_block = f['features'][i:min(size,i+block_step)]
    x_norm, scaler = scale(x_block,scaler)
    print i
    fw['features'][i:min(size,i+block_step)] = x_norm
    fw['targets'][i:min(size,i+block_step)] = f['targets'][i:min(size,i+block_step)]
    fw['index'][i:min(size,i+block_step)] = f['index'][i:min(size,i+block_step)]
    
scaler_file=common.DATASETS_DIR+'/train_data/scaler_%s_%sx%s.pk' % (DATASET_NAME,N_PATCHES,WINDOW)
pickle.dump(scaler,open(scaler_file,'wb'))
