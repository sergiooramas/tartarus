from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd
import sys
import h5py
import pickle
sys.path.insert(0, '../')
import common

MAX_N_SCALER = 300000
FEATS_MEAN = -37.27
FEATS_VAR = 80.0

DATASET_NAME = "MSD-AG-S"
WINDOW = 15
N_PATCHES = 3

hdf5_file = common.PATCHES_DIR + "/patches_%s_%sx%s.hdf5" % (
    DATASET_NAME, N_PATCHES, WINDOW)
hdf5_newfile = common.PATCHES_DIR + "/patches_%s_%sx%s_norm.hdf5" % (
    DATASET_NAME, N_PATCHES, WINDOW)
print hdf5_file
f = h5py.File(hdf5_file,"r")
fw = h5py.File(hdf5_newfile,"w")
x_dset = fw.create_dataset("features", f['features'].shape, dtype='f')
y_dset = fw.create_dataset("targets", f['targets'].shape, dtype='f')
i_dset = fw.create_dataset("index", f['index'].shape, dtype='S18')
block_step = 10000
size = f['targets'].shape[0]
for i in range(0, size, block_step):
    x_block = f['features'][i:min(size, i+block_step)]
    x_norm = (x_block - FEATS_MEAN) / float(FEATS_VAR)
    print i
    fw['features'][i:min(size,i+block_step)] = x_norm
    fw['targets'][i:min(size,i+block_step)] = f['targets'][i:min(size,i+block_step)]
    fw['index'][i:min(size,i+block_step)] = f['index'][i:min(size,i+block_step)]
