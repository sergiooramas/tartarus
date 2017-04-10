from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import os
import sys
sys.path.insert(0, '../')
import common
import librosa
import h5py
from random import shuffle

SECONDS = 15
SR = 22050
HR = 1024
N_FRAMES = int(SECONDS * SR / float(HR)) # 10 seconds of audio
N_SAMPLES=1
N_BINS = 96
DATASET_NAME='multi2deT'
SPECTRO_FOLDER='spectro_MSD_cqt'
Y_PATH='class_250'
MAX_N_SCALER=300000

PATCH_MEAN = -0.0027567206  # Computed from 50k patches
PATCH_STD = 0.8436051       # Computed from 50k patches

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


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def sample_patch(mel_spec, n_frames):
    """Randomly sample a part of the mel spectrogram."""
    r_idx = np.random.randint(0, high=mel_spec.shape[0] - n_frames + 1)
    return mel_spec[r_idx:r_idx + n_frames]

def prepare_trainset(dataset_name, set_name, normalize=True, with_factors=True, scaler=None):
    if not os.path.exists(common.PATCHES_DIR):
        os.makedirs(common.PATCHES_DIR)
    f = h5py.File(common.PATCHES_DIR+'/patches_%s_%s_%sx%s.hdf5' % (set_name,dataset_name,N_SAMPLES,SECONDS),'w')
    spec_folder=common.SPECTRO_PATH+SPECTRO_FOLDER+"/"
    items = open(common.DATASETS_DIR+'/items_index_%s_%s.tsv' % (set_name, dataset_name)).read().splitlines()
    n_items = len(items) * N_SAMPLES
    print n_items
    x_dset = f.create_dataset("features", (n_items,1,N_FRAMES,N_BINS), dtype='f')
    i_dset = f.create_dataset("index", (n_items,), maxshape=(n_items,), dtype='S18')
    if with_factors:
        factors = np.load(common.DATASETS_DIR+'/item_factors_%s_%s_%s.npy' % (set_name, Y_PATH,dataset_name))
        y_dset = f.create_dataset("targets", (n_items,factors.shape[1]), dtype='f')
    k=0
    itemset = []
    itemset_index = []
    for t,track_id in enumerate(items):
        msd_folder = track_id[2]+"/"+track_id[3]+"/"+track_id[4]+"/"
        file = spec_folder+msd_folder+track_id+".pk"
        try:
            spec = pickle.load(open(file))
            spec = librosa.logamplitude(np.abs(spec) ** 2,ref_power=np.max).T
            for i in range(0,N_SAMPLES):
                try:
                    sample = sample_patch(spec,N_FRAMES)
                    x_dset[k,:,:,:] = sample.reshape(-1,sample.shape[0],sample.shape[1])
                    if with_factors:
                        y_dset[k,:] = factors[t]
                    i_dset[k] = track_id
                    itemset.append(track_id)
                    itemset_index.append(t)
                    k+=1
                except Exception as e:
                    print 'Error',e
                    print file
        except:
            pass
        if t%1000==0:
            print t

    print x_dset.shape

    # Normalize
    if normalize:
        print "Normalizing"
        block_step = 10000
        for i in range(0,len(itemset),block_step):
            x_block = f['features'][i:min(len(itemset),i+block_step)]
            x_norm = (x_block - PATCH_MEAN) / float(PATCH_STD)
            f['features'][i:min(len(itemset),i+block_step)] = x_norm

def prepare_testset(dataset_name):
    spec_folder=common.SPECTRO_PATH+SPECTRO_FOLDER+"/"
    test_folder=common.DATA_DIR+'/spectro_%s_testset/' % DATASET_NAME
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    items = open(common.DATASETS_DIR+'/items_index_test_%s.tsv' % dataset_name).read().splitlines()
    testset = []
    testset_index = []
    for t,track_id in enumerate(items):
        msd_folder = track_id[2]+"/"+track_id[3]+"/"+track_id[4]+"/"
        file = spec_folder+msd_folder+track_id+".pk"
        try:
            spec = pickle.load(open(file))
            if spec.shape[1] >= 322:
                spec = librosa.logamplitude(np.abs(spec) ** 2,ref_power=np.max).T
                pickle.dump(spec, open(test_folder+track_id+".pk","wb"))
                testset.append(track_id)
                testset_index.append(t)
                if t%1000==0:
                    print t
        except:
            print "no exist", file

if __name__ == '__main__':
    prepare_trainset(DATASET_NAME,"train", with_factors=False)
    prepare_trainset(DATASET_NAME,"val", with_factors=False)
    prepare_trainset(DATASET_NAME,"test", with_factors=False)
    # prepare_testset(DATASET_NAME)
