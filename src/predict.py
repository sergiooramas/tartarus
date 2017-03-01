"""Script to predict factors from a trained model."""
from __future__ import print_function
from __future__ import division
import argparse
from joblib import Parallel, delayed
import logging
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
import time
import glob
import pickle
import pdb
from sklearn.metrics import accuracy_score, average_precision_score, coverage_error, label_ranking_average_precision_score, label_ranking_loss
from sklearn.preprocessing import StandardScaler, normalize
from keras.models import model_from_json
from scipy.sparse import csr_matrix
import json
import common
#import librosa
#import theano

# Files and extensions
DEFAULT_TRIM_COEFF = 0.15
TESTSET_FILE = common.DATASETS_DIR+'/testset_W2.tsv'
RESULTS_TSV = "results.tsv"
OUT_EXT='.pk'
RANDOM_SELECTION = False
#SEED_FACTORS_FOLDER='factors_2016_07_05/seed_factors/'

AGGREGATE_DICT = {
    "max": np.max,
    "mean": np.mean,
    "median": np.median
}
DATASET_NAME='W2_10_all_pmi_wl_200'


SR = 22050
HR = 1024
N_FRAMES = int(10 * SR / float(HR)) # 10 seconds of audio
#DATASET_FOLDER= 'dataset_initial_2_33/'
SPEC_FOLDER=common.DATA_DIR+'/spectro_W2/'

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def load_model(model_file):
    """Loads the model from the given model file."""
    with open(model_file) as f:
        json_str = json.load(f)
    return model_from_json(json_str)

def read_model(model_config):
    """Reads the model."""
    model_json = os.path.join(common.MODELS_DIR, model_config["model_id"][0],
                              model_config["model_id"][0] + common.MODEL_EXT)
    model_weights = os.path.join(common.MODELS_DIR, model_config["model_id"][0],
                                 model_config["model_id"][0] + common.WEIGHTS_EXT)
    model = load_model(model_json)
    model.load_weights(model_weights)
    return model

def get_patches(spec, frame_length, hop_length):
    """Get the set of patches from the mel_spectrogram.

    Parameters
    ----------
    spec: np.array(N, F)
        N observations and F mel bins.
    frame_length: int
        Length of the patches (in frames).
    hop_length: int
        Length of the hop size (in frames).

    Returns
    -------
    spec_patches: np.array(M, frame_length, F)
        The patches of the mel spectrogram.
    """
    bins = spec.shape[1]

    # Get number of frames (this may truncate spectrogram)
    n_frames = 1 + int((spec.shape[0] - frame_length) / hop_length)
    if n_frames < 1:
        raise ValueError('Buffer is too short (n={:d}) for frame_length'
                         '={:d}'.format(spec.shape[0], frame_length))

    # Get the actual patches
    spec_patches = as_strided(spec, shape=(n_frames, frame_length, bins),
                              strides=(spec.itemsize * bins * hop_length,
                                       spec.itemsize * bins,
                                       spec.itemsize))
    return spec_patches

def read_melspec(track_uid, data_dir, ext):
    file = data_dir + track_uid + ext
    spec = pickle.load(open(file))
    #spec = librosa.logamplitude(np.abs(spec) ** 2,ref_power=np.max).T
    return spec


def predict_track(model, model_config, track_uid, agg_method, trim_coeff, rnd_selection, spectro_folder="", with_metadata=False, metadata=[]):
    """Predicts piano for a given track.

    Parameters
    ----------
    model: keras.model
        Model with the pre-trained weights.
    model_config: dict
        Configuration parameters of the model.
    track_uid: str
        Track UID to predict piano detection from.
    agg_method: str
        Aggregation method (possible values `AGGREGATE_DICT.keys()`).
    trim_coeff: float
        Percentage of track to trim from beginning and end.

    Returns
    -------
    prediction: float
        Probablity of having piano in the given track.
    """
    # Get the mel spectrogram
    if spectro_folder == "":
        SPEC_FOLDER=common.DATA_DIR+'/spectro_%s_testset/' % eval(model_config["dataset_settings"][0])["dataset"]
    else:
        SPEC_FOLDER=common.DATA_DIR+'/%s/' % spectro_folder
    mel_spec = read_melspec(track_uid, data_dir=SPEC_FOLDER, ext=OUT_EXT)

    # Trim spectrogram around borders
    assert trim_coeff < 0.5 and trim_coeff >= 0
    trim_frames = int(mel_spec.shape[0] * trim_coeff)
    if trim_frames != 0:
        mel_spec = mel_spec[trim_frames:-trim_frames, :]

    #if model_config["norm"][0]:
    #    mel_spec = normalize(mel_spec)

    # Frames per patch
    #n_frames = model.input_shape[2]  # shape should be (None, 1, frames, bins)
    n_frames = 322

    # Get all patches into a numpy array

    try:
        patches = get_patches(mel_spec, n_frames, n_frames / 2)
        patches = patches.copy()  # Avoid memory overlap
        patches = patches.reshape(-1, 1, n_frames, patches.shape[-1])

        if model_config["whiten"][0]:
            scaler_file = model_config["whiten_scaler"][0]            
            scaler = joblib.load(common.TRAINDATA_DIR+'/'+scaler_file[scaler_file.rfind('/')+1:])
            patches, _ = common.preprocess_data(patches, scaler)
        if rnd_selection:
            #pred = random.sample(preds,1)[0]
            patches = np.asarray([patches[int(patches.shape[0]/2),:,:,:]])
        if with_metadata:
            patches_meta = np.zeros((patches.shape[0],metadata.shape[0]))
            for i,patch in enumerate(patches):
                patches_meta[i,:] = metadata[:]
            patches = [patches,patches_meta]
        # Make predictions           
        preds = model.predict(patches)
        #imd_f = theano.function([model.input],
        #                model.nodes[-1].get_output(train=False))
        #new_preds = imd_f(patches)
        #print(new_preds.shape)
        #print(new_preds)
        # Aggregate
        #pred = AGGREGATE_DICT[agg_method](preds)
        #pdb.set_trace()
        pred = np.mean(preds,axis=0)
    except Exception,e:
        pred = []
        print(str(e))
        print('Error predicting track')
    return pred

def predict_track_metadata(model, metadata=[]):
    """Predicts piano for a given track.

    Parameters
    ----------
    model: keras.model
        Model with the pre-trained weights.
    model_config: dict
        Configuration parameters of the model.
    track_uid: str
        Track UID to predict piano detection from.
    agg_method: str
        Aggregation method (possible values `AGGREGATE_DICT.keys()`).
    trim_coeff: float
        Percentage of track to trim from beginning and end.

    Returns
    -------
    prediction: float
        Probablity of having piano in the given track.
    """
    try:
        # Make predictions
        patches_meta = np.zeros((1,metadata.shape[0]))
        patches_meta[0,:] = metadata[:]            
        pred = model.predict(patches_meta)

    except Exception,e:
        pred = []
        print(str(e))
        print('Error predicting track')
    return pred[0]

def obtain_factors(model_config, dataset, model_id, trim_coeff=0.15, model=False, spectro_folder="", with_metadata=False, only_metadata=False, metadata_source='rovi', on_trainset=False):
    """Evaluates the model across the whole dataset."""
    # Read the pre-trained model
    agg_method="mean"
    rnd_selection = False
    if not model:
        model = read_model(model_config)
    factors = dict()
    #with_metadata=True
    factors=[]
    factors_index=[]
    print(len(dataset))
    if with_metadata:
        dataset_name = eval(model_config["dataset_settings"][0])["dataset"]
        #all_X_meta = np.load(common.DATASETS_DIR+'/train_data/X_test_%s_%s.npy' % (metadata_source,dataset_name))
        if 'w2v' in metadata_source:
            sequence_length = eval(model_config["model_arch"][0])["sequence_length"]
            if on_trainset:
                all_X_meta = np.load(common.DATASETS_DIR+'/train_data/X_train_%s_%s.npy' % (metadata_source,dataset_name))[:,:int(sequence_length)]
            else:
                all_X_meta = np.load(common.DATASETS_DIR+'/train_data/X_test_%s_%s.npy' % (metadata_source,dataset_name))[:,:int(sequence_length)]
        elif 'model' in metadata_source:
            if on_trainset:
                all_X_meta = np.load(common.DATASETS_DIR+'/train_data/X_train_%s_%s.npy' % (metadata_source,dataset_name))
            else:
                all_X_meta = np.load(common.DATASETS_DIR+'/train_data/X_test_%s_%s.npy' % (metadata_source,dataset_name))
        else:
            if on_trainset:
                all_X_meta = load_sparse_csr(common.DATASETS_DIR+'/train_data/X_train_%s_%s.npz' % (metadata_source,dataset_name)).toarray()
            else:
                all_X_meta = load_sparse_csr(common.DATASETS_DIR+'/train_data/X_test_%s_%s.npz' % (metadata_source,dataset_name)).toarray()

        if on_trainset:
            index_meta = open(common.DATASETS_DIR+'/items_index_train_%s.tsv' % (dataset_name)).read().splitlines()
        else:
            index_meta = open(common.DATASETS_DIR+'/items_index_test_%s.tsv' % (dataset_name)).read().splitlines()
        index_meta_inv = dict()
        for i,item in enumerate(index_meta):
            index_meta_inv[item] = i

    for i, track_uid in enumerate(dataset):
        if with_metadata:
            if only_metadata:
                pred = predict_track_metadata(model, all_X_meta[index_meta_inv[track_uid]])
            else:
                pred = predict_track(model, model_config, track_uid, agg_method,
                                     trim_coeff, rnd_selection, spectro_folder=spectro_folder, with_metadata=True, metadata=all_X_meta[index_meta.index(track_uid)])
        else:
            pred = predict_track(model, model_config, track_uid, agg_method,
                                 trim_coeff, rnd_selection, spectro_folder=spectro_folder)
        if pred != []:
            factors.append(pred)
            factors_index.append(track_uid)
        if i%1000==0:
            print(i)
    suffix = ''        
    if rnd_selection:
        suffix = '_rnd'
    if spectro_folder != '':
        suffix += '_' + spectro_folder
    factors = np.asarray(factors)
    if on_trainset:
        np.save(common.TRAINDATA_DIR+'/X_train_%s-pred_%s.npy' % (model_id,dataset_name), factors)
    else:
        np.save(common.FACTORS_DIR+'/factors_%s%s' % (model_id,suffix),factors)
        fw=open(common.FACTORS_DIR+'/index_factors_%s%s.tsv' % (model_id,suffix),'w')
        fw.write('\n'.join(factors_index))
        fw.close()
    return factors, factors_index


def predict(model_id, trained_tsv=common.DEFAULT_TRAINED_MODELS_FILE, test_file="", spectro_folder="", on_trainset=False):
    """Main process to perform the training.

    Parameters
    ----------
    model_id: str
        Model identifier to use for the evaluate.
    trained_tsv: str
        Path to the tsv with trained models info.
    dataset_tsv: str
        Path to the tsv with the dataset for evaluation.
    results_tsv: str
        Path to the tsv to store results.
    trim_coeff: float
        Percentage of track to trim from beginning and end.
    agg_method: str
        Aggregation method, must be a key in `AGGREGATE_DICT`.
    """
    # Get model configuration
    trained_models = pd.read_csv(trained_tsv, sep='\t')
    model_config = trained_models[trained_models["model_id"] == model_id]
    if model_config.empty:
        raise ValueError("Can't find the model %s in %s" %
                         (model_id, trained_tsv))
    model_config = model_config.to_dict(orient="list")
    model_settings=eval(model_config['dataset_settings'][0])

    if test_file == "":    
        if on_trainset:
            dataset_tsv = common.DATASETS_DIR+'/items_index_train_%s.tsv' % model_settings['dataset']
        else:
            dataset_tsv = common.DATASETS_DIR+'/items_index_test_%s.tsv' % model_settings['dataset']
    else:
        dataset_tsv = common.DATASETS_DIR+'/%s' % test_file

    # Read dataset
    f=open(dataset_tsv)
    dataset = f.read().splitlines()
    factors, factors_index = obtain_factors(model_config, dataset, model_id, spectro_folder=spectro_folder, with_metadata=model_settings['with_metadata'], only_metadata=model_settings['only_metadata'], metadata_source=model_settings['meta-suffix'],on_trainset=on_trainset)
    print('Factors created')
    #do_eval(model_id)
    #evaluate_factors(factors,eval(model_config['dataset_settings'][0]),model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluates the model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest="model_id",
                        type=str,
                        help='Identifier of the Model to evaluate')
    parser.add_argument('-t',
                        '--trained_tsv',
                        dest="trained_tsv",
                        type=str,
                        help='Path to the results tsv file',
                        default=common.DEFAULT_TRAINED_MODELS_FILE)
    parser.add_argument('-tf',
                        '--test_file',
                        dest="test_file",
                        type=str,
                        help='File with a list of songs to predict',
                        default="")
    parser.add_argument('-sf',
                        '--spectro_folder',
                        dest="spectro_folder",
                        type=str,
                        help='Folder where the spectrograms are',
                        default="")
    parser.add_argument('-ts',
                        '--trainset',
                        dest="on_trainset",
                        action='store_true',
                        help='Predict on train set',
                        default=False)

    # Setup logger
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

    # Log time
    start_time = time.time()

    # Parse arguments and call main process
    args = parser.parse_args()
    predict(args.model_id, args.trained_tsv, args.test_file, args.spectro_folder, args.on_trainset)

    # Finish
    logging.info("Done! Took %.2f seconds" % (time.time() - start_time))
