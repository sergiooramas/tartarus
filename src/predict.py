"""Script to predict predictions from a trained model."""
from __future__ import print_function
from __future__ import division
import argparse
from joblib import Parallel, delayed
import logging
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import pandas as pd
from sklearn.externals import joblib
from keras import backend as K
import time
import pickle
from sklearn.metrics import accuracy_score, average_precision_score, coverage_error, label_ranking_average_precision_score, label_ranking_loss
from sklearn.preprocessing import StandardScaler, normalize
from keras.models import model_from_json
from scipy.sparse import csr_matrix
import json
import h5py
import common

# Files and extensions
DEFAULT_TRIM_COEFF = 0.15
TESTSET_FILE = common.DATASETS_DIR+'/testset_W2.tsv'
RESULTS_TSV = "results.tsv"
OUT_EXT='.pk'
RANDOM_SELECTION = False

AGGREGATE_DICT = {
    "max": np.max,
    "mean": np.mean,
    "median": np.median
}
DATASET_NAME='SUPER'

SR = 22050
HR = 1024
N_FRAMES = int(10 * SR / float(HR)) # 10 seconds of audio
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

def get_activations(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
    return activations

def predict_track(model, model_config, track_uid, agg_method, trim_coeff, spectro_folder="", with_metadata=False, metadata=[], rnd_selection=False, output_layer=-1):
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

    # Frames per patch
    n_frames = model.input_shape[2]  # shape should be (None, 1, frames, bins)

    # Get all patches into a numpy array

    try:
        patches = get_patches(mel_spec, n_frames, int(n_frames / 2))
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
        if output_layer == -1:       
            preds = model.predict(patches)
            pred = np.mean(preds,axis=0)
        else:
            #for i in range(0,18):
            pred = get_activations(model, output_layer, patches)[0][0]
            #    print(i,len(preds[0]),len(preds[0][0]))
    except Exception,e:
        pred = []
        print(str(e))
        print('Error predicting track')
    return pred

def predict_track_metadata(model, metadata=[], output_layer=-1):
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
        if output_layer == -1:       
            pred = model.predict(patches_meta)            
        else:
            pred = get_activations(model, output_layer, patches_meta)[0]

    except Exception,e:
        pred = []
        print(str(e))
        print('Error predicting track')
    return pred[0]

def obtain_predictions(model_config, dataset, model_id, trim_coeff=0.15, model=False, spectro_folder="", with_metadata=False, only_metadata=False, metadata_source='rovi', set_name="test", rnd_selection=False, output_layer=-1, with_patches=False, pred_dataset=""):
    """Evaluates the model across the whole dataset."""
    # Read the pre-trained model
    agg_method="mean"
    print(model_id)
    if not model:
        model = read_model(model_config)

    predictions = dict()
    params = eval(model_config["dataset_settings"][0])
    predictions=[]
    predictions_index=[]

    if pred_dataset == "":
        dataset_name = params["dataset"]
    else:
        dataset_name = pred_dataset
    if with_metadata:
        if 'sparse' not in params:
            params['sparse'] = True
        #all_X_meta = np.load(common.TRAINDATA_DIR+'/X_test_%s_%s.npy' % (metadata_source,dataset_name))
        if 'w2v' in metadata_source:
            sequence_length = eval(model_config["model_arch"][0])["sequence_length"]
            all_X_meta = np.load(common.TRAINDATA_DIR+'/X_%s_%s_%s.npy' % (set_name,metadata_source,dataset_name))[:,:int(sequence_length)]
        elif 'model' in metadata_source or not params['sparse']:
            all_X_meta = np.load(common.TRAINDATA_DIR+'/X_%s_%s_%s.npy' % (set_name,metadata_source,dataset_name))
            print ("meta1",all_X_meta.shape)
            print (metadata_source)
        else:
            all_X_meta = load_sparse_csr(common.TRAINDATA_DIR+'/X_%s_%s_%s.npz' % (set_name,metadata_source,dataset_name)).toarray()

        if 'meta-suffix2' in params:
            metadata_source2 = params['meta-suffix2']
            if 'w2v' in metadata_source2:
                sequence_length = eval(model_config["model_arch"][0])["sequence_length"]
                all_X_meta2 = np.load(common.TRAINDATA_DIR+'/X_%s_%s_%s.npy' % (set_name,metadata_source2,dataset_name))[:,:int(sequence_length)]
            elif 'model' in metadata_source or not params['sparse']:
                all_X_meta2 = np.load(common.TRAINDATA_DIR+'/X_%s_%s_%s.npy' % (set_name,metadata_source2,dataset_name))
                print ("meta2",all_X_meta2.shape)
                print (metadata_source2)
            else:
                all_X_meta2 = load_sparse_csr(common.TRAINDATA_DIR+'/X_%s_%s_%s.npz' % (set_name,metadata_source2,dataset_name)).toarray()

        if 'meta-suffix3' in params:
            metadata_source3 = params['meta-suffix3']
            if 'w2v' in metadata_source3:
                sequence_length = eval(model_config["model_arch"][0])["sequence_length"]
                all_X_meta3 = np.load(common.TRAINDATA_DIR+'/X_%s_%s_%s.npy' % (set_name,metadata_source3,dataset_name))[:,:int(sequence_length)]
            elif 'model' in metadata_source or not params['sparse']:
                all_X_meta3 = np.load(common.TRAINDATA_DIR+'/X_%s_%s_%s.npy' % (set_name,metadata_source3,dataset_name))
                print ("meta3",all_X_meta3.shape)
                print (metadata_source3)
            else:
                all_X_meta3 = load_sparse_csr(common.TRAINDATA_DIR+'/X_%s_%s_%s.npz' % (set_name,metadata_source3,dataset_name)).toarray()

        index_meta = open(common.DATASETS_DIR+'/items_index_%s_%s.tsv' % (set_name,dataset_name)).read().splitlines()
        index_meta_inv = dict()
        for i,item in enumerate(index_meta):
            index_meta_inv[item] = i

    if with_patches:
        hdf5_file = common.PATCHES_DIR+"/patches_%s_%s_%sx%s.hdf5" % (set_name,dataset_name,params["npatches"],params["window"])
        f = h5py.File(hdf5_file,"r")
        block_step = 100
        N_train = f['features'].shape[0]
        for i in range(0,N_train,block_step):
            x_block = f['features'][i:min(N_train,i+block_step)]
            #index_block = f['index'][i:min(N_train,i+block_step)]
            if output_layer == -1:    
                preds = model.predict(x_block)
            else:
                preds = get_activations(model, output_layer, x_block)[0]
            predictions.extend([prediction for prediction in preds])
            print(i)
        predictions_index = open(common.DATASETS_DIR+'/items_index_%s_%s.tsv' % (set_name,dataset_name)).read().splitlines()
    elif only_metadata:
        block_step = 1000
        N_train = all_X_meta.shape[0]
        for i in range(0,N_train,block_step):
            if 'meta-suffix3' in params:
                x_block = [all_X_meta[i:min(N_train,i+block_step)],all_X_meta2[i:min(N_train,i+block_step)],all_X_meta3[i:min(N_train,i+block_step)]]
                print(len(x_block))
            elif 'meta-suffix2' in params:
                x_block = [all_X_meta[i:min(N_train,i+block_step)],all_X_meta2[i:min(N_train,i+block_step)]]
                print(len(x_block))
            else:
                x_block = all_X_meta[i:min(N_train,i+block_step)]
            if output_layer == -1:    
                preds = model.predict(x_block)
            else:
                preds = get_activations(model, output_layer, x_block)[0]
            predictions.extend([prediction for prediction in preds])
            print(i)
        predictions_index = index_meta
    else:
        for i, track_uid in enumerate(dataset):
            if with_metadata:
                #if only_metadata:
                #    pred = predict_track_metadata(model, all_X_meta[index_meta_inv[track_uid]], output_layer=output_layer)
                #else:
                pred = predict_track(model, model_config, track_uid, agg_method,
                                     trim_coeff, spectro_folder=spectro_folder, with_metadata=True, metadata=all_X_meta[index_meta.index(track_uid)], rnd_selection=rnd_selection, output_layer=output_layer)
            else:
                pred = predict_track(model, model_config, track_uid, agg_method,
                                     trim_coeff, spectro_folder=spectro_folder, rnd_selection=rnd_selection, output_layer=output_layer)
            if pred != []:
                predictions.append(pred)
                predictions_index.append(track_uid)
            if i%100==0:
                print(i)
    suffix = ''        
    #if rnd_selection:
    #    suffix = '_rnd'
    if spectro_folder != '':
        suffix += '_' + spectro_folder
    predictions = np.asarray(predictions)
    if output_layer != -1:
        if not os.path.isdir(common.TRAINDATA_DIR):
            os.makedirs(common.TRAINDATA_DIR)
        np.save(common.TRAINDATA_DIR+'/X_%s_%s-pred_%s_%s.npy' % (set_name,model_id,output_layer,dataset_name), predictions)
        fw=open(common.TRAINDATA_DIR+'/items_index_%s_%s-pred_%s_%s.tsv' % (set_name,model_id,output_layer,dataset_name),'w')
        fw.write('\n'.join(predictions_index))
        fw.close()
    else:
        if set_name == "test":
            if pred_dataset != "":
                suffix = suffix+"_"+pred_dataset
            if not os.path.isdir(common.PREDICTIONS_DIR):
                os.makedirs(common.PREDICTIONS_DIR)
            np.save(common.PREDICTIONS_DIR+'/pred_%s%s' % (model_id,suffix),predictions)
            fw=open(common.PREDICTIONS_DIR+'/index_pred_%s%s.tsv' % (model_id,suffix),'w')
            fw.write('\n'.join(predictions_index))
            fw.close()
    print(len(predictions))
    print(len(predictions_index))
    return predictions, predictions_index


def predict(model_id, trained_tsv=common.DEFAULT_TRAINED_MODELS_FILE, test_file="", pred_dataset="", spectro_folder="", set_name="test", rnd_selection=False, output_layer=-1, with_patches=False):
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
    print("Model settings loaded")
    if test_file == "":    
        if model_settings['only_metadata']:
            dataset_tsv = common.DATASETS_DIR+'/items_index_%s_%s.tsv' % (set_name,model_settings['dataset'])
        else:
            dataset_tsv = common.DATASETS_DIR+'/items_index_%s_%s.tsv' % (set_name,model_settings['dataset'])
    else:
        dataset_tsv = common.DATASETS_DIR+'/%s' % test_file

    # Read dataset
    f=open(dataset_tsv)
    dataset = f.read().splitlines()
    print(len(dataset))
    predictions, predictions_index = obtain_predictions(model_config, dataset, model_id, spectro_folder=spectro_folder, with_metadata=model_settings['with_metadata'], only_metadata=model_settings['only_metadata'], metadata_source=model_settings['meta-suffix'],set_name=set_name, rnd_selection=rnd_selection, output_layer=output_layer, with_patches=with_patches, pred_dataset=pred_dataset)
    print('Factors created')


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
    parser.add_argument('-pd',
                        '--pred_dataset',
                        dest="pred_dataset",
                        type=str,
                        help='File with metadata X to predict',
                        default="")
    parser.add_argument('-sf',
                        '--spectro_folder',
                        dest="spectro_folder",
                        type=str,
                        help='Folder where the spectrograms are',
                        default="")
    parser.add_argument('-s',
                        '--set_name',
                        dest="set_name",
                        help='Specify the set prefix for prediction',
                        default="test")
    parser.add_argument('-rnd',
                        '--randomsel',
                        dest="rnd_selection",
                        action='store_true',
                        help='Select only one patch from the middle',
                        default=False)
    parser.add_argument('-l',
                        '--layer',
                        dest="output_layer",
                        type=int,
                        help='Output layer (-1 for normal output)',
                        default=-1)
    parser.add_argument('-p',
                        '--patches',
                        dest="with_patches",
                        action='store_true',
                        help='Use h5py patches file for test',
                        default=False)

    # Setup logger
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

    # Log time
    start_time = time.time()

    # Parse arguments and call main process
    args = parser.parse_args()
    predict(args.model_id, args.trained_tsv, args.test_file, args.pred_dataset, args.spectro_folder, args.set_name, args.rnd_selection, args.output_layer, args.with_patches)

    # Finish
    logging.info("Done! Took %.2f seconds" % (time.time() - start_time))
