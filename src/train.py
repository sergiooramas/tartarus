from __future__ import print_function
import argparse
from collections import OrderedDict
import json
import os
import logging
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten, Permute, Lambda, Input, merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.utils.io_utils import HDF5Matrix
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from scipy.sparse import csr_matrix
#from keras.utils.visualize_util import plot
from keras.optimizers import SGD, Adam
from sklearn.metrics import r2_score
import numpy as np
import theano.tensor as tt
import pandas as pd
import random
import common
import models
from predict import obtain_factors
from eval import do_eval
import h5py

import keras.backend as K
import theano

SR = 22050
HR = 1024


class Config(object):
    """Configuration for the training process."""
    def __init__(self, params, normalize=False, whiten=True):
        self.model_id = common.get_next_model_id()
        self.norm = normalize
        self.whiten = whiten
        self.x_path = '%s_%sx%s' % (params['dataset']['dataset'],params['dataset']['npatches'],params['dataset']['window'])
        self.y_path = '%s_%s_%s' % (params['dataset']['fact'],params['dataset']['dim'],params['dataset']['dataset'])
        self.dataset_settings = params['dataset']
        self.training_params = params['training']
        self.model_arch = params['cnn']
        self.predicting_params = params['predicting']

    def get_dict(self):
        object_dict = self.__dict__
        first_key = "model_id"
        conf_dict = OrderedDict({first_key: object_dict[first_key]})
        conf_dict.update(object_dict)
        return conf_dict


def _squared_magnitude(x):
    return tt.sqr(x).sum(axis=-1)


def _magnitude(x):
    return tt.sqrt(tt.maximum(_squared_magnitude(x), np.finfo(x.dtype).tiny))


def cosine(x, y):
    return tt.clip((1 - (x * y).sum(axis=-1) /
                    (_magnitude(x) * _magnitude(y))) / 2, 0, 1)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def build_model(config):
    """Builds the cnn."""
    params = config.model_arch
    get_model = getattr(models, 'get_model_'+str(params['architecture']))
    model = get_model(params)
    #model = model_kenun.build_convnet_model(params)
    # Learning setup
    t_params = config.training_params
    sgd = SGD(lr=t_params["learning_rate"], decay=t_params["decay"],
              momentum=t_params["momentum"], nesterov=t_params["nesterov"])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    optimizer = eval(t_params['optimizer'])
    metrics = ['mean_squared_error']
    if config.model_arch["final_activation"] == 'softmax':
        metrics.append('categorical_accuracy')
    if t_params['loss_func'] == 'cosine':
        loss_func = eval(t_params['loss_func'])
    else:
        loss_func = t_params['loss_func']
    model.compile(loss=loss_func, optimizer=optimizer,metrics=metrics)

    return model

def load_data_preprocesed(params, X_path, Y_path, dataset, val_percent, test_percent, n_samples, with_metadata=False, only_metadata=False, metadata_source='rovi'):
    factors = np.load(common.DATASETS_DIR+'/item_factors_train_'+Y_path+'.npy') # OJO remove S
    index_factors = open(common.DATASETS_DIR+'/items_index_train_'+dataset+'.tsv').read().splitlines()
    if not only_metadata:
        all_X = np.load(common.DATASETS_DIR+'/train_data/X_train_'+X_path+'.npy')
        index_train = open(common.DATASETS_DIR+'/train_data/index_train_%s.tsv' % (X_path)).read().splitlines()
        all_Y = np.zeros((len(index_train),factors.shape[1]))
        index_factors_inv = dict()
        for i,item in enumerate(index_factors):
            index_factors_inv[item] = i
        for i,item in enumerate(index_train):
            all_Y[i,:] = factors[index_factors_inv[item]]
    else:
        all_Y = factors
    if with_metadata:
        if 'w2v' in metadata_source:
            all_X_meta = np.load(common.DATASETS_DIR+'/train_data/X_train_%s_%s.npy' % (metadata_source,dataset))[:,:int(params['cnn']['sequence_length'])]
        elif 'model' in metadata_source or not params['dataset']['sparse']:
            all_X_meta = np.load(common.DATASETS_DIR+'/train_data/X_train_%s_%s.npy' % (metadata_source,dataset))
        else:
            all_X_meta = load_sparse_csr(common.DATASETS_DIR+'/train_data/X_train_%s_%s.npz' % (metadata_source,dataset)).todense()

        all_X_in_meta = all_X = all_X_meta
        print(all_X_in_meta.shape)

    print(all_X.shape)
    print(all_Y.shape)
    if n_samples != 'all':
        n_samples = int(n_samples)
        all_X = all_X[:n_samples]
        all_Y = all_Y[:n_samples]
        if with_metadata:
            all_X_in_meta = all_X_in_meta[:n_samples]

    if params['training']['normalize_y'] == True:
        normalize(all_Y,copy=False)
    N = all_Y.shape[0]
    train_percent = 1 - val_percent - test_percent
    N_train = int(train_percent * N)
    N_val = int(val_percent * N)
    logging.debug("Training data points: %d" % N_train)
    logging.debug("Validation data points: %d" % N_val)
    logging.debug("Test data points: %d" % (N - N_train - N_val))

    if params['training']["val_from_file"]:
        Y_val = np.load(common.DATASETS_DIR+'/item_factors_val_'+Y_path+'.npy')
        Y_test = np.load(common.DATASETS_DIR+'/item_factors_test_'+Y_path+'.npy') #!!! OJO remove S from trainS
        X_val = np.load(common.DATASETS_DIR+'/train_data/X_val_%s_%s.npy' % (metadata_source,dataset))
        X_test = np.load(common.DATASETS_DIR+'/train_data/X_test_%s_%s.npy' % (metadata_source,dataset))
        X_train = all_X
        Y_train = all_Y
    else:
        if not only_metadata:
            # Slice data
            X_train = all_X[:N_train]
            X_val = all_X[N_train:N_train + N_val]
            X_test = all_X[N_train + N_val:]
        Y_train = all_Y[:N_train]
        Y_val = all_Y[N_train:N_train + N_val]
        Y_test = all_Y[N_train + N_val:]

        if with_metadata:
            if only_metadata:
                X_train = all_X_in_meta[:N_train]
                X_val = all_X_in_meta[N_train:N_train + N_val]
                X_test = all_X_in_meta[N_train + N_val:]
            else:
                X_train = [X_train,all_X_in_meta[:N_train]]
                X_val = [X_val,all_X_in_meta[N_train:N_train + N_val]]
                X_test = [X_test,all_X_in_meta[N_train + N_val:]]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def single_file_generator(params, y_path):
    items_index = open(common.DATASETS_DIR+'/items_index_train_'+params['dataset']['dataset']+'.tsv').read().splitlines()
    factors = np.load(common.DATASETS_DIR+'/item_factors_'+y_path+'.npy')
    f = h5py.File(common.PATCHES_DIR+"/patches_train_%s_15.hdf5" % params['dataset']['dataset'],"r")
    patches = f["patches"]
    batch_size = params["training"]["n_minibatch"]
    items_list = range(int(len(items_index)*0.8))
    #random.shuffle(items_list)
    while 1:
        for i in range(0,len(items_list),batch_size):
            if i+batch_size <= len(items_list):
                items_in_batch = items_list[i:i+batch_size]
            else:
                items_in_batch = items_list[len(items_list)-batch_size:]
            x = np.zeros((batch_size,1,params['cnn']['n_frames'],params['cnn']['n_mel']))
            y = []
            x = patches[items_in_batch]
            x = x.reshape(-1,1,322,96)
            for i,index in enumerate(items_in_batch):
                y.append(factors[index])
            y = np.asarray(y)
            if params['training']['normalize_y'] == True:
                normalize(y,copy=False)
            yield (np.asarray(x), np.asarray(y))

def load_data_hf5(params,val_percent, test_percent):
    hdf5_file = common.PATCHES_DIR+"/patches_train_%s_%s.hdf5" % (params['dataset']['dataset'],params['dataset']['window'])
    f = h5py.File(hdf5_file,"r")
    N = f["targets"].shape[0]
    f.close()
    train_percent = 1 - val_percent - test_percent
    N_train = int(train_percent * N)
    N_val = int(val_percent * N)
    X_train = HDF5Matrix(hdf5_file, 'features', start=0, end=N_train)
    Y_train = HDF5Matrix(hdf5_file, 'targets', start=0, end=N_train)
    X_val = HDF5Matrix(hdf5_file, 'features', start=N_train, end=N_train+N_val)
    Y_val = HDF5Matrix(hdf5_file, 'targets', start=N_train, end=N_train+N_val)
    X_test = HDF5Matrix(hdf5_file, 'features', start=N_train+N_val, end=N)
    Y_test = HDF5Matrix(hdf5_file, 'targets', start=N_train+N_val, end=N)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, N_train

def load_data_hf5_memory(params,val_percent, test_percent, y_path, id2gt, X_meta = None, val_from_file = False):
    if val_from_file:
        hdf5_file = common.PATCHES_DIR+"/patches_train_%s_%sx%s.hdf5" % (params['dataset']['dataset'],params['dataset']['npatches'],params['dataset']['window'])
        f = h5py.File(hdf5_file,"r")
        index_train = f["index"][:]
        index_train = np.delete(index_train, np.where(index_train == ""))
        N_train = index_train.shape[0]
        
        val_hdf5_file = common.PATCHES_DIR+"/patches_val_%s_%sx%s.hdf5" % (params['dataset']['dataset'],params['dataset']['npatches'],params['dataset']['window'])
        f_val = h5py.File(val_hdf5_file,"r")
        X_val = f_val['features'][:]
        #Y_val = f_val['targets'][:]
        factors_val = np.load(common.DATASETS_DIR+'/item_factors_val_'+y_path+'.npy')
        index_factors_val = open(common.DATASETS_DIR+'/items_index_val_'+params['dataset']['dataset']+'.tsv').read().splitlines()
        id2gt_val = dict((index,factor) for (index,factor) in zip(index_factors_val,factors_val))            
        index_val = f_val['index'][:]
        X_val = np.delete(X_val, np.where(index_val == ""), axis=0)
        index_val = np.delete(index_val, np.where(index_val == ""))                
        Y_val = np.asarray([id2gt_val[id] for id in index_val])

        test_hdf5_file = common.PATCHES_DIR+"/patches_test_%s_%sx%s.hdf5" % (params['dataset']['dataset'],params['dataset']['npatches'],params['dataset']['window'])
        f_test = h5py.File(test_hdf5_file,"r")
        X_test = f_test['features'][:]
        #Y_test = f_test['targets'][:]
        factors_test = np.load(common.DATASETS_DIR+'/item_factors_test_'+y_path+'.npy')
        index_factors_test = open(common.DATASETS_DIR+'/items_index_test_'+params['dataset']['dataset']+'.tsv').read().splitlines()
        id2gt_test = dict((index,factor) for (index,factor) in zip(index_factors_test,factors_test))            
        index_test = f_test['index'][:]
        X_test = np.delete(X_test, np.where(index_test == ""), axis=0)
        index_test = np.delete(index_test, np.where(index_test == ""))                
        Y_test = np.asarray([id2gt_test[id] for id in index_test])
    else:
        hdf5_file = common.PATCHES_DIR+"/patches_train_%s_%sx%s.hdf5" % (params['dataset']['dataset'],params['dataset']['npatches'],params['dataset']['window'])
        f = h5py.File(hdf5_file,"r")
        index_all = f["index"][:]
        #index_all = np.delete(index_all, np.where(index_all == ""))
        N = index_all.shape[0]
        train_percent = 1 - val_percent - test_percent
        N_train = int(train_percent * N)
        N_val = int(val_percent * N)
        X_val = f['features'][N_train:N_train+N_val]
        index_val = f['index'][N_train:N_train+N_val]
        print(index_val)
        X_val = np.delete(X_val, np.where(index_val == ""), axis=0)
        index_val = np.delete(index_val, np.where(index_val == ""))                
        print("val",len(index_val))
        Y_val = np.asarray([id2gt[id] for id in index_val])
        X_test = f['features'][N_train+N_val:N]
        index_test = f['index'][N_train+N_val:N]
        X_test = np.delete(X_test, np.where(index_test == ""), axis=0)
        index_test = np.delete(index_test, np.where(index_test == ""))                
        Y_test = np.asarray([id2gt[id] for id in index_test])
        print("test",len(index_test))
        index_train = f['index'][:N_train]
        index_train = np.delete(index_train, np.where(index_train == ""))
        N_train = index_train.shape[0]
    if X_meta != None:
        X_val = [X_val,X_meta[N_train:N_train+N_val]]
        X_test = [X_test,X_meta[N_train+N_val:N]]
    return X_val, Y_val, X_test, Y_test, N_train

def batch_block_generator(params, y_path, N_train, id2gt, X_meta = None, val_from_file = False):
    hdf5_file = common.PATCHES_DIR+"/patches_train_%s_%sx%s.hdf5" % (params['dataset']['dataset'],params['dataset']['npatches'],params['dataset']['window'])
    f = h5py.File(hdf5_file,"r")
    block_step = 50000
    batch_size = 32
    randomize = True
    with_meta = False
    if X_meta != None:
        with_meta = True
    while 1:
        for i in range(0,N_train,block_step):
            x_block = f['features'][i:min(N_train,i+block_step)]
            index_block = f['index'][i:min(N_train,i+block_step)]
            #y_block = f['targets'][i:min(N_train,i+block_step)]
            x_block = np.delete(x_block, np.where(index_block == ""), axis=0)
            index_block = np.delete(index_block, np.where(index_block == ""))
            y_block = np.asarray([id2gt[id] for id in index_block])
            if params['training']['normalize_y'] == True:
                normalize(y_block,copy=False)
            items_list = range(x_block.shape[0])
            if randomize:
                random.shuffle(items_list)
            for j in range(0,len(items_list),batch_size):
                if j+batch_size <= x_block.shape[0]:
                    items_in_batch = items_list[j:j+batch_size]
                    x_batch = x_block[items_in_batch]
                    y_batch = y_block[items_in_batch]
                    if with_meta:
                        x_batch = [x_batch,X_meta[items_in_batch]]
                    yield (x_batch, y_batch)

def process(params,with_predict=True,with_eval=True):
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    params['cnn']['n_out'] = int(params['dataset']['dim'])
    params['cnn']['n_frames'] =  int(params['dataset']['window'] * SR / float(HR))
    with_metadata = params['dataset']['with_metadata']
    only_metadata = params['dataset']['only_metadata']
    metadata_source = params['dataset']['meta-suffix']
    if with_metadata:
        if 'w2v' in metadata_source:
            X_meta = np.load(common.DATASETS_DIR+'/train_data/X_train_%s_%s.npy' % (metadata_source,params['dataset']['dataset']))[:,:int(params['cnn']['sequence_length'])]
            params['cnn']['n_metafeatures'] = len(X_meta[0])
        elif 'model' in metadata_source or not params['dataset']['sparse']:
            X_meta = np.load(common.DATASETS_DIR+'/train_data/X_train_%s_%s.npy' % (metadata_source,params['dataset']['dataset']))
            params['cnn']['n_metafeatures'] = len(X_meta[0])
        else:
            X_meta = load_sparse_csr(common.DATASETS_DIR+'/train_data/X_train_%s_%s.npz' % (metadata_source,params['dataset']['dataset'])).todense()
            params['cnn']['n_metafeatures'] = X_meta.shape[1]
        print(X_meta.shape)
    else:
        X_meta = None

    config = Config(params)
    model_dir = os.path.join(common.MODELS_DIR, config.model_id)
    common.ensure_dir(common.MODELS_DIR)
    common.ensure_dir(model_dir)
    model_file = os.path.join(model_dir, config.model_id + common.MODEL_EXT)
    logging.debug("Building Network...")
    #model = build_model(config)
    model = build_model(config)
    print(model.summary())
    #plot(model, to_file='model2.png', show_shapes=True)
    trained_model = config.get_dict()

    # Save model
    #plot(model, to_file=os.path.join(model_dir, config.model_id + PLOT_EXT))
    common.save_model(model, model_file)

    logging.debug(trained_model["model_id"])

    logging.debug("Loading Data...")

    with_generator = True

    if only_metadata:
        X_train, Y_train, X_val, Y_val, X_test, Y_test = \
            load_data_preprocesed(params, config.x_path, config.y_path, params['dataset']['dataset'], config.training_params["validation"],
                      config.training_params["test"], config.dataset_settings["nsamples"], with_metadata, only_metadata, metadata_source)
        if 'meta-suffix2' in params['dataset']:
            X_train2, Y_train2, X_val2, Y_val2, X_test2, Y_test2 = \
                load_data_preprocesed(params, config.x_path, config.y_path, params['dataset']['dataset'], config.training_params["validation"],
                          config.training_params["test"], config.dataset_settings["nsamples"], with_metadata, only_metadata, params['dataset']['meta-suffix2'])
            X_train = [X_train,X_train2]
            X_val = [X_val,X_val2]
            X_test = [X_test,X_test2]
            #Y_train = [Y_train,Y_train2]
            #Y_val = [Y_val,Y_val2]
            #Y_test = [Y_test,Y_test2]
            print("X_train bi", len(X_train))
    else:
        if with_generator:
            id2gt = dict()
            factors = np.load(common.DATASETS_DIR+'/item_factors_train_'+config.y_path+'.npy')
            index_factors = open(common.DATASETS_DIR+'/items_index_train_'+params['dataset']['dataset']+'.tsv').read().splitlines()
            id2gt = dict((index,factor) for (index,factor) in zip(index_factors,factors))            
            X_val, Y_val, X_test, Y_test, N_train = load_data_hf5_memory(params,config.training_params["validation"],config.training_params["test"],config.y_path,id2gt,X_meta,config.training_params["val_from_file"])
            if params['dataset']['nsamples'] != 'all':
                N_train = min(N_train,params['dataset']['nsamples'])

        else:
            X_train, Y_train, X_val, Y_val, X_test, Y_test, N_train = load_data_hf5(params,config.training_params["validation"],config.training_params["test"])

    trained_model["whiten_scaler"] = common.DATASETS_DIR+'/train_data/scaler_%s.pk' % config.x_path
    #logging.debug(X_train.shape)
    logging.debug("Training...")

    if config.model_arch["final_activation"] == 'softmax':
        monitor_metric = 'val_categorical_accuracy'
    else:
        monitor_metric = 'val_loss'
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=4)
    
    if only_metadata:
        epochs = model.fit(X_train, Y_train,
                  batch_size=config.training_params["n_minibatch"],
                  #shuffle='batch',
                  nb_epoch=config.training_params["n_epochs"],
                  verbose=2, validation_data=(X_val, Y_val),
                  callbacks=[early_stopping])
    else:
        if with_generator:
            print(N_train)
            epochs = model.fit_generator(batch_block_generator(params,config.y_path,N_train,id2gt,X_meta,config.training_params["val_from_file"]),
                        samples_per_epoch = N_train-(N_train % config.training_params["n_minibatch"]),
                        nb_epoch = config.training_params["n_epochs"],
                        verbose=2,
                        validation_data = (X_val, Y_val),
                        callbacks=[early_stopping])
        else:
            epochs = model.fit(X_train, Y_train,
                      batch_size=config.training_params["n_minibatch"],
                      shuffle='batch',
                      nb_epoch=config.training_params["n_epochs"],
                      verbose=2, 
                      validation_data=(X_val, Y_val),
                      callbacks=[early_stopping])

    model.save_weights(os.path.join(model_dir, config.model_id + common.WEIGHTS_EXT))
    logging.debug("Saving trained model %s in %s..." %
                  (trained_model["model_id"], common.DEFAULT_TRAINED_MODELS_FILE))
    common.save_trained_model(common.DEFAULT_TRAINED_MODELS_FILE, trained_model)

    logging.debug("Evaluating...")
    #convout1_f = theano.function([model.get_input(train=False)], xout.get_output(train=False))
    #f = K.function([K.learning_phase(), model.layers[0].input], [model.layers[1].output])
    #C1 = f(X_test)
    #C1 = np.squeeze(C1)
    #print("C1 shape : ", C1.shape)
    #np.save('results/last_layer_%s' % trained_model["model_id"],C1)
    #print(C1)
    # Step prediction


    preds=model.predict(X_test)
    print(preds.shape)
    if params["dataset"]["evaluation"] in ['binary','multiclass']:
        #if preds.shape[-1] > 1:
        #    y_pred = preds.argmax(axis=-1)
        #else:
        y_pred = (preds > 0.5).astype('int32')        
        acc = accuracy_score(Y_test,y_pred)
        prec = precision_score(Y_test,y_pred,average='macro')
        recall = recall_score(Y_test,y_pred,average='macro')
        f1 = f1_score(Y_test,y_pred,average='macro')
        print('Accuracy', acc)
        print("%.3f\t%.3f\t%.3f" % (prec,recall,f1))
    if params["dataset"]["fact"] == 'class':
        good_classes = np.nonzero(Y_test.sum(0))[0]
        roc_auc=roc_auc_score(Y_test[:,good_classes],preds[:,good_classes])
        logging.debug('ROC-AUC '+str(roc_auc))
        r2 = roc_auc
    else:
        r2s = []
        for i,pred in enumerate(preds):
            r2 = r2_score(Y_test[i],pred)
            r2s.append(r2)
        r2 = np.asarray(r2s).mean()
        logging.debug('R2 avg '+str(r2))
    # Batch prediction
    score = model.evaluate(X_test, Y_test, verbose=0)
    logging.debug(score)
    logging.debug(model.metrics_names)
    print(score)
    trained_model["loss_score"] = score[0]
    trained_model["mse"] = score[1]
    trained_model["r2"] = r2

    fw=open(common.DATA_DIR+'/results/train_results.txt','a')
    fw.write(trained_model["model_id"]+'\n')
    if params["training"]["loss_func"] == 'binary_crossentropy':
        fw.write('ROC-AUC: '+str(roc_auc)+'\n')
        print('ROC-AUC: '+str(roc_auc))
    else:
        fw.write('R2 avg: '+str(r2)+'\n')
        print('R2 avg: '+str(r2))
    fw.write('Loss: '+str(score[0])+' ('+config.training_params["loss_func"]+')\n')
    fw.write('MSE: '+str(score[1])+'\n')
    fw.write(json.dumps(epochs.history)+"\n\n")
    fw.close()

    if with_predict:
        trained_models = pd.read_csv(common.DEFAULT_TRAINED_MODELS_FILE, sep='\t')
        model_config = trained_models[trained_models["model_id"] == trained_model["model_id"]]
        model_config = model_config.to_dict(orient="list")
        testset = open(common.DATASETS_DIR+'/items_index_test_%s.tsv' % (config.dataset_settings["dataset"])).read().splitlines()
        #if with_metadata:
        #    testset = open(common.DATASETS_DIR+'/train_data/index_test_%s_%s.tsv' % (metadata_source,config.dataset_settings["dataset"])).read().splitlines()
        #else:
        #    testset=open(common.DATASETS_DIR+'/testset_%s.tsv' % config.dataset_settings["dataset"]).read().splitlines()
        if config.training_params["val_from_file"]:
            factors, factors_index = obtain_factors(model_config, testset, trained_model["model_id"], config.predicting_params["trim_coeff"], model=model, with_metadata=with_metadata, only_metadata=only_metadata, metadata_source=metadata_source, with_patches=True)
        else:
            factors, factors_index = obtain_factors(model_config, testset, trained_model["model_id"], config.predicting_params["trim_coeff"], model=model, with_metadata=with_metadata, only_metadata=only_metadata, metadata_source=metadata_source)
        #predict(trained_model["model_id"])
        print("Factors created")

    if with_eval:
        do_eval(trained_model["model_id"],get_roc=False,get_map=True,get_p=False,factors=factors,factors_index=factors_index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates the model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p',
                        '--params',
                        dest="params_file",
                        help='JSON file with params',
                        default=False)
    parser.add_argument('-pred',
                        '--predict',
                        dest="with_predict",
                        help='Predict factors',
                        action='store_true',
                        default=False)
    parser.add_argument('-eval',
                        '--eval',
                        dest="with_eval",
                        help='Eval factors',
                        action='store_true',
                        default=False)
    parser.add_argument('-m',
                        '--metadata',
                        dest="with_metadata",
                        help='Use metadata',
                        action='store_true',
                        default=False)
    parser.add_argument('-om',
                        '--only_metadata',
                        dest="only_metadata",
                        help='Use only metadata',
                        action='store_true',
                        default=False)
    parser.add_argument('-ms',
                        '--metadata_source',
                        dest="metadata_source",
                        type=str,
                        help='Suffix of metadata files',
                        default="rovi")
    args = parser.parse_args()
    params = models.params_1
    if args.params_file:
        params = json.load(open(args.params_file))
    process(params)
