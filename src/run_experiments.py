from train import process
import models
from subprocess import call
import argparse
import json
import copy

def run(params,gpu):
    json.dump(params,open("params_gpu%d.json","w") % gpu)
    call("THEANO_FLAGS='device=gpu%d' python train.py -p params_gpu%d.json" % (gpu, gpu))

def add_extra_params(params,extra_params):
    if extra_params != '':
        new_params = extra_params.replace("'","").split("&")
        for p in new_params:
            t,v = p.split("=")
            t1,t2 = t.split(".")
            try:
                params[t1][t2] = eval(v)
            except:
                params[t1][t2] = v
            print t1,t2,params[t1][t2]

def get_configuration(suffix, meta_suffix='bow', meta_suffix2='bow', meta_suffix3='bow', meta_suffix4='bow', extra_params=''):
    params = dict()

    ###################################################################
    ### Dummy experiments on SUPER dataset, multiclass classification
    ###################################################################

    # Audio experiment SUPER
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multiclass' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = False
    nparams["dataset"]["only_metadata"] = False
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 3
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["n_minibatch"] = 3
    nparams["cnn"]["architecture"] = '1'
    nparams["cnn"]["n_frames"] = 322
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.0
    nparams["cnn"]["final_activation"] = 'softmax'
    nparams["cnn"]["n_filters_1"] = 64
    nparams["cnn"]["n_filters_2"] = 128
    nparams["cnn"]["n_filters_3"] = 0
    nparams["cnn"]["n_filters_4"] = 0
    nparams["cnn"]["n_pool_2"] = (2,1)
    nparams["cnn"]["n_pool_3"] = (1,1)
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'SUPER'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    add_extra_params(nparams, extra_params)
    params['dummy_audio'] = copy.deepcopy(nparams)

    # Text experiment with VSM for SUPER
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multiclass' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = True
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 3
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["n_minibatch"] = 3
    nparams["cnn"]["architecture"] = '3'
    nparams["cnn"]["n_dense"] = 128
    nparams["cnn"]["n_dense_2"] = 128
    nparams["cnn"]["dropout_factor"] = 0.5
    nparams["cnn"]["final_activation"] = 'softmax'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'SUPER'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    add_extra_params(nparams, extra_params)
    params['dummy_text_vsm'] = copy.deepcopy(nparams)

    # Text experiment with w2v for SUPER
    nparams = copy.deepcopy(models.params_w2v)
    nparams["dataset"]["evaluation"] = 'multiclass' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 3
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["n_minibatch"] = 3
    nparams["cnn"]["architecture"] = '4'
    nparams["cnn"]["sequence_length"] = 500
    nparams["cnn"]["embedding_dim"] = 300
    nparams["cnn"]["filter_sizes"] = (2,3)
    nparams["cnn"]["n_dense"] = 128
    nparams["cnn"]["final_activation"] = 'softmax'
    nparams["cnn"]["embeddings_suffix"] = 'SUPER'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'SUPER'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    add_extra_params(nparams, extra_params)
    params['dummy_text_w2v'] = copy.deepcopy(nparams)

    # Multimodal experiment SUPER (audio+text)
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multiclass' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 3
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '32'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.0
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'SUPER'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["dataset"]["meta-suffix2"] = meta_suffix2
    add_extra_params(nparams, extra_params)
    params['dummy_multimodal'] = copy.deepcopy(nparams)


    ################################################
    ### DLRS 2017 Experiments (Recommendation)
    ################################################

    #Regression experiment with sparse matrices as input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'recommendation' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 200
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = True
    nparams["training"]["val_from_file"] = False
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '3'
    nparams["cnn"]["n_dense"] = 2048
    nparams["cnn"]["n_dense_2"] = 2048
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-A-artists'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["cnn"]["final_activation"] = 'linear'
    add_extra_params(nparams, extra_params)
    params['rec_sparse'] = copy.deepcopy(nparams)

    #Regression experiment with dense matrices as input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'recommendation' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 200
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = True
    nparams["training"]["val_from_file"] = False
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '3'
    nparams["cnn"]["n_dense"] = 2048
    nparams["cnn"]["n_dense_2"] = 2048
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-A-artists' #MSD-A-songs for song rec
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["cnn"]["final_activation"] = 'linear'
    add_extra_params(nparams, extra_params)
    params['rec_dense'] = copy.deepcopy(nparams)

    #Regression multimodal with two embedding vectors as input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'recommendation' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = False
    nparams["dataset"]["dim"] = 200 #397
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '32'
    nparams["cnn"]["n_dense"] = 512
    nparams["cnn"]["dropout_factor"] = 0.7
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-A-songs'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["dataset"]["meta-suffix2"] = meta_suffix2
    #nparams["cnn"]["n_metafeatures2"] = 256
    add_extra_params(nparams, extra_params)
    params['rec_multi'] = copy.deepcopy(nparams)      


    ################################################
    # ISMIR 2017 / TISMIR Multimodal Experiments (Classification)
    ################################################

    #LOGISTIC multimodal one feature vector input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 250
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '3'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.0
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-albums'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    add_extra_params(nparams, extra_params)
    params['logistic_multilabel_vector'] = copy.deepcopy(nparams)

    #COSINE multimodal one feature vectors input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 50
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '3'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.7
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-albums'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    add_extra_params(nparams, extra_params)
    params['cosine_multilabel_vector'] = copy.deepcopy(nparams)   

    #LOGISTIC multimodal two feature vectors input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 250
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '32'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.0
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-albums'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["dataset"]["meta-suffix2"] = meta_suffix2
    add_extra_params(nparams, extra_params)
    params['logistic_multilabel_bi'] = copy.deepcopy(nparams)

    #COSINE multimodal two feature vectors input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 50
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '32'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.7
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-albums'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["dataset"]["meta-suffix2"] = meta_suffix2
    add_extra_params(nparams, extra_params)
    params['cosine_multilabel_bi'] = copy.deepcopy(nparams)    


    #LOGISTIC multimodal three feature vectors input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 250 #397
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '33'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.0
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-albums'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["dataset"]["meta-suffix2"] = meta_suffix2
    nparams["dataset"]["meta-suffix3"] = meta_suffix3
    add_extra_params(nparams, extra_params)
    params['logistic_multilabel_tri'] = copy.deepcopy(nparams)

    #COSINE multimodal three feature vectors input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 50
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '33'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.7
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-albums'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["dataset"]["meta-suffix2"] = meta_suffix2
    nparams["dataset"]["meta-suffix3"] = meta_suffix3
    add_extra_params(nparams, extra_params)
    params['cosine_multilabel_tri'] = copy.deepcopy(nparams)  

    #Audio 4x70-high-COSINE
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel'
    nparams["dataset"]["fact"] = 'pmi'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = False
    nparams["dataset"]["only_metadata"] = False
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 50
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '1'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["n_kernel_1"] = (4, 70)
    nparams["cnn"]["n_kernel_2"] = (4, 6)
    nparams["cnn"]["n_kernel_3"] = (4, 1)
    nparams["cnn"]["n_kernel_4"] = (1, 1)
    nparams["cnn"]["n_filters_1"] = 256
    nparams["cnn"]["n_filters_2"] = 512
    nparams["cnn"]["n_filters_3"] = 1024
    nparams["cnn"]["n_filters_4"] = 1024
    nparams["cnn"]["n_pool_1"] = (4, 4)
    nparams["cnn"]["n_pool_2"] = (4, 1)
    nparams["cnn"]["n_pool_3"] = (4, 1)
    nparams["cnn"]["n_pool_4"] = (1, 1)
    nparams["cnn"]["dropout_factor"] = 0.5
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-tracks'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    add_extra_params(nparams, extra_params)
    params['cosine_audio_multilabel'] = copy.deepcopy(nparams)

    #Audio 4x70-low-LOGISTIC
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel'
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = False
    nparams["dataset"]["only_metadata"] = False
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 250
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '1'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["n_kernel_1"] = (4, 70)
    nparams["cnn"]["n_kernel_2"] = (4, 6)
    nparams["cnn"]["n_kernel_3"] = (4, 1)
    nparams["cnn"]["n_kernel_4"] = (1, 1)
    nparams["cnn"]["n_filters_1"] = 64
    nparams["cnn"]["n_filters_2"] = 128
    nparams["cnn"]["n_filters_3"] = 128
    nparams["cnn"]["n_filters_4"] = 64
    nparams["cnn"]["n_pool_1"] = (4, 4)
    nparams["cnn"]["n_pool_2"] = (4, 1)
    nparams["cnn"]["n_pool_3"] = (4, 1)
    nparams["cnn"]["n_pool_4"] = (1, 1)
    nparams["cnn"]["dropout_factor"] = 0.0
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-tracks'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    add_extra_params(nparams, extra_params)
    params['logistic_audio_multilabel'] = copy.deepcopy(nparams)

    #LOGISTIC text, meta_suffix = VSM / VSM-SEM
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = True
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 250
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '3'
    nparams["cnn"]["n_dense"] = 2048
    nparams["cnn"]["n_dense_2"] = 2048
    nparams["cnn"]["dropout_factor"] = 0.5
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-albums'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    add_extra_params(nparams, extra_params)
    params['logistic_text_multilabel'] = copy.deepcopy(nparams)

    #COSINE text, meta_suffix = VSM / VSM-SEM
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = True
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 50
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '3'
    nparams["cnn"]["n_dense"] = 2048
    nparams["cnn"]["n_dense_2"] = 2048
    nparams["cnn"]["dropout_factor"] = 0.5
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-albums'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    add_extra_params(nparams, extra_params)
    params['cosine_text_multilabel'] = copy.deepcopy(nparams)

    ################################################
    # TISMIR Single-label Experiments (Classification)
    ################################################

    #Audio experiment
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multiclass'
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["dim"] = 15
    nparams["training"]["loss_func"] = 'categorical_crossentropy'
    nparams["cnn"]["architecture"] = '1'
    nparams["dataset"]["npatches"] = 1  
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-I'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["with_metadata"] = False
    nparams["dataset"]["only_metadata"] = False
    nparams["dataset"]["configuration"] = suffix
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["n_kernel_1"] = (4, 96)
    nparams["cnn"]["n_kernel_2"] = (4, 1)
    nparams["cnn"]["n_kernel_3"] = (4, 1)
    nparams["cnn"]["n_filters_1"] = 64
    nparams["cnn"]["n_filters_2"] = 128
    nparams["cnn"]["n_filters_3"] = 256
    nparams["cnn"]["n_filters_4"] = 0
    nparams["cnn"]["n_pool_1"] = (4, 1)
    nparams["cnn"]["n_pool_2"] = (4, 1)
    nparams["cnn"]["n_pool_3"] = (2, 1)
    nparams["cnn"]["final_activation"] = 'softmax'
    nparams["dataset"]["meta-suffix"] = ""
    add_extra_params(nparams, extra_params)
    params['single-label-audio'] = copy.deepcopy(nparams)

    #Class experiment with one feature vector input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multiclass' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 15
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["training"]["loss_func"] = 'categorical_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '3'
    nparams["cnn"]["dropout_factor"] = 0.5
    nparams["cnn"]["final_activation"] = 'softmax'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["n_dense2"] = 0
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-I'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    add_extra_params(nparams, extra_params)
    params['single-label-vector'] = copy.deepcopy(nparams)     

    #multimodal two feature vectors input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multiclass' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 15
    nparams["training"]["loss_func"] = 'categorical_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '32'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.5
    nparams["cnn"]["final_activation"] = 'softmax'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-I'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["dataset"]["meta-suffix2"] = meta_suffix2
    add_extra_params(nparams, extra_params)
    params['single-label-multimodal'] = copy.deepcopy(nparams)

    #multimodal four feature vectors input
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multiclass' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 15
    nparams["training"]["loss_func"] = 'categorical_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '34'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.5
    nparams["cnn"]["final_activation"] = 'softmax'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-I'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["dataset"]["meta-suffix2"] = meta_suffix2
    nparams["dataset"]["meta-suffix3"] = meta_suffix3
    nparams["dataset"]["meta-suffix4"] = meta_suffix4
    add_extra_params(nparams, extra_params)
    params['single-label-all'] = copy.deepcopy(nparams)

    return params[suffix]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('suffix', default="class_bow", help='Suffix of experiment params')
    parser.add_argument('meta_suffix', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('meta_suffix2', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('meta_suffix3', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('meta_suffix4', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('extra_params', nargs='?', default="", help='Specific extra parameters')
    args = parser.parse_args()
    print args.extra_params
    params = get_configuration(args.suffix,args.meta_suffix,args.meta_suffix2,args.meta_suffix3,args.meta_suffix4,args.extra_params)
    process(params)
