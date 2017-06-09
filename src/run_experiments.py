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

def get_configuration(suffix, meta_suffix='bow', meta_suffix2='bow', meta_suffix3='bow', extra_params=''):
    params = dict()
    #Class experiment w2v
    nparams = copy.deepcopy(models.params_82)
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 300
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = 82
    nparams["cnn"]["sequence_length"] = 500
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG'
    nparams["dataset"]["meta-suffix"] = meta_suffix #w2v
    add_extra_params(nparams, extra_params)
    params['class_w2v'] = copy.deepcopy(nparams)

    #Class experiment LSTM w2v
    nparams = copy.deepcopy(models.params_82)
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 300
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = 10
    nparams["cnn"]["sequence_length"] = 500
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG'
    nparams["dataset"]["meta-suffix"] = meta_suffix #w2v
    add_extra_params(nparams, extra_params)
    params['class_lstm'] = copy.deepcopy(nparams)

    #Class experiment bow
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 300
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '81'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG'
    nparams['dataset']['sparse'] = True
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["training"]["val_from_file"] = False
    add_extra_params(nparams, extra_params)
    params['class_bow'] = copy.deepcopy(nparams)

    #Factors experiment bow
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 200
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = True
    nparams["training"]["val_from_file"] = True #OJO!!!!!!
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '8'
    nparams["cnn"]["n_dense"] = 2048
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG-S' #OJO!!!!!!
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["cnn"]["final_activation"] = 'linear'
    add_extra_params(nparams, extra_params)
    params['fact_bow'] = copy.deepcopy(nparams)

    #Factors experiment w2v
    nparams = copy.deepcopy(models.params_82)
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 200
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = False
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = 82
    nparams["cnn"]["sequence_length"] = 300
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["cnn"]["hidden_dims"] = 1024
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG'
    nparams["dataset"]["meta-suffix"] = meta_suffix #w2v
    nparams["cnn"]["final_activation"] = 'linear'
    add_extra_params(nparams, extra_params)
    params['fact_w2v'] = copy.deepcopy(nparams)


    #Class experiment bow TEST
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 300
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '81'
    nparams["dataset"]["nsamples"] = '100'
    nparams["dataset"]["dataset"] = 'MSD-AG'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    add_extra_params(nparams, extra_params)
    params['test'] = copy.deepcopy(nparams)

    #Audio experiment
    nparams = copy.deepcopy(models.params_5)
    nparams["dataset"]["evaluation"] = 'recommendation' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["dim"] = 200
    nparams["training"]["loss_func"] = 'cosine'
    nparams["cnn"]["architecture"] = '5'
    nparams["dataset"]["npatches"] = 3
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG-S'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["training"]["val_from_file"] = False
    nparams["dataset"]["with_metadata"] = False
    nparams["dataset"]["only_metadata"] = False
    nparams["cnn"]["n_filters_1"] = 512
    nparams["cnn"]["n_filters_2"] = 1024
    nparams["cnn"]["n_filters_3"] = 2048
    nparams["cnn"]["n_filters_4"] = 2048
    nparams["cnn"]["n_pool_3"] = (4,1)
    nparams["cnn"]["n_pool_4"] = (4,1)
    nparams["cnn"]["n_dense"] = 0
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["meta-suffix"] = ""
    nparams["cnn"]["final_activation"] = 'linear'
    add_extra_params(nparams, extra_params)
    params['audio'] = copy.deepcopy(nparams)

    #Audio Class1 experiment
    nparams = copy.deepcopy(models.params_5)
    nparams["dataset"]["evaluation"] = 'binary' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["dim"] = 15
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["cnn"]["architecture"] = '5'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-IGT'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["with_metadata"] = False
    nparams["dataset"]["only_metadata"] = False
    nparams["dataset"]["configuration"] = suffix
    #nparams["cnn"]["n_dense"] = 2048
    nparams["cnn"]["n_filters_1"] = 64
    nparams["cnn"]["n_filters_2"] = 128
    nparams["cnn"]["n_filters_3"] = 256
    nparams["cnn"]["n_filters_4"] = 0
    nparams["cnn"]["n_pool_3"] = (2,1)
    nparams["cnn"]["final_activation"] = 'softmax'
    nparams["dataset"]["meta-suffix"] = ""
    add_extra_params(nparams, extra_params)
    params['audio_genre'] = copy.deepcopy(nparams)

    #Audio Multilabel experiment
    nparams = copy.deepcopy(models.params_5) # 5 tartarus 13 keunwoo
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["dim"] = 250
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["window"] = 29 # CRNN
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'multi2deT'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["val_from_file"] = True
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["dataset"]["with_metadata"] = False
    nparams["dataset"]["only_metadata"] = False
    nparams["dataset"]["configuration"] = suffix
    # Our architecture
    nparams["cnn"]["architecture"] = '15' # 5 tartarus / 15 keunwoo
    # Jordi
    #nparams["cnn"]["n_kernel_1"] = (4,70)
    #nparams["cnn"]["n_kernel_2"] = (4,6)
    #nparams["cnn"]["n_pool_1"] = (4,4)
    # Keun
    nparams["cnn"]["n_kernel_1"] = (3,3)
    nparams["cnn"]["n_kernel_2"] = (3,3)
    nparams["cnn"]["n_kernel_3"] = (3,3)
    nparams["cnn"]["n_kernel_4"] = (3,3)
    nparams["cnn"]["n_kernel_5"] = (3,3)
    nparams["cnn"]["n_pool_1"] = (2,4)
    nparams["cnn"]["n_pool_2"] = (2,4)
    nparams["cnn"]["n_pool_3"] = (2,4)
    nparams["cnn"]["n_pool_4"] = (4,1)
    nparams["cnn"]["n_pool_5"] = (4,1)
    nparams["cnn"]["dropout_factor"] = 0.0
    # Ours
    nparams["cnn"]["n_pool_3"] = (4,1)
    # High
    nparams["cnn"]["n_filters_1"] = 256
    nparams["cnn"]["n_filters_2"] = 512
    nparams["cnn"]["n_filters_3"] = 1024
    nparams["cnn"]["n_filters_4"] = 1024
    nparams["cnn"]["n_filters_5"] = 0
    # Low
    #nparams["cnn"]["n_filters_1"] = 64
    #nparams["cnn"]["n_filters_2"] = 128
    #nparams["cnn"]["n_filters_3"] = 128
    #nparams["cnn"]["n_filters_4"] = 64
    #nparams["cnn"]["n_filters_5"] = 0

    #nparams["cnn"]["n_dense"] = 2048
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["meta-suffix"] = ""
    add_extra_params(nparams, extra_params)
    params['audio_multilabel'] = copy.deepcopy(nparams)

    #Audio Multilabel experiment
    nparams = copy.deepcopy(models.params_5)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    nparams["dataset"]["dim"] = 50
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'multi2deT'
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["with_metadata"] = False
    nparams["dataset"]["only_metadata"] = False
    nparams["dataset"]["configuration"] = suffix
    nparams["cnn"]["architecture"] = '5' # 5 tartarus / 13 keunwoo
    # Jordi
    #nparams["cnn"]["n_kernel_1"] = (4,70)
    #nparams["cnn"]["n_kernel_2"] = (4,6)
    #nparams["cnn"]["n_pool_1"] = (4,4)
    # Keun
    nparams["cnn"]["n_kernel_1"] = (3,3)
    nparams["cnn"]["n_kernel_2"] = (3,3)
    nparams["cnn"]["n_kernel_3"] = (3,3)
    nparams["cnn"]["n_kernel_4"] = (3,3)
    nparams["cnn"]["n_kernel_5"] = (3,3)
    nparams["cnn"]["n_pool_1"] = (2,4)
    nparams["cnn"]["n_pool_2"] = (2,4)
    nparams["cnn"]["n_pool_3"] = (2,4)
    nparams["cnn"]["n_pool_4"] = (4,1)
    nparams["cnn"]["n_pool_5"] = (4,1)
    nparams["cnn"]["dropout_factor"] = 0.0
    # Ours
    nparams["cnn"]["n_pool_3"] = (4,1)
    # High
    nparams["cnn"]["n_filters_1"] = 256
    nparams["cnn"]["n_filters_2"] = 512
    nparams["cnn"]["n_filters_3"] = 1024
    nparams["cnn"]["n_filters_4"] = 1024
    nparams["cnn"]["n_filters_5"] = 0
    # Low
    #nparams["cnn"]["n_filters_1"] = 64
    #nparams["cnn"]["n_filters_2"] = 128
    #nparams["cnn"]["n_filters_3"] = 128
    #nparams["cnn"]["n_filters_4"] = 64
    #nparams["cnn"]["n_filters_5"] = 0
    
    #nparams["cnn"]["n_dense"] = 2048
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["meta-suffix"] = ""
    add_extra_params(nparams, extra_params)
    params['audio_fact_multilabel'] = copy.deepcopy(nparams)


    #Audio Multilabel experiment with new Jordi multi-filters
    nparams = copy.deepcopy(models.params_5)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    nparams["dataset"]["dim"] = 50
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'multi2deT'
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["with_metadata"] = False
    nparams["dataset"]["only_metadata"] = False
    nparams["dataset"]["configuration"] = suffix
    nparams["cnn"]["architecture"] = '51' # 5 tartarus / 13 keunwoo
    # Jordi
    #nparams["cnn"]["n_kernel_1"] = (4,70)
    #nparams["cnn"]["n_kernel_2"] = (4,6)
    #nparams["cnn"]["n_pool_1"] = (4,4)
    # Jordi multi-filter
    nparams["cnn"]["n_pool_1"] = (4,16)
    nparams["cnn"]["n_kernel_2"] = (4,6)
    # Keun
    #nparams["cnn"]["n_kernel_1"] = (3,3)
    #nparams["cnn"]["n_kernel_2"] = (3,3)
    #nparams["cnn"]["n_kernel_3"] = (3,3)
    #nparams["cnn"]["n_kernel_4"] = (3,3)
    #nparams["cnn"]["n_kernel_5"] = (3,3)
    #nparams["cnn"]["n_pool_1"] = (2,4)
    #nparams["cnn"]["n_pool_2"] = (2,4)
    #nparams["cnn"]["n_pool_3"] = (2,4)
    #nparams["cnn"]["n_pool_4"] = (4,1)
    #nparams["cnn"]["n_pool_5"] = (4,1)
    nparams["cnn"]["dropout_factor"] = 0.0
    # Ours
    nparams["cnn"]["n_pool_3"] = (4,1)
    # High
    nparams["cnn"]["n_filters_1"] = 256
    nparams["cnn"]["n_filters_2"] = 512
    nparams["cnn"]["n_filters_3"] = 1024
    nparams["cnn"]["n_filters_4"] = 1024
    nparams["cnn"]["n_filters_5"] = 0
    # Low
    #nparams["cnn"]["n_filters_1"] = 64
    #nparams["cnn"]["n_filters_2"] = 128
    #nparams["cnn"]["n_filters_3"] = 128
    #nparams["cnn"]["n_filters_4"] = 64
    #nparams["cnn"]["n_filters_5"] = 0
    
    #nparams["cnn"]["n_dense"] = 2048
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["meta-suffix"] = ""
    add_extra_params(nparams, extra_params)
    params['audio_fact_multilabel_jordi'] = copy.deepcopy(nparams)



    # MLP fact
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 50
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '81'
    nparams["cnn"]["dropout_factor"] = 0.0
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'multi2deA2'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    add_extra_params(nparams, extra_params)
    params['mlp_fact'] = copy.deepcopy(nparams)

    # MLP class
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 250
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '81'
    nparams["cnn"]["dropout_factor"] = 0.0
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'multi2deA2'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    add_extra_params(nparams, extra_params)
    params['mlp_class'] = copy.deepcopy(nparams)

    #Multimodal experiment bow
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 200
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = False
    nparams["dataset"]["configuration"] = suffix
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '6'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG-S'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["cnn"]["n_filters_1"] = 256
    nparams["cnn"]["n_filters_2"] = 512
    nparams["cnn"]["n_filters_3"] = 1024
    nparams["cnn"]["n_filters_4"] = 1024
    nparams["cnn"]["n_pool_3"] = (4,1)
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["sparse"] = True
    nparams["training"]["val_from_file"] = False
    add_extra_params(nparams, extra_params)
    params['fact_multi'] = copy.deepcopy(nparams)


    #Class experiment for genre Multilabel fusion
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 15
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '81'
    nparams["cnn"]["dropout_factor"] = 0.5
    nparams["cnn"]["final_activation"] = 'softmax'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-IGT'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    add_extra_params(nparams, extra_params)
    params['multi_genre'] = copy.deepcopy(nparams)


    #Fact experiment for MSD-AG
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 200
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = True #!!!!ojo
    nparams["training"]["val_from_file"] = False #!!!!ojo
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '81'
    nparams["cnn"]["dropout_factor"] = 0.7
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG-S'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    add_extra_params(nparams, extra_params)
    params['multi_fact'] = copy.deepcopy(nparams)



    #Fact multimodal bi metadata input
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 200
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True #OJO !!!!!!!!
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["training"]["n_epochs"] = 200
    nparams["cnn"]["architecture"] = '812'
    nparams["cnn"]["n_dense"] = 512
    nparams["cnn"]["dropout_factor"] = 0.7
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG-S'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["meta-suffix2"] = meta_suffix2 #bow
    nparams["cnn"]["n_metafeatures2"] = 200 #MSD-AG-S 4096 para audio
    add_extra_params(nparams, extra_params)
    params['multi_fact_bi'] = copy.deepcopy(nparams)



    #Fact multimodal bi metadata input
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 15
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '812'
    nparams["cnn"]["n_dense"] = 512
    nparams["cnn"]["dropout_factor"] = 0.7
    nparams["cnn"]["final_activation"] = 'softmax'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-IGT'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["meta-suffix2"] = meta_suffix2 #bow
    nparams["cnn"]["n_metafeatures2"] = 2048
    add_extra_params(nparams, extra_params)
    params['multi_genre_bi'] = copy.deepcopy(nparams)


    #Bow binary experiment
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["evaluation"] = 'binary' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["dim"] = 13
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'mard'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["val_from_file"] = False
    nparams["training"]["loss_func"] = 'categorical_crossentropy'
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["sparse"] = True
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["configuration"] = suffix
    nparams["cnn"]["architecture"] = '8' # 5
    #nparams["cnn"]["n_dense"] = 2048
    nparams["cnn"]["final_activation"] = 'softmax'
    add_extra_params(nparams, extra_params)
    params['bow_binary'] = copy.deepcopy(nparams)

    #w2v multiclass experiment freesound
    nparams = copy.deepcopy(models.params_82)
    nparams["dataset"]["evaluation"] = 'multiclass' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["dim"] = 397
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'fsd2'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["val_from_file"] = False
    nparams["training"]["loss_func"] = 'categorical_crossentropy'
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["sparse"] = False
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["configuration"] = suffix
    nparams["cnn"]["architecture"] = '82' # 5
    nparams["cnn"]["sequence_length"] = 10
    nparams["cnn"]["embedding_dim"] = 300
    nparams["cnn"]["n_dense"] = 512
    nparams["cnn"]["filter_sizes"] = (1,)
    nparams["cnn"]["final_activation"] = 'softmax'
    add_extra_params(nparams, extra_params)
    params['w2v_multiclass'] = copy.deepcopy(nparams)    

    #w2v multilabel experiment freesound
    nparams = copy.deepcopy(models.params_82)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    #nparams["dataset"]["dim"] = 419 # Freesound
    nparams["dataset"]["dim"] = 250 # multi2de
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["nsamples"] = 'all'
    #nparams["dataset"]["dataset"] = 'fsd-m'
    nparams["dataset"]["dataset"] = 'multi2deR'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["val_from_file"] = True
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["sparse"] = False
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["configuration"] = suffix
    nparams["cnn"]["architecture"] = '82' # 5
    #nparams["cnn"]["sequence_length"] = 10 #Freesound
    nparams["cnn"]["sequence_length"] = 300
    nparams["cnn"]["embedding_dim"] = 300
    nparams["cnn"]["n_dense"] = 300
    nparams["cnn"]["filter_sizes"] = (2,3)
    nparams["cnn"]["final_activation"] = 'sigmoid'
    add_extra_params(nparams, extra_params)
    params['w2v_class_multilabel'] = copy.deepcopy(nparams)    

    #w2v fact multilabel
    nparams = copy.deepcopy(models.params_82)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    #nparams["dataset"]["dim"] = 419 # Freesound
    nparams["dataset"]["dim"] = 50 # multi2de
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["nsamples"] = 'all'
    #nparams["dataset"]["dataset"] = 'fsd-m'
    nparams["dataset"]["dataset"] = 'multi2deA2'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["val_from_file"] = True
    nparams["training"]["loss_func"] = 'cosine'
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["sparse"] = False
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["configuration"] = suffix
    nparams["cnn"]["architecture"] = '82' # 5
    #nparams["cnn"]["sequence_length"] = 10 #Freesound
    nparams["cnn"]["sequence_length"] = 300
    nparams["cnn"]["embedding_dim"] = 300
    nparams["cnn"]["n_dense"] = 512
    nparams["cnn"]["filter_sizes"] = (3,8)
    nparams["cnn"]["final_activation"] = 'linear'
    add_extra_params(nparams, extra_params)
    params['w2v_fact_multilabel'] = copy.deepcopy(nparams)   

    #bow multilabel experiment freesound
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    #nparams["dataset"]["dim"] = 419 # Freesound
    nparams["dataset"]["dim"] = 250 # multi2de
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["nsamples"] = 'all'
    #nparams["dataset"]["dataset"] = 'fsd-m'
    nparams["dataset"]["dataset"] = 'multi2deA2'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["val_from_file"] = True
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["sparse"] = True
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["configuration"] = suffix
    nparams["cnn"]["architecture"] = '8' # 5
    #nparams["cnn"]["sequence_length"] = 10 #Freesound
    #nparams["cnn"]["sequence_length"] = 300
    #nparams["cnn"]["embedding_dim"] = 300
    #nparams["cnn"]["n_dense"] = 512
    #nparams["cnn"]["filter_sizes"] = (3,8)
    nparams["cnn"]["final_activation"] = 'sigmoid'
    add_extra_params(nparams, extra_params)
    params['bow_class_multilabel'] = copy.deepcopy(nparams) 


    #bow multilabel experiment
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    #nparams["dataset"]["dim"] = 419 # Freesound
    nparams["dataset"]["dim"] = 50 # multi2de
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'multi2deA2'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["training"]["val_from_file"] = True
    nparams["training"]["loss_func"] = 'cosine'
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["sparse"] = True
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["configuration"] = suffix
    nparams["cnn"]["dropout_factor"] = 0.3
    nparams["cnn"]["architecture"] = '8' # 5
    nparams["cnn"]["final_activation"] = 'linear'
    add_extra_params(nparams, extra_params)
    params['bow_fact_multilabel'] = copy.deepcopy(nparams) 

    #bow multiclass experiment freesound
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["evaluation"] = 'multiclass' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["dim"] = 397
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'fsd'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["val_from_file"] = False
    nparams["training"]["loss_func"] = 'categorical_crossentropy'
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["sparse"] = True
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["configuration"] = suffix
    nparams["cnn"]["architecture"] = '81' # 5
    #nparams["cnn"]["n_dense"] = 512
    nparams["cnn"]["final_activation"] = 'softmax'
    add_extra_params(nparams, extra_params)
    params['bow_multiclass'] = copy.deepcopy(nparams)     


    #Audio multiclass experiment fsd
    nparams = copy.deepcopy(models.params_5)
    nparams["dataset"]["evaluation"] = 'multiclass' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["dim"] = 397
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["cnn"]["architecture"] = '5'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["window"] = 3
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'fsd-s'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["training"]["val_from_file"] = False
    nparams["dataset"]["with_metadata"] = False
    nparams["dataset"]["only_metadata"] = False
    nparams["dataset"]["configuration"] = suffix
    #nparams["cnn"]["n_dense"] = 2048
    nparams["cnn"]["n_filters_1"] = 64
    nparams["cnn"]["n_filters_2"] = 128
    nparams["cnn"]["n_filters_3"] = 0
    nparams["cnn"]["n_filters_4"] = 0
    nparams["cnn"]["n_pool_2"] = (2,1)
    nparams["cnn"]["n_pool_3"] = (1,1)
    nparams["cnn"]["final_activation"] = 'softmax'
    nparams["dataset"]["meta-suffix"] = ""
    add_extra_params(nparams, extra_params)
    params['audio_multiclass'] = copy.deepcopy(nparams)


    #Fact multimodal bi metadata input
    nparams = copy.deepcopy(models.params_6)
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
    nparams["cnn"]["architecture"] = '813'
    nparams["cnn"]["n_dense"] = 512
    nparams["cnn"]["dropout_factor"] = 0.0
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'multi2deA2'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["meta-suffix2"] = meta_suffix2 #bow
    #nparams["cnn"]["n_metafeatures2"] = 256
    add_extra_params(nparams, extra_params)
    params['multi_bi_class_multilabel'] = copy.deepcopy(nparams)

    #Fact multimodal bi metadata input
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 50 #397
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '813'
    nparams["cnn"]["n_dense"] = 512
    nparams["cnn"]["dropout_factor"] = 0.7
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'multi2deA2'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["meta-suffix2"] = meta_suffix2 #bow
    #nparams["cnn"]["n_metafeatures2"] = 256
    add_extra_params(nparams, extra_params)
    params['multi_bi_fact_multilabel'] = copy.deepcopy(nparams)    


    #Fact multimodal bi metadata input
    nparams = copy.deepcopy(models.params_6)
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
    nparams["cnn"]["architecture"] = '814'
    nparams["cnn"]["n_dense"] = 512
    nparams["cnn"]["dropout_factor"] = 0.0
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'multi2deA2'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["meta-suffix2"] = meta_suffix2 #bow
    nparams["dataset"]["meta-suffix3"] = meta_suffix3 #bow
    #nparams["cnn"]["n_metafeatures2"] = 256
    add_extra_params(nparams, extra_params)
    params['multi_tri_class_multilabel'] = copy.deepcopy(nparams)

    #Fact multimodal bi metadata input
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 50 #397
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '814'
    nparams["cnn"]["n_dense"] = 512
    nparams["cnn"]["dropout_factor"] = 0.7
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'multi2deA2'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["dataset"]["meta-suffix2"] = meta_suffix2 #bow
    nparams["dataset"]["meta-suffix3"] = meta_suffix3 #bow
    #nparams["cnn"]["n_metafeatures2"] = 256
    add_extra_params(nparams, extra_params)
    params['multi_tri_fact_multilabel'] = copy.deepcopy(nparams)   

    '''
    #Multimodal experiment
    nparams = params_6.copy()
    nparams["dataset"]["fact"] = 'nmf'
    nparams["dataset"]["dim"] = 200
    nparams["training"]["loss_func"] = 'cosine'
    nparams["cnn"]["architecture"] = '6'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'W2'
    process(nparams,with_metadata=True,only_metadata=False,metadata_source='roviM')

    # CNN text experiment
    nparams = params_82.copy()
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 200
    nparams["training"]["loss_func"] = 'cosine'
    nparams["cnn"]["architecture"] = '82'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG'
    nparams["dataset"]["meta-suffix"] = 'w2v'
    process(nparams,with_metadata=True,only_metadata=True)

    nparams = params_6.copy()
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 200
    nparams["training"]["loss_func"] = 'cosine'
    nparams["cnn"]["architecture"] = '10'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSDA'
    nparams["dataset"]["meta-suffix"] = 'w2v-bow2'
    process(nparams,with_metadata=True,only_metadata=True)


    nparams = params_5.copy()
    nparams["dataset"]["fact"] = 'nmf'
    nparams["dataset"]["dim"] = 200
    nparams["training"]["loss_func"] = 'cosine'
    nparams["cnn"]["architecture"] = '5'
    nparams["cnn"]["n_kernel_1"] = (3,3)
    nparams["cnn"]["n_kernel_2"] = (3,3)
    nparams["cnn"]["n_kernel_3"] = (3,3)
    nparams["cnn"]["n_kernel_4"] = (3,3)
    nparams["cnn"]["n_pool_1"] = (4,2)
    nparams["cnn"]["n_pool_2"] = (4,2)
    nparams["cnn"]["n_pool_3"] = (2,2)
    nparams["cnn"]["n_pool_4"] = (3,3)
    nparams["cnn"]["n_filters_1"] = 512
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'W2'
    process(nparams)

    '''
    
    #XAVIER
    #Class experiment bow
    nparams = copy.deepcopy(models.params_6)
    nparams["dataset"]["evaluation"] = 'multilabel' # 
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["dim"] = 492
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '8'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'fsd'
    nparams['dataset']['sparse'] = False
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["training"]["val_from_file"] = False
    add_extra_params(nparams, extra_params)
    params['class_bow_xavier'] = copy.deepcopy(nparams)
    
    return params[suffix]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('suffix', default="class_bow", help='Suffix of experiment params')
    parser.add_argument('meta_suffix', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('meta_suffix2', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('meta_suffix3', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('extra_params', nargs='?', default="", help='Specific extra parameters')
    args = parser.parse_args()
    print args.extra_params
    params = get_configuration(args.suffix,args.meta_suffix,args.meta_suffix2,args.meta_suffix3,args.extra_params)
    process(params)
    # Tartarus Experiment

    #run(params,0)
    #qsub -o __class_w2v_512f -l h=node06 -v s="class_w2v",m="w2v",p="'cnn.num_filters=512&cnn.filter_sizes=(1,2,3)'" run.sub
    #qsub -o __class_bow -l h=node06 -v s="class_bow",m="bow-bi10k" run.sub
    #qsub -o __fact_audio -v s="audio" run.sub
    #qsub -o __class_audio_genre -v s="audio_genre" run.sub
    #qsub -o __fact_multi_G-hs-10k -v s="audio" -v s="fact_multi",m="G-hs-10k-babelfy" run.sub
    #qsub -o __fact_multi_model_434 -v s="fact_multi",m="model_434_pred" run.sub
    #qsub -o __fact_multi_model_434_5 -v s="multi_fact",m="multi_434_561" run.sub
    #qsub -o __fact_multi_model_289_5_S -l h=node06 -v s="multi_fact",m="model_289-pred_5" run.sub
    #qsub -o __fact_multi_bi_431_561 -l h=node06 -v s="multi_fact_bi",m="model_431-pred_5",p="model_561-pred_16" run2.sub
