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

def get_configuration(suffix, meta_suffix='bow', extra_params=''):
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
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
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
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '8'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG'
    nparams["dataset"]["meta-suffix"] = meta_suffix #bow
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
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = 82
    nparams["cnn"]["sequence_length"] = 300
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG'
    nparams["dataset"]["meta-suffix"] = meta_suffix #w2v
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
    nparams["dataset"]["fact"] = 'als'
    nparams["dataset"]["dim"] = 200
    nparams["training"]["loss_func"] = 'cosine'
    nparams["cnn"]["architecture"] = '5'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MSD-AG-S'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["dataset"]["with_metadata"] = False
    nparams["dataset"]["only_metadata"] = False
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["meta-suffix"] = ""
    add_extra_params(nparams, extra_params)
    params['audio'] = copy.deepcopy(nparams)

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
    add_extra_params(nparams, extra_params)
    params['fact_multi'] = copy.deepcopy(nparams)

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
    return params[suffix]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('suffix', default="class_bow", help='Suffix of experiment params')
    parser.add_argument('meta_suffix', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('extra_params', nargs='?', default="", help='Specific extra parameters')
    args = parser.parse_args()
    print args.extra_params
    params = get_configuration(args.suffix,args.meta_suffix,args.extra_params)
    process(params)
    # Tartarus Experiment

    #run(params,0)
    #qsub -o __class_w2v_512f -l h=node06 -v s="class_w2v",m="w2v",p="'cnn.num_filters=512&cnn.filter_sizes=(1,2,3)'" run.sub
    #qsub -o __class_bow -l h=node06 -v s="class_bow",m="bow-bi10k" run.sub
    #qsub -o __fact_audio -v s="audio" run.sub
    #qsub -o __fact_multi_G-hs-10k -v s="audio" -v s="fact_multi",m="G-hs-10k-babelfy" run.sub
    #qsub -o __fact_multi_model_434 -v s="audio" -v s="fact_multi",m="model_434_pred" run.sub
