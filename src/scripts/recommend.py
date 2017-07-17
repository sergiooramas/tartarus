from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise, precision_recall_curve, average_precision_score, roc_curve
import json
import argparse
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import tempfile
import shutil
import os
import common
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WL=[]
sim_matrix=[]
seed_index=[]
song_index=[]

def do_process_rank(model_id,seed_index,song_index,th,i,with_wl):
    global WL
    global sim_matrix
    sim_list = sim_matrix[i,:]
    rank=np.argsort(sim_list)[::-1]
    added = 0
    for j in range(len(rank)):
        if ((not with_wl) or (with_wl and WL[i,rank[j]] == 0)) and sim_list[rank[j]] > th and added <= 300:
            nseed=seed_index[i]
            nsong=song_index[rank[j]]
            added += 1
            fw=open('rec/rec_%s_top300_th_std.tsv' % model_id,'a')
            fw.write(nseed+'\t'+nsong+'\t'+'%d\n' % int(round(sim_list[rank[j]],3)*1000))
            fw.close()
        elif sim_list[rank[j]] <= th or added > 300:
            break

def get_top_300(model_id,seed_index,song_index,th,with_wl):
    fw=open('rec/rec_%s_top300_th_std.tsv' % model_id,'w')
    Parallel(n_jobs=20)(delayed(do_process_rank)(model_id,seed_index,song_index,th,i,with_wl)
                           for i in range(len(seed_index)))

def max_subarray(A):
    max_ending_here = max_so_far = 0
    pos=0
    for i,x in enumerate(A):
        max_ending_here = max(0, max_ending_here + x)
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            pos = i
        #max_so_far = max(max_so_far, max_ending_here)
    return pos

def get_rec(models, ths, factors_suffix='', with_eval=True, with_rec=False):
    models = models.split(',')
    ths = ths.split(',')
    rec_lists = dict()
    topk = [1,3,5,10,15,25,50,100]
    for k in topk:
        rec_lists[k] = dict()
    for i, (model_id, th) in enumerate(zip(models,ths)):
        if th != '':
            th = float(th)
        if 'model' not in model_id:
            items = model_id.split('_')
            model_settings = dict()
            if 'pmi' in model_id:
                model_settings['fact'] = items[1]+'_'+items[2]
                model_settings['dim'] = int(items[3])
                model_settings['dataset'] = items[4]
            else:
                model_settings['fact'] = items[1]
                model_settings['dim'] = int(items[2])
                model_settings['dataset'] = items[3]
            factors_suffix_model = items[0]+'_'+model_settings['fact']+'_'+str(model_settings['dim'])+'_'+factors_suffix
        else:
            trained_models = pd.read_csv(common.DEFAULT_TRAINED_MODELS_FILE, sep='\t')
            model_config = trained_models[trained_models["model_id"] == model_id]
            if model_config.empty:
                raise ValueError("Can't find the model %s in %s" %
                                 (model_id, common.DEFAULT_TRAINED_MODELS_FILE))
            model_config = model_config.to_dict(orient="list")
            model_settings=eval(model_config['dataset_settings'][0])
            if factors_suffix != '':
                factors_suffix_model = model_id + '_spectro_' + factors_suffix
            else:
                factors_suffix_model = model_id
        seeds_dataset = model_settings['dataset']
        seed_factors=np.load(common.DATASETS_DIR+'/seed_factors_%s_%s_%sonly.npy' % (model_settings['fact'],model_settings['dim'],seeds_dataset))
        seed_index=open(common.DATASETS_DIR+'/seeds_index_%sonly.tsv' % seeds_dataset).read().splitlines()
        
        #if factors_suffix != '':
        #    factors_suffix = '_spectro_'+factors_suffix
        factors=np.load(common.FACTORS_DIR+'/factors_%s.npy' % (factors_suffix_model))
        factors_index=np.asarray(open(common.FACTORS_DIR+'/index_factors_%s.tsv' % (factors_suffix_model)).read().splitlines())

        sim_matrix = seed_factors.dot(normalize(np.nan_to_num(factors),copy=False).T)
        sim_matrixT = sim_matrix.T
        #sim_matrix = common.minmax_normalize(sim_matrix)
        print 'Computed similarity matrix'

        if with_eval:
            print 'Loading WL'
            WL = np.load(common.DATASETS_DIR+'/wl_matrix_%s.npy' % model_settings['dataset'])
            print 'Creating inverted index'
            song_index_wl=open(common.DATASETS_DIR+'/songs_index_%s.tsv' % model_settings['dataset']).read().splitlines()
            inv_song_index_wl = {k:v for v, k in enumerate(song_index_wl)}
            seed_index_wl=open(common.DATASETS_DIR+'/seeds_index_%s.tsv' % model_settings['dataset']).read().splitlines()
            inv_seed_index_wl = {k:v for v, k in enumerate(seed_index_wl)}
            WLf = WL[:,[inv_song_index_wl[song] for song in factors_index]]
            WLf = WLf[[inv_seed_index_wl[seed] for seed in seed_index],:]
            print 'Obtaining threshold'
            print WLf.shape
            print sim_matrix.shape
            good_scores = sim_matrix[WLf==1]
            th = good_scores.mean()
            std = good_scores.std()
            print 'Mean th',th
            print 'Std',std
            p, r, thresholds = precision_recall_curve(WLf.flatten(), sim_matrix.flatten())
            f = np.nan_to_num((2 * (p*r) / (p+r)) * (p>r))
            fth = thresholds[np.argmax(f)]
            print 'F th %.2f' % fth

        if with_rec:
            print 'Computing recommendations'
            #for i,j in zip(*np.where(sim_matrix>th)):
            #    rec_list.setdefault(seed_index[i],[]).append((factors_index[j],int(round(sim_matrix[i,j],3)*1000)))
            
            #get_top_300(factors_suffix, seed_index, factors_index, th, with_eval)
            
            for i in range(sim_matrixT.shape[0]):
                rank=np.argsort(sim_matrixT[i])[::-1]
                for k in topk:
                    for j in rank[:k]:
                        if sim_matrixT[i,j] > th:
                            rec_lists[k].setdefault(seed_index[j],[]).append((factors_index[i],int(round(sim_matrixT[i,j],3)*1000)))

    if with_rec:
        for k, rec_list in rec_lists.iteritems():
            print k
            print len(rec_list)
            suffix = "_".join([m[m.find('_')+1:] for m in models])
            fw = open(common.REC_DIR+'/rec_models_%s_%s_std_top%s.tsv' % (suffix,factors_suffix,k),'w')
            n=0
            for seed, items_list in rec_list.iteritems():
                ordered = sorted(items_list, key=lambda k: k[1], reverse=True)
                for items in ordered[:300]:
                    fw.write('%s\t%s\t%s\n' % (seed,items[0],items[1]))
                    n+=1
            print 'Recommendations',n

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create recommendation list',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e',
                        '--eval',
                        dest="with_eval",
                        help='Compute the threshold',
                        action='store_true',
                        default=False)
    parser.add_argument('-r',
                        '--recommend',
                        dest="with_rec",
                        help='Compute the threshold',
                        action='store_true',
                        default=False)
    parser.add_argument('-m',
                        dest="models",
                        help='List of models separated with commas, for 2,4,8 and 16 weeks',
                        default='')
    parser.add_argument('-th',
                        dest="ths",
                        help='List of thresholds separated with commas, for 2,4,8 and 16 weeks',
                        default='')
    parser.add_argument('-fs',
                        '--factors_suffix',
                        dest="factors_suffix",
                        help='Suffix of the factors file',
                        default='')
    args = parser.parse_args()
    get_rec(args.models,args.ths,factors_suffix=args.factors_suffix,with_eval=args.with_eval,with_rec=args.with_rec)
