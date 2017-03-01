from sklearn.metrics import accuracy_score, average_precision_score, coverage_error, label_ranking_average_precision_score, label_ranking_loss, pairwise, roc_curve, auc, roc_auc_score, average_precision_score,precision_recall_curve
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import pandas as pd
import argparse
from joblib import Parallel, delayed
import os
import tempfile
import shutil
import common
import json

RANDOM_SELECTION=False
test_matrix=[]
test_matrix_imp = []
sim_matrix=[]

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def dcg_at_k(r, k):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def precision_at_k(r, k):
    """Score is precision @ k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    """
    #return np.mean(np.asarray(r)[:k])
    ### ALS evaluation
    rk = r[:k]
    return rk[rk>0].shape[0]*1.0/k


def do_process_map(i,K,mapk):
    sim_list = sim_matrix[:,i]
    rank = np.argsort(sim_list)[::-1]
    pred = np.asarray(test_matrix[rank[:K],i].todense()).reshape(-1)
    p=0.0
    for k in range(1,K+1):
        p+=precision_at_k(pred,k)
    mapk[i]=p/K

def do_process(i,predicted_row,actual_row,ks,p,ndcg,adiv):
    rank = np.argsort(predicted_row)[::-1]
    pred = np.asarray(actual_row[rank[:ks[-1]]]).reshape(-1)
    for j,k in enumerate(ks):
        p[j][i] += precision_at_k(pred,k)
        ndcg[j][i] += ndcg_at_k(pred,k)
        adiv[j][rank[:k]] = 1

def evaluate(model_id,model_settings,str_config,factors,factors_index):  
    global test_matrix
    global sim_matrix 

    print factors.shape

    if model_settings['fact'] == 'class':
        actual_matrix = np.load(common.DATASETS_DIR+'/item_factors_test_class_300_%s.npy' % (model_settings['dataset']))
        good_classes = np.nonzero(actual_matrix.sum(0))[0]
        print len(good_classes)
        actual_matrix = actual_matrix[:,good_classes]
        actual_matrix_roc = actual_matrix
    else:
        user_factors = np.load(common.DATASETS_DIR+'/user_factors_%s_%s_%s.npy' % (model_settings['fact'],model_settings['dim'],model_settings['dataset']))[:1000,:]
        actual_matrix = load_sparse_csr(common.DATASETS_DIR+'/matrix_test_%s.npz' % model_settings['dataset'])[:,:1000].toarray().T
        actual_matrix_roc = actual_matrix.copy().T
        actual_matrix_roc[np.where(actual_matrix_roc > 0)] = 1
        good_classes = np.nonzero(actual_matrix_roc.sum(0))[0]
        print len(good_classes)
        actual_matrix_roc = actual_matrix_roc[:,good_classes]

    print actual_matrix.shape

    #items_index = open(common.DATASETS_DIR+'/items_index_test_%s.tsv' % model_settings['dataset']).read().splitlines()

    if model_settings['fact'] == 'class':
        predicted_matrix = factors
        predicted_matrix = predicted_matrix[:,good_classes]
        predicted_matrix_roc = predicted_matrix
    else:
        if model_settings['fact'] == 'pmi':
            predicted_matrix = pairwise.cosine_similarity(np.nan_to_num(factors),np.nan_to_num(user_factors))
        else:
            #sim_matrix = user_factors.dot(normalize(np.nan_to_num(factors.T),copy=False))
            #sim_matrix = normalize(np.nan_to_num(factors)).dot(user_factors.T)
            predicted_matrix = normalize(np.nan_to_num(factors)).dot(user_factors.T).T
        predicted_matrix_roc = predicted_matrix.T
    print 'Computed similarity matrix'


    # ROC
    #actual_matrix[np.where(actual_matrix > 0)] = 1
    #fpr,tpr,_=roc_curve(actual_matrix.ravel(),predicted_matrix.ravel())
    #roc_auc = auc(fpr, tpr)
    #print roc_auc
    roc_auc = roc_auc_score(actual_matrix_roc,predicted_matrix_roc)
    print 'ROC-AUC',roc_auc

    # MAP@k
    k = 500
    actual = [list(np.where(actual_matrix[i] > 0)[0]) for i in range(actual_matrix.shape[0])]
    predicted = list([list(l)[::-1][:k] for l in predicted_matrix.argsort(axis=1)])
    map500 = mapk(actual, predicted, k)
    print 'MAP',map500

    # P@k
    ks = [1,3,5]
    folder = tempfile.mkdtemp()
    p = np.memmap(os.path.join(folder, 'p'), dtype='f',shape=(len(ks),predicted_matrix.shape[0]), mode='w+')
    adiv = np.memmap(os.path.join(folder, 'adiv'), dtype='f',shape=(len(ks),predicted_matrix.shape[1]), mode='w+')
    ndcg = np.memmap(os.path.join(folder, 'ndcg'), dtype='f',shape=(len(ks),predicted_matrix.shape[0]), mode='w+')
    Parallel(n_jobs=20)(delayed(do_process)(i,predicted_matrix[i],actual_matrix[i],ks,p,ndcg,adiv)
                           for i in range(0,predicted_matrix.shape[0]))


    
    fw=open(common.DATA_DIR+'/results/eval_results.txt','a')
    print model_id
    print model_settings['dataset']
    print model_settings['configuration']
    fw.write(model_id+'\n')
    fw.write(model_settings['dataset']+"\n")
    fw.write(model_settings['configuration']+"\n")
    if 'meta-suffix' in model_settings:
        print model_settings['meta-suffix']
        fw.write(model_settings['meta-suffix']+"\n")
    line_p=[]
    line_n=[]
    line_a=[]
    for i,k in enumerate(ks):
        pk = p[i].mean()
        nk = ndcg[i].mean()
        ak = adiv[i].sum() / predicted_matrix.shape[1]
        print 'P@%d: %.2f' % (k, pk)
        print 'nDCG@%d: %.2f' % (k, nk)
        print 'ADiv@%d: %.2f' % (k, ak)
        fw.write('P@%d: %.2f\n' % (k, pk))
        fw.write('nDCG@%d: %.2f\n' % (k, nk))
        fw.write('ADiv@%d: %.2f\n' % (k, ak))
        line_p.append(pk)
        line_n.append(nk)
        line_a.append(ak)
    #mapkn = mapk.mean()
    print model_id
    print 'MAP@500: %.5f' % map500
    print 'ROC-AUC: %.5f' % roc_auc
    fw.write('MAP@500: %.5f\n' % map500)
    fw.write('ROC-AUC: %.5f\n' % roc_auc)
    #fw.write('LRAP users: %.2f/n' % lrap)
    #fw.write('LRAP items: %.2f/n' % lraps)
    #fw.write("["+",".join(["%.2f" % p for p in line_p])+","+",".join(["%.2f" % p for p in line_n])+","+",".join(["%.2f" % p for p in line_a])+"]\n")
    fw.write('\n')
    fw.write(str_config)
    fw.write('\n')
    fw.close()
    try:
        shutil.rmtree(folder)
    except:
        print("Failed to delete: " + folder)

def do_eval(model_id, get_knn=True, get_map=False, get_p=False, factors=[], factors_index=[], meta=""):
    if 'model' not in model_id:
        items = model_id.split('_')
        model_settings = dict()
        model_settings['fact'] = items[1]
        model_settings['dim'] = int(items[2])
        model_settings['dataset'] = items[3]
    else:
        trained_models = pd.read_csv(common.DEFAULT_TRAINED_MODELS_FILE, sep='\t')
        model_config = trained_models[trained_models["model_id"] == model_id]
        if model_config.empty:
            raise ValueError("Can't find the model %s in %s" %
                             (model_id, common.DEFAULT_TRAINED_MODELS_FILE))
        model_config = model_config.to_dict(orient="list")
        model_settings=eval(model_config['dataset_settings'][0])
        model_arch=eval(model_config['model_arch'][0])
        model_training=eval(model_config['training_params'][0])
    str_config = json.dumps(model_settings)+"\n"+json.dumps(model_arch)+"\n"+json.dumps(model_training)+"\n"
    if meta != "" and "meta_suffix" not in model_settings:
        model_settings["meta-suffix"] = meta
    model_settings["loss"] = model_training['loss_func']
    if factors==[]:
        factors=np.load(common.FACTORS_DIR+'/factors_%s.npy' % (model_id))
        #factors=np.load(common.DATASETS_DIR+'/item_factors_%s_%s_%s.npy' % (model_settings['fact'],model_settings['dim'],model_settings['dataset']))
        #factors_index=open(common.FACTORS_DIR+'/index_factors_%s.tsv' % (model_id)).read().splitlines()
        factors_index=open(common.DATASETS_DIR+'/items_index_test_%s.tsv' % (model_settings['dataset'])).read().splitlines()

    #if get_map:
    #    map_eval(model_id,model_settings,factors,factors_index)

    evaluate(model_id, model_settings, str_config, factors, factors_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluates the model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest="model_id",
                        type=str,
                        help='Identifier of the Model to evaluate')
    parser.add_argument('-knn',
                        '--knn',
                        dest="get_knn",
                        help='Knn evaluation',
                        action='store_true',
                        default=False)
    parser.add_argument('-map',
                        '--map',
                        dest="get_map",
                        help='Map evaluation',
                        action='store_true',
                        default=False)
    parser.add_argument('-p',
                        '--precision',
                        dest="get_p",
                        help='Precision evaluation',
                        action='store_true',
                        default=False)
    parser.add_argument('-ms',
                        '--meta',
                        dest="meta",
                        help='Meta suffix',
                        default="")
    args = parser.parse_args()
    do_eval(args.model_id,args.get_knn,args.get_map,args.get_p,meta=args.meta)
