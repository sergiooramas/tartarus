from sklearn.metrics import accuracy_score, average_precision_score, coverage_error, label_ranking_average_precision_score, pairwise, roc_curve, auc, roc_auc_score, average_precision_score,precision_recall_curve, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
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
import time
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
#matplotlib.rcParams['ylabel.size'] = 20
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

RANDOM_SELECTION=False
PLOT_MATRIX=False
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

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #plt.title(title, **csfont)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=90)
        plt.yticks(tick_marks, labels)
        #plt.xticks(tick_marks, rotation=90)
        #plt.yticks(tick_marks) 
        csfont = {'fontname':'Times', 'fontsize':'17'}
        plt.tight_layout()
        plt.ylabel('True label', **csfont)
        plt.xlabel('Predicted label', **csfont)

def evaluate(model_id,model_settings,str_config,predictions,predictions_index,binary_classification=False,start_user=0,num_users=1000,get_roc=False,get_map=False,get_p=False,batch=False):  
    global test_matrix
    global sim_matrix 
    local_path = common.DATASETS_DIR
    # Load factors and ground truth
    if model_settings['evaluation'] in ['binary','multiclass','multilabel']:
        actual_matrix = np.load(common.DATASETS_DIR+'/y_test_%s_%s_%s.npy' % (model_settings['fact'],model_settings['dim'],model_settings['dataset']))            
        good_classes = np.nonzero(actual_matrix.sum(0))[0]
        actual_matrix_roc = actual_matrix_map = actual_matrix[:,good_classes]
    else:
        index_matrix = open(common.DATASETS_DIR+'/items_index_test_%s.tsv' % (model_settings['dataset'])).read().splitlines()
        index_matrix_inv = dict((item,i) for i,item in enumerate(index_matrix))
        index_good = [index_matrix_inv[item] for item in predictions_index]
        actual_matrix = load_sparse_csr(local_path+'/matrix_test_%s.npz' % model_settings['dataset'])
        actual_matrix_map = actual_matrix[:,start_user:min(start_user+num_users,actual_matrix.shape[1])]
        actual_matrix_roc = actual_matrix_map[index_good] # Items-Users matrix        
    if model_settings['fact'] in ['pmi','als']:
        if model_settings['evaluation'] == 'recommendation':
            user_factors = np.load(local_path+'/user_factors_%s_%s_%s.npy' % (model_settings['fact'],model_settings['dim'],model_settings['dataset']))
        else:
            user_factors = np.load(local_path+'/class_factors_%s_%s_%s.npy' % (model_settings['fact'],model_settings['dim'],model_settings['dataset']))
        user_factors = user_factors[start_user:min(start_user+num_users,user_factors.shape[0])]        

    # Predicted matrix
    if model_settings['fact'] == 'class':
        predicted_matrix_map = predictions
        predicted_matrix_roc = predictions[:,good_classes]
    else:
        if model_settings['fact'] == 'pmi':
            predicted_matrix_roc = pairwise.cosine_similarity(np.nan_to_num(predictions),np.nan_to_num(user_factors))
            predicted_matrix_map = predicted_matrix_roc.copy()
        else:
            predicted_matrix_roc = normalize(np.nan_to_num(predictions)).dot(user_factors.T) # Items-Users
            predicted_matrix_map = predicted_matrix_roc.copy().T # Users-Items
 
    if get_map and model_settings['evaluation'] in ['recommendation']:
        actual_matrix_map = actual_matrix_roc.T.toarray()
    if get_roc and model_settings['evaluation'] in ['recommendation']:
        actual_matrix_roc.data = actual_matrix_roc.data / actual_matrix_roc.data
        good_classes = np.nonzero(actual_matrix_roc.sum(axis=0))[1]
        actual_matrix_roc = actual_matrix_roc[:,good_classes].toarray()
        predicted_matrix_roc = predicted_matrix_roc[:,good_classes]
    
    print 'Computed prediction matrix'
    print model_id
    print model_settings['dataset']
    print model_settings['configuration']
    if 'meta-suffix' in model_settings:
        print model_settings['meta-suffix']

    if not batch:
        if not os.path.exists(common.RESULTS_DIR):
            os.makedirs(common.RESULTS_DIR)
        fw=open(common.RESULTS_DIR+'/eval_results.txt','a')
        fw.write(model_id+'\n')
        fw.write(model_settings['dataset']+"\n")
        fw.write(model_settings['configuration']+"\n")
        if 'meta-suffix' in model_settings:
            fw.write(model_settings['meta-suffix']+"\n")

    print model_settings['evaluation']
    if model_settings['evaluation'] in ['binary','multiclass']:
        print "entro y no deberia"
        actual_matrix_map = actual_matrix_map
        labels = open(common.DATASETS_DIR+"/genre_labels_%s.tsv" % model_settings['dataset']).read().splitlines()
        predicted_matrix_binary = np.zeros(predicted_matrix_roc.shape)
        predicted_labels = []
        actual_labels = []
        for i in range(predicted_matrix_roc.shape[0]):
            predicted_matrix_binary[i,np.argmax(predicted_matrix_roc[i])] = 1
            predicted_labels.append(labels[np.argmax(predicted_matrix_roc[i])])
            actual_labels.append(labels[np.argmax(actual_matrix_roc[i])])

        acc = accuracy_score(actual_labels,predicted_labels)
        prec = precision_score(actual_labels,predicted_labels,average='macro',labels=labels)
        recall = recall_score(actual_labels,predicted_labels,average='macro',labels=labels)
        f1 = f1_score(actual_labels,predicted_labels,average='macro',labels=labels)
        print 'Accuracy', acc
        print "Precision %.3f\tRecall %.3f\tF1 %.3f" % (prec,recall,f1)
        print [(i,l) for i,l in enumerate(labels)]
        micro_prec = precision_score(actual_labels,predicted_labels,average='micro',labels=labels)
        print "Micro precision", micro_prec
        print classification_report(actual_labels,predicted_labels,target_names=labels)

        if PLOT_MATRIX:
            # Normalize the confusion matrix by row (i.e by the number of samples in each class)
            cm = confusion_matrix(actual_labels,predicted_labels,labels=labels)
            #print_cm(cm, labels)
            #plt.figure()
            #plot_confusion_matrix(cm, title='Not Normalized confusion matrix')
            #plt.savefig('confusion_notNormalized.png')

            #M = cm.sum(axis=1)
            #print M
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print('Normalized confusion matrix')
            #print_cm(cm, labels)
            plt.figure()
            plot_confusion_matrix(cm, labels, title='Normalized confusion matrix')
            plt.savefig('confusion_%s.png' % model_id)
    if batch:
        try:
            if not os.path.exists(common.DATA_DIR+"/eval/%s-%s/" % (model_id,num_users)):
                os.makedirs(common.DATA_DIR+"/eval/%s-%s/" % (model_id,num_users))
        except:
            pass

    # MAP@k
    if get_map:
        fname = common.DATA_DIR+"/eval/%s-%s/map_%s.txt" % (model_id,num_users,start_user)
        if not os.path.isfile(fname) or not batch:
            k = 500
            actual = [list(np.where(actual_matrix_map[i] > 0)[0]) for i in range(actual_matrix_map.shape[0])]
            predicted = list([list(l)[::-1][:k] for l in predicted_matrix_map.argsort(axis=1)])
            map500 = mapk(actual, predicted, k)
            if batch:
                fw_map = open(fname,"w")
                fw_map.write(str(map500))
                fw_map.close()
            else:
                fw.write('MAP@500: %.5f\n' % map500)
            print 'MAP@500: %.5f' % map500

    # ROC
    if get_roc:
        fname = common.DATA_DIR+"/eval/%s-%s/roc_%s.txt" % (model_id,num_users,start_user)
        #if not os.path.isfile(fname):
        roc_auc = roc_auc_score(actual_matrix_roc,predicted_matrix_roc)
        print 'ROC-AUC: %.5f' % roc_auc
        pr_auc = average_precision_score(actual_matrix_roc,predicted_matrix_roc)
        print 'PR-AUC: %.5f' % pr_auc
        if batch:
            fw_roc = open(fname,"w")
            fw_roc.write(str(roc_auc))
            fw_roc.close()
        else:
            fw.write('ROC-AUC: %.5f\n' % roc_auc)
            fw.write('PR-AUC: %.5f\n' % pr_auc)

    # P@k
    if get_p:
        ks = [1,3,5]
        folder = tempfile.mkdtemp()
        p = np.memmap(os.path.join(folder, 'p'), dtype='f',shape=(len(ks),predicted_matrix_map.shape[0]), mode='w+')
        adiv = np.memmap(os.path.join(folder, 'adiv'), dtype='f',shape=(len(ks),predicted_matrix_map.shape[1]), mode='w+')
        ndcg = np.memmap(os.path.join(folder, 'ndcg'), dtype='f',shape=(len(ks),predicted_matrix_map.shape[0]), mode='w+')
        Parallel(n_jobs=20)(delayed(do_process)(i,predicted_matrix_map[i,:],actual_matrix_map[i,:],ks,p,ndcg,adiv)
                               for i in range(0,predicted_matrix_map.shape[0]))
        line_p=[]
        line_n=[]
        line_a=[]
        for i,k in enumerate(ks):
            pk = p[i].mean()
            nk = ndcg[i].mean()
            ak = adiv[i].sum() / predicted_matrix_map.shape[1]
            print 'P@%d: %.2f' % (k, pk)
            print 'nDCG@%d: %.2f' % (k, nk)
            print 'ADiv/C@%d: %.2f' % (k, ak)
            fw.write('P@%d: %.2f\n' % (k, pk))
            fw.write('nDCG@%d: %.2f\n' % (k, nk))
            fw.write('ADiv/C@%d: %.2f\n' % (k, ak))
            line_p.append(pk)
            line_n.append(nk)
            line_a.append(ak)
        try:
            shutil.rmtree(folder)
        except:
            print("Failed to delete: " + folder)

    if not batch:
        fw.write('\n')
        fw.write(str_config)
        fw.write('\n')
        fw.close()
    print model_id
    
def do_eval(model_id, get_roc=False, get_map=False, get_p=False, start_user=0, num_users=10000, batch=False, predictions=[], predictions_index=[], meta=""):
    if 'model' not in model_id:
        items = model_id.split('_')
        model_settings = dict()
        model_settings['fact'] = items[1]
        model_settings['dim'] = int(items[2])
        model_settings['dataset'] = items[3]
        model_arch = dict()
        if model_settings['fact'] == 'class':
            model_arch['final_activation'] = 'softmax'
        else:
            model_arch['final_activation'] = 'linear'            
        model_settings['configuration'] = "gt"
        str_config = model_id
    else:
        read = False
        x=0
        while not read or x >= 100:
            try:
                trained_models = pd.read_csv(common.DEFAULT_TRAINED_MODELS_FILE, sep='\t')
                model_config = trained_models[trained_models["model_id"] == model_id]
                if model_config.empty:
                    raise ValueError("Can't find the model %s in %s" %
                                     (model_id, common.DEFAULT_TRAINED_MODELS_FILE))
                model_config = model_config.to_dict(orient="list")
                read = True
            except:
                pass
                x+=1
                time.sleep(1)
        model_settings=eval(model_config['dataset_settings'][0])
        model_arch=eval(model_config['model_arch'][0])
        model_training=eval(model_config['training_params'][0])
        str_config = json.dumps(model_settings)+"\n"+json.dumps(model_arch)+"\n"+json.dumps(model_training)+"\n"
        if meta != "" and "meta_suffix" not in model_settings:
            model_settings["meta-suffix"] = meta
        model_settings["loss"] = model_training['loss_func']
    if predictions==[]:
        predictions=np.load(common.PREDICTIONS_DIR+'/pred_%s.npy' % (model_id))
        predictions_index=open(common.FACTORS_DIR+'/index_pred_%s.tsv' % (model_id)).read().splitlines()

    binary_classification = False
    if model_settings["evaluation"] == "binary":
        binary_classification = True

    evaluate(model_id, model_settings, str_config, predictions, predictions_index, binary_classification, start_user, num_users, get_roc, get_map, get_p, batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluates the model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest="model_id",
                        type=str,
                        help='Identifier of the Model to evaluate')
    parser.add_argument('-roc',
                        '--roc',
                        dest="get_roc",
                        help='Roc-auc evaluation',
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
    parser.add_argument('-su',
                        '--start_user',
                        dest="start_user",
                        type=int,
                        help='First user to start evaluation',
                        default=0)
    parser.add_argument('-nu',
                        '--num_users',
                        dest="num_users",
                        type=int,
                        help='Number of users for evaluation',
                        default=1000)
    parser.add_argument('-ls',
                        '--local_storage',
                        dest="use_local_storage",
                        help='Set if use local copy of matrices in the node',
                        action='store_true',
                        default=False)
    parser.add_argument('-b',
                        '--batch',
                        dest="batch",
                        help='Batch process in cluster',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    do_eval(args.model_id,args.get_roc,args.get_map,args.get_p,args.start_user,args.num_users,args.batch,meta=args.meta)
