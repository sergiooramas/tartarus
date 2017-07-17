from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import glob
import numpy as np
import os
import sys
sys.path.insert(0, '../')
import common
from load_w2v import clean_str

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

TEXT_DIR = common.DATA_DIR + "/text"
DATASET_NAME = 'SUPER'
index_file = common.INDEX_PATH + "index_text_%s.tsv" % DATASET_NAME
suffix = 'bow'
N_WORDS = 300
loadY = True

train = open(common.DATASETS_DIR + "/items_index_train_%s.tsv" % (DATASET_NAME)).read().splitlines()
test = open(common.DATASETS_DIR + "/items_index_test_%s.tsv" % (DATASET_NAME)).read().splitlines()
val = open(common.DATASETS_DIR + "/items_index_val_%s.tsv" % (DATASET_NAME)).read().splitlines()
texts = dict()

file2id = dict()
f=open(index_file)
for line in f.readlines():
    id, text_file = line.strip().split("\t")
    file2id[text_file] = id

print TEXT_DIR
files = glob.glob(TEXT_DIR+"/*.txt")
print len(files)
for file in files:
    filename = file[file.rfind("/")+1:]
    id = file2id[filename]
    text = open(file).read()
    sentences = text.split("\n")
    clean_sentences = [s.split(" ") for s in sentences]
    words = [word for s in clean_sentences for word in s]
    texts[id] = " ".join(words[:N_WORDS])

print texts

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', max_features=10000)
data_train = []
for i,item in enumerate(train):
    data_train.append(texts[item])
data_val = []
for i,item in enumerate(val):
    data_val.append(texts[item])
data_test = []
for i,item in enumerate(test):
    data_test.append(texts[item])


X_train = vectorizer.fit_transform(data_train)
X_val = vectorizer.transform(data_val)
X_test = vectorizer.transform(data_test)

print vectorizer.get_feature_names()[:100]

if not os.path.isdir(common.TRAINDATA_DIR):
    os.makedirs(common.TRAINDATA_DIR)
X_file = common.TRAINDATA_DIR+'/X_train_%s_%s' % (suffix,DATASET_NAME)
save_sparse_csr(X_file,csr_matrix(X_train))
X_file = common.TRAINDATA_DIR+'/X_test_%s_%s' % (suffix,DATASET_NAME)
save_sparse_csr(X_file,csr_matrix(X_test))
X_file = common.TRAINDATA_DIR+'/X_val_%s_%s' % (suffix,DATASET_NAME)
save_sparse_csr(X_file,csr_matrix(X_val))
