from scipy.sparse import csr_matrix
import numpy as np
import re
import itertools
from collections import Counter
import sys
sys.path.insert(0, '../')
import common
from gensim.models import word2vec
from os.path import join, exists, split
from nltk import sent_tokenize
import os
import numpy as np
import pickle
import json
import glob


DATASET_NAME = 'SUPER'
suffix = 'w2v'
TEXT_DIR = common.DATA_DIR + "/text/"
index_file = common.INDEX_PATH + "index_text_%s.tsv" % DATASET_NAME
GOOGLE_VECTORS = False
SEQUENCE_LENGTH = 500



def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
   
    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {str:int}
    num_features    # Word vector dimensionality                      
    min_word_count  # Minimum word count                        
    context         # Context window size 
    """
    if GOOGLE_VECTORS:
        embedding_model = word2vec.Word2Vec.load_word2vec_format('word2vec_models/GoogleNews-vectors-negative300.bin', binary=True)
    else:
        model_dir = 'word2vec_models'
        model_name = "{:d}features_{:d}minwords_{:d}context_suffix".format(num_features, min_word_count, context, suffix)
        model_name = join(model_dir, model_name)
        if exists(model_name):
            embedding_model = word2vec.Word2Vec.load(model_name)
            print 'Loading existing Word2Vec model \'%s\'' % split(model_name)[-1]
        else:
            # Set values for various parameters
            num_workers = 2       # Number of threads to run in parallel
            downsampling = 1e-3   # Downsample setting for frequent words
            
            # Initialize and train the model
            print "Training Word2Vec model..."
            #sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
            sentences = sentence_matrix
            embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, \
                                size=num_features, min_count = min_word_count, \
                                window = context, sample = downsampling)
            
            # If we don't plan to train the model any further, calling 
            # init_sims will make the model much more memory-efficient.
            embedding_model.init_sims(replace=True)
            
            # Saving the model for later use. You can load it later using Word2Vec.load()
            if not exists(model_dir):
                os.mkdir(model_dir)
            print 'Saving Word2Vec model \'%s\'' % split(model_name)[-1]
            embedding_model.save(model_name)
    
    #  add unknown words
    embedding_weights = [np.array([embedding_model[w] if w in embedding_model\
                                                        else np.random.uniform(-0.25,0.25,embedding_model.vector_size)\
                                                        for w in vocabulary_inv])]
    return embedding_weights


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_dash(word):
    if len(word) > 0:
        if word[0] == "-":
            word = " "+word[1:]
        if word[-1] == "-":
            word = word[:-1]+" "  
    return word

def get_sentences_entities(artist):
    ner = json.load(open(nel_dir+artist+".json"))
    new_sentences = ["" for i in range(len(ner))]
    for sentence in ner:
        s = sentence['text']
        for entity in sentence['entities']:
            if " " in entity['label']:
                middle = s[entity['startChar']:entity['endChar']].replace(" ","-")             
                s = s[:entity['startChar']] + middle + s[entity['endChar']:]
        new_sentences[sentence['index']] = s
    return new_sentences

def load_data_set(index):
    id2file = dict()
    f=open(index_file)
    for line in f.readlines():
        id, text_file = line.strip().split("\t")
        id2file[id] = text_file

    texts = []
    all_sentences = []
    for item in index:
        file = TEXT_DIR+id2file[item]
        text = open(file).read()
        sentences = text.split("\n")

        clean_sentences = [clean_str(s).split(" ") for s in sentences]
        all_sentences.extend(clean_sentences)

        clean_words = [word for s in clean_sentences for word in s]
        texts.append(clean_words)

    return texts, all_sentences


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    #sequence_length = max(len(x) for x in sentences)
    #sequence_length = min(sequence_length,SEQUENCE_LENGTHE)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = SEQUENCE_LENGTH - len(sentence)
        if num_padding > 0:
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:SEQUENCE_LENGTH]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    #x = np.array([[vocabulary[word] for word in sentence if word in vocabulary] for sentence in sentences])
    x = np.array([[vocabulary[word] if word in vocabulary\
                                                        else 0\
                                                        for word in sentence] for sentence in sentences])
    return x


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    train_index = open(common.DATASETS_DIR + "/items_index_train_%s.tsv" % (DATASET_NAME)).read().splitlines()
    val_index = open(common.DATASETS_DIR + "/items_index_val_%s.tsv" % (DATASET_NAME)).read().splitlines()
    test_index = open(common.DATASETS_DIR + "/items_index_test_%s.tsv" % (DATASET_NAME)).read().splitlines()

    documents, sentences = load_data_set(train_index)
    documents_padded = pad_sentences(documents)
    vocabulary, vocabulary_inv = build_vocab(documents_padded)
    x_train = build_input_data(documents_padded, vocabulary)

    documents, _ = load_data_set(val_index)
    documents_padded = pad_sentences(documents)
    x_val = build_input_data(documents_padded, vocabulary)

    documents, _ = load_data_set(test_index)
    documents_padded = pad_sentences(documents)
    x_test = build_input_data(documents_padded, vocabulary)

    return [x_train, x_val, x_test, vocabulary, vocabulary_inv, sentences]

if __name__ == '__main__':
    x_train, x_val, x_test, vocabulary, vocabulary_inv, sentences = load_data()
    print x_train.shape
    print len(vocabulary)

    if not os.path.isdir(common.TRAINDATA_DIR):
        os.makedirs(common.TRAINDATA_DIR)
    embedding_weights = train_word2vec(sentences, vocabulary_inv)
    pickle.dump(embedding_weights,open(common.TRAINDATA_DIR+'/embedding_weights_%s_%s.pk' % (suffix,DATASET_NAME),'wb'))
    X_file = common.TRAINDATA_DIR+'/X_train_%s_%s' % (suffix,DATASET_NAME)
    np.save(X_file,x_train)
    X_file = common.TRAINDATA_DIR+'/X_val_%s_%s' % (suffix,DATASET_NAME)
    np.save(X_file,x_val)
    X_file = common.TRAINDATA_DIR+'/X_test_%s_%s' % (suffix,DATASET_NAME)
    np.save(X_file,x_test)
    print "done"

