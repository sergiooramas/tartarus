from keras.layers import Dense, Dropout, Activation, Flatten, Permute, Lambda, Input, merge, BatchNormalization, Embedding, LSTM, Bidirectional, Reshape, GRU, Merge
from keras.layers import Convolution1D, GlobalMaxPooling1D, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, MaxPooling1D
from keras.regularizers import l2
from keras import regularizers
from keras.models import Sequential, Model
import logging
from keras import backend as K
import keras
import theano.tensor as T
import numpy as np
import pickle


params_5 = {
    # dataset params
    'dataset' : {
        'fact' : 'nmf',
        'dim' : 200,
        'dataset' : 'W2',
        'window' : 15,
        'nsamples' : 'all',
        'npatches' : 3
    },

    # training params
    'training' : {
        'decay' : 1e-6,
        'learning_rate' : 0.1,
        'momentum' : 0.95,
        'n_epochs' : 100,
        'n_minibatch' : 32,
        'nesterov' : True,
        'validation' : 0.1,
        'test' : 0.1,
        'loss_func' : 'cosine',
        'optimizer' : 'adam'
    },
    # cnn params
    'cnn' : {
        'dropout_factor' : 0.5,
        'n_dense' : 0,
        'n_filters_1' : 1024,
        'n_filters_2' : 1024,
        'n_filters_3' : 2048,
        'n_filters_4' : 2048,
        'n_kernel_1' : (4, 96),
        'n_kernel_2' : (4, 1),
        'n_kernel_3' : (4, 1),
        'n_kernel_4' : (1, 1),
        'n_out' : '',
        'n_pool_1' : (4, 1),
        'n_pool_2' : (4, 1),
        'n_pool_3' : (1, 1),
        'n_pool_4' : (1, 1),
        'n_frames' : '',
        'n_mel' : 96,
        'architecture' : 2
    },
    'predicting' : {
        'trim_coeff' : 0.15
    },
    'evaluating' : {
        'get_map' : False,
        'get_p' : True,
        'get_knn' : False
    }
}

def get_model_5(params):
    model = Sequential()
    model.add(Convolution2D(params["n_filters_1"], params["n_kernel_1"][0],
                            params["n_kernel_1"][1],
                            border_mode='valid',
                            input_shape=(1, params["n_frames"],
                                         params["n_mel"]),
                            init="uniform"))
    #model.add(BatchNormalization())
    model.add(Activation("relu"))
    logging.debug("Input CNN: %s" % str(model.input_shape))
    logging.debug("Output Conv2D: %s" % str(model.output_shape))

    model.add(MaxPooling2D(pool_size=(params["n_pool_1"][0],
                                      params["n_pool_1"][1])))
    logging.debug("Output MaxPool2D: %s" % str(model.output_shape))
    model.add(Dropout(params["dropout_factor"]))

    #model.add(Permute((3,2,1)))

    model.add(Convolution2D(params["n_filters_2"], params["n_kernel_2"][0],
                            params["n_kernel_2"][1],
                            border_mode='valid',
                            init="uniform"))
    #model.add(BatchNormalization())
    model.add(Activation("relu"))
    #logging.debug("Input CNN: %s" % str(model.input_shape))
    logging.debug("Output Conv2D: %s" % str(model.output_shape))

    model.add(MaxPooling2D(pool_size=(params["n_pool_2"][0],
                                      params["n_pool_2"][1])))
    logging.debug("Output MaxPool2D: %s" % str(model.output_shape))
    model.add(Dropout(params["dropout_factor"]))

    #model.add(Permute((3,2,1)))

    model.add(Convolution2D(params["n_filters_3"],
                            params["n_kernel_3"][0],
                            params["n_kernel_3"][1],
                            init="uniform"))
    #model.add(BatchNormalization())
    model.add(Activation("relu"))
    logging.debug("Output Conv2D: %s" % str(model.output_shape))

    model.add(MaxPooling2D(pool_size=(params["n_pool_3"][0],
                                      params["n_pool_3"][1])))

    logging.debug("Output MaxPool2D: %s" % str(model.output_shape))
    model.add(Dropout(params["dropout_factor"]))

    if params["n_filters_4"] > 0:
        model.add(Convolution2D(params["n_filters_4"],
                                params["n_kernel_4"][0],
                                params["n_kernel_4"][1],
                                init="uniform"))
        model.add(Activation("relu"))
        #model.add(BatchNormalization())
        logging.debug("Output Conv2D: %s" % str(model.output_shape))

        model.add(MaxPooling2D(pool_size=(params["n_pool_4"][0],
                                          params["n_pool_4"][1])))

        logging.debug("Output MaxPool2D: %s" % str(model.output_shape))

        model.add(Dropout(params["dropout_factor"]))

    model.add(Flatten())
    logging.debug("Output Flatten: %s" % str(model.output_shape))

    model.add(Dropout(params["dropout_factor"]))

    if params["n_dense"] > 0:
        model.add(Dense(output_dim=params["n_dense"], init="uniform"))
        model.add(Activation("relu"))
        model.add(Dropout(params["dropout_factor"]))
        logging.debug("Output Dense: %s" % str(model.output_shape))

        model.add(Dense(output_dim=params["n_dense"], init="uniform"))
        model.add(Activation("relu"))
        model.add(Dropout(params["dropout_factor"]))
        logging.debug("Output Dense: %s" % str(model.output_shape))

    model.add(Dense(output_dim=params["n_out"], init="uniform"))
    model.add(Activation(params["final_activation"]))
    logging.debug("Output CNN: %s" % str(model.output_shape))

    if params['final_activation'] == 'linear':
        model.add(Lambda(lambda x :K.l2_normalize(x, axis=1)))

    return model

params_6 = {
    # dataset params
    'dataset' : {
        'fact' : 'nmf',
        'dim' : 200,
        'dataset' : 'W2',
        'window' : 15,
        'nsamples' : 'all',
        'npatches' : 3,
        'meta-suffix' : ''
    },

    # training params
    'training' : {
        'decay' : 1e-6,
        'learning_rate' : 0.1,
        'momentum' : 0.95,
        'n_epochs' : 100,
        'n_minibatch' : 32,
        'nesterov' : True,
        'validation' : 0.1,
        'test' : 0.1,
        'loss_func' : 'cosine',
        'optimizer' : 'sgd'
    },
    # cnn params
    'cnn' : {
        'dropout_factor' : 0.5,
        'n_dense' : 2048,
        'n_dense_2' : 2048,
        'n_filters_1' : 1024,
        'n_filters_2' : 1024,
        'n_filters_3' : 2048,
        'n_filters_4' : 2048,
        'n_kernel_1' : (4, 96),
        'n_kernel_2' : (4, 1),
        'n_kernel_3' : (4, 1),
        'n_kernel_4' : (1, 1),
        'n_out' : '',
        'n_pool_1' : (4, 1),
        'n_pool_2' : (4, 1),
        'n_pool_3' : (1, 1),
        'n_pool_4' : (1, 1),
        'n_frames' : '',
        'n_mel' : 96,
        'architecture' : 2,
        'n_metafeatures' : 7927#5393
    },
    'predicting' : {
        'trim_coeff' : 0.15
    },
    'evaluating' : {
        'get_map' : False,
        'get_p' : True,
        'get_knn' : False
    }
}

def get_model_6(params):
    inputs = Input(shape=(1, params["n_frames"],
                                         params["n_mel"]))

    conv1 = Convolution2D(params["n_filters_1"], params["n_kernel_1"][0],
                            params["n_kernel_1"][1],
                            border_mode='valid',
                            activation='relu',
                            input_shape=(1, params["n_frames"],
                                         params["n_mel"]),
                            init="uniform")
    x = conv1(inputs)
    #logging.debug("Input CNN: %s" % str(inputs.output_shape))
    logging.debug("Output Conv2D: %s" % str(conv1.output_shape))

    pool1 = MaxPooling2D(pool_size=(params["n_pool_1"][0],
                                      params["n_pool_1"][1]))
    x = pool1(x)
    logging.debug("Output MaxPool2D: %s" % str(pool1.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    conv2 = Convolution2D(params["n_filters_2"], params["n_kernel_2"][0],
                            params["n_kernel_2"][1],
                            border_mode='valid',
                            activation='relu',
                            init="uniform")
    x = conv2(x)
    logging.debug("Output Conv2D: %s" % str(conv2.output_shape))

    pool2 = MaxPooling2D(pool_size=(params["n_pool_2"][0],
                                      params["n_pool_2"][1]))
    x = pool2(x)
    logging.debug("Output MaxPool2D: %s" % str(pool2.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    #model.add(Permute((3,2,1)))

    conv3 = Convolution2D(params["n_filters_3"],
                            params["n_kernel_3"][0],
                            params["n_kernel_3"][1],
                            activation='relu',
                            init="uniform")
    x = conv3(x)
    logging.debug("Output Conv2D: %s" % str(conv3.output_shape))

    pool3 = MaxPooling2D(pool_size=(params["n_pool_3"][0],
                                      params["n_pool_3"][1]))
    x = pool3(x)
    logging.debug("Output MaxPool2D: %s" % str(pool3.output_shape))
    x = Dropout(params["dropout_factor"])(x)

    conv4 = Convolution2D(params["n_filters_4"],
                            params["n_kernel_4"][0],
                            params["n_kernel_4"][1],
                            activation='relu',
                            init="uniform")
    x = conv4(x)
    logging.debug("Output Conv2D: %s" % str(conv4.output_shape))

    pool4 = MaxPooling2D(pool_size=(params["n_pool_4"][0],
                                      params["n_pool_4"][1]))
    x = pool4(x)
    logging.debug("Output MaxPool2D: %s" % str(pool4.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    flat = Flatten()
    x = flat(x)
    logging.debug("Output Flatten: %s" % str(flat.output_shape))

    #dense1 = Dense(output_dim=params["n_dense"], init="uniform", activation='linear')
    #x = dense1(x)
    #logging.debug("Output CNN: %s" % str(dense1.output_shape))

    # metadata
    inputs2 = Input(shape=(params["n_metafeatures"],))
    dense2 = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
    x2 = dense2(inputs2)
    logging.debug("Output CNN: %s" % str(dense2.output_shape))

    x2 = Dropout(params["dropout_factor"])(x2)

    # merge
    xout = merge([x, x2], mode='concat', concat_axis=1)

    dense3 = Dense(output_dim=params["n_out"], init="uniform", activation='linear')
    xout = dense3(xout)
    logging.debug("Output CNN: %s" % str(dense3.output_shape))

    lambda1 = Lambda(lambda x :K.l2_normalize(x, axis=1))
    xout = lambda1(xout)

    model = Model(input=[inputs,inputs2], output=xout)

    return model


def get_model_7(params):
    inputs = Input(shape=(1, params["n_frames"],
                                         params["n_mel"]), name='input')

    conv1 = Convolution2D(params["n_filters_1"], params["n_kernel_1"][0],
                            params["n_kernel_1"][1],
                            border_mode='valid',
                            activation='relu',
                            input_shape=(1, params["n_frames"],
                                         params["n_mel"]),
                            init="uniform")
    x = conv1(inputs)
    #logging.debug("Input CNN: %s" % str(inputs.output_shape))
    logging.debug("Output Conv2D: %s" % str(conv1.output_shape))

    pool1 = MaxPooling2D(pool_size=(params["n_pool_1"][0],
                                      params["n_pool_1"][1]))
    x = pool1(x)
    logging.debug("Output MaxPool2D: %s" % str(pool1.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    conv2 = Convolution2D(params["n_filters_2"], params["n_kernel_2"][0],
                            params["n_kernel_2"][1],
                            border_mode='valid',
                            activation='relu',
                            init="uniform")
    x = conv2(x)
    logging.debug("Output Conv2D: %s" % str(conv2.output_shape))

    pool2 = MaxPooling2D(pool_size=(params["n_pool_2"][0],
                                      params["n_pool_2"][1]))
    x = pool2(x)
    logging.debug("Output MaxPool2D: %s" % str(pool2.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    #model.add(Permute((3,2,1)))

    conv3 = Convolution2D(params["n_filters_3"],
                            params["n_kernel_3"][0],
                            params["n_kernel_3"][1],
                            activation='relu',
                            init="uniform")
    x = conv3(x)
    logging.debug("Output Conv2D: %s" % str(conv3.output_shape))

    pool3 = MaxPooling2D(pool_size=(params["n_pool_3"][0],
                                      params["n_pool_3"][1]))
    x = pool3(x)
    logging.debug("Output MaxPool2D: %s" % str(pool3.output_shape))
    x = Dropout(params["dropout_factor"])(x)

    conv4 = Convolution2D(params["n_filters_4"],
                            params["n_kernel_4"][0],
                            params["n_kernel_4"][1],
                            activation='relu',
                            init="uniform")
    x = conv4(x)
    logging.debug("Output Conv2D: %s" % str(conv4.output_shape))

    pool4 = MaxPooling2D(pool_size=(params["n_pool_4"][0],
                                      params["n_pool_4"][1]))
    x = pool4(x)
    logging.debug("Output MaxPool2D: %s" % str(pool4.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    flat = Flatten(name='flat')
    xflat = flat(x)
    logging.debug("Output Flatten: %s" % str(flat.output_shape))

    #dense1 = Dense(output_dim=params["n_dense"], init="uniform", activation='linear')
    #x = dense1(x)
    #logging.debug("Output CNN: %s" % str(dense1.output_shape))

    dense3 = Dense(output_dim=params["n_out"], init="uniform", activation='linear')
    xout = dense3(xflat)
    logging.debug("Output CNN: %s" % str(dense3.output_shape))

    lambda1 = Lambda(lambda x :K.l2_normalize(x, axis=1))
    xout = lambda1(xout)

    model = Model(input=inputs, output=xout)

    return model


# METADATA buena !!!
def get_model_8(params):

    # metadata
    inputs2 = Input(shape=(params["n_metafeatures"],))
    x2 = Dropout(params["dropout_factor"])(inputs2)

    dense2 = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
    x2 = dense2(x2)
    logging.debug("Output CNN: %s" % str(dense2.output_shape))

    x2 = Dropout(params["dropout_factor"])(x2)

    dense3 = Dense(output_dim=params["n_dense_2"], init="uniform", activation='relu')
    x2 = dense3(x2)
    logging.debug("Output CNN: %s" % str(dense3.output_shape))

    x2 = Dropout(params["dropout_factor"])(x2)

    dense4 = Dense(output_dim=params["n_out"], init="uniform", activation='linear')
    xout = dense4(x2)
    logging.debug("Output CNN: %s" % str(dense4.output_shape))

    lambda1 = Lambda(lambda x :K.l2_normalize(x, axis=1))
    xoutl2 = lambda1(xout)

    model = Model(input=inputs2, output=xoutl2)

    return model

# Metadata
def get_model_81(params):

    # metadata
    inputs2 = Input(shape=(params["n_metafeatures"],))
    x2 = Dropout(params["dropout_factor"])(inputs2)

    """
    dense2 = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
    x2 = dense2(x2)
    logging.debug("Output CNN: %s" % str(dense2.output_shape))

    x2 = Dropout(params["dropout_factor"])(x2)

    dense3 = Dense(output_dim=params["n_dense_2"], init="uniform", activation='relu')
    x2 = dense3(x2)
    logging.debug("Output CNN: %s" % str(dense3.output_shape))

    x2 = Dropout(params["dropout_factor"])(x2)
    """

    dense4 = Dense(output_dim=params["n_out"], init="uniform", activation=params['final_activation'])
    xout = dense4(x2)
    logging.debug("Output CNN: %s" % str(dense4.output_shape))

    if params['final_activation'] == 'linear':
        reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
        xout = reg(xout)

    model = Model(input=inputs2, output=xout)

    return model


# Metadata 1 inputs, al estilo Metadata 2 inputs
def get_model_811(params):

    # metadata
    inputs = Input(shape=(params["n_metafeatures"],))

    norm = BatchNormalization()
    x = norm(inputs)

    x = Dropout(params["dropout_factor"])(x)

    dense = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
    x = dense(x)
    logging.debug("Output CNN: %s" % str(dense.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    dense4 = Dense(output_dim=params["n_out"], init="uniform", activation=params['final_activation'])
    xout = dense4(x)
    logging.debug("Output CNN: %s" % str(dense4.output_shape))

    if params['final_activation'] == 'linear':
        reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
        xout = reg(xout)

    model = Model(input=inputs, output=xout)

    return model


# Metadata 2 inputs, necesita meta-suffix2
def get_model_812(params):

    # metadata
    inputs = Input(shape=(params["n_metafeatures"],))

    norm = BatchNormalization()
    x = norm(inputs)

    x = Dropout(params["dropout_factor"])(x)

    dense = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
    x = dense(x)
    logging.debug("Output CNN: %s" % str(dense.output_shape))

    x = Dropout(params["dropout_factor"])(x)


    inputs2 = Input(shape=(params["n_metafeatures2"],))

    norm2 = BatchNormalization()
    x2 = norm2(inputs2)

    x2 = Dropout(params["dropout_factor"])(x2)

    dense2 = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
    x2 = dense2(x2)
    logging.debug("Output CNN: %s" % str(dense2.output_shape))

    x2 = Dropout(params["dropout_factor"])(x2)

    # merge
    xout = merge([x, x2], mode='concat', concat_axis=1)

    dense4 = Dense(output_dim=params["n_out"], init="uniform", activation=params['final_activation'])
    xout = dense4(xout)
    logging.debug("Output CNN: %s" % str(dense4.output_shape))

    if params['final_activation'] == 'linear':
        reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
        xout = reg(xout)

    model = Model(input=[inputs,inputs2], output=xout)

    return model


params_82 = {
    # dataset params
    'dataset' : {
        'fact' : 'als',
        'dim' : 200,
        'dataset' : 'W2',
        'window' : 15,
        'nsamples' : 'all',
        'npatches' : 1,
        'meta-suffix' : ''
    },

    # training params
    'training' : {
        'decay' : 1e-6,
        'learning_rate' : 0.1,
        'momentum' : 0.95,
        'n_epochs' : 100,
        'n_minibatch' : 32,
        'nesterov' : True,
        'validation' : 0.1,
        'test' : 0.1,
        'loss_func' : 'cosine',
        'optimizer' : 'sgd'
    },
    # cnn params
    'cnn' : {
        'dropout_factor' : 0.5,
        'sequence_length' : 500,
        'embedding_dim' : 300,
        'filter_sizes' : (2, 3, 4),
        'num_filters' : 150,
        'dropout_prob' : (0.6, 0.7),
        'hidden_dims' : 2048,
        'batch_size' : 32,
        'num_epochs' : 100,
        'val_split' : 0.1,
        'model_variation' : 'CNN-rnd',
        'n_out' : 200,
        'n_frames' : '',
        'n_mel' : 96,
        'architecture' : 82,
        'n_metafeatures' : 7927,#5393
        'final_activation' : 'linear'
    },
    'predicting' : {
        'trim_coeff' : 0.15
    },
    'evaluating' : {
        'get_map' : False,
        'get_p' : True,
        'get_knn' : False
    }
}

def get_model_82(params):
    embedding_weights = pickle.load(open("../data/datasets/train_data/embedding_weights_w2v-google_MSD-AG.pk","rb"))
    graph_in = Input(shape=(params['sequence_length'], params['embedding_dim']))
    convs = []
    for fsz in params['filter_sizes']:
        conv = Convolution1D(nb_filter=params['num_filters'],
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)
        x = conv(graph_in)
        logging.debug("Filter size: %s" % fsz)
        logging.debug("Output CNN: %s" % str(conv.output_shape))

        #conv2 = Convolution1D(nb_filter=params['num_filters'],
        #                     filter_length=4,
        #                     border_mode='valid',
        #                     activation='relu',
        #                     subsample_length=4)
        #x = conv2(x)
        #logging.debug("Output Conv2: %s" % str(conv2.output_shape))

        pool = GlobalMaxPooling1D()
        x = pool(x)
        logging.debug("Output Pooling: %s" % str(pool.output_shape))
        #flatten = Flatten()
        #x = flatten(x)
        #logging.debug("Flatten: %s" % str(flatten.output_shape))
        convs.append(x)

    if len(params['filter_sizes'])>1:
        merge = Merge(mode='concat')
        out = merge(convs)
        logging.debug("Merge: %s" % str(merge.output_shape))
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()
    if not params['model_variation']=='CNN-static':
        model.add(Embedding(len(embedding_weights[0]), params['embedding_dim'], input_length=params['sequence_length'],
                            weights=embedding_weights))
    model.add(Dropout(params['dropout_prob'][0], input_shape=(params['sequence_length'], params['embedding_dim'])))
    model.add(graph)
    model.add(Dense(params['hidden_dims']))
    model.add(Dropout(params['dropout_prob'][1]))
    model.add(Activation('relu'))

    model.add(Dense(output_dim=params["n_out"], init="uniform"))
    model.add(Activation(params['final_activation']))
    logging.debug("Output CNN: %s" % str(model.output_shape))

    if params['final_activation'] == 'linear':
        model.add(Lambda(lambda x :K.l2_normalize(x, axis=1)))

    return model

# Dileman arch
params_9 = {
    # dataset params
    'dataset' : {
        'fact' : 'nmf',
        'dim' : 200,
        'dataset' : 'W2',
        'window' : 15,
        'nsamples' : 'all',
        'npatches' : 3
    },

    # training params
    'training' : {
        'decay' : 1e-6,
        'learning_rate' : 0.1,
        'momentum' : 0.95,
        'n_epochs' : 200,
        'n_minibatch' : 32,
        'nesterov' : True,
        'validation' : 0.1,
        'test' : 0.1,
        'loss_func' : 'mean_squared_error',
        'optimizer' : 'adam'
    },
    # cnn params
    'cnn' : {
        'dropout_factor' : 0.5,
        'n_dense' : 2048,
        'n_filters_1' : 256,
        'n_filters_2' : 256,
        'n_filters_3' : 512,
        'n_kernel_1' : (4, 96),
        'n_kernel_2' : (4, 1),
        'n_kernel_3' : (4, 1),
        'n_out' : '',
        'n_pool_1' : (4, 1),
        'n_pool_2' : (4, 1),
        'n_pool_3' : (4, 1),
        'n_frames' : '',
        'n_mel' : 96,
        'architecture' : 9,
    },
    'predicting' : {
        'trim_coeff' : 0.15
    },
    'evaluating' : {
        'get_map' : False,
        'get_p' : True,
        'get_knn' : False
    }
}

def l2pooling(input,n_filters_per_unit):
    output = input.reshape((input.shape[0], input.shape[1] / n_filters_per_unit, n_filters_per_unit, input.shape[2]))
    padding = 0.000001
    output = T.sqrt(T.mean(output**2, 2) + padding)
    return output

def get_model_9(params):
    inputs = Input(shape=(1, params["n_frames"],
                                         params["n_mel"]))

    conv1 = Convolution2D(params["n_filters_1"], params["n_kernel_1"][0],
                            params["n_kernel_1"][1],
                            border_mode='valid',
                            activation='relu',
                            input_shape=(1, params["n_frames"],
                                         params["n_mel"]),
                            init="uniform")
    x = conv1(inputs)
    #logging.debug("Input CNN: %s" % str(inputs.output_shape))
    logging.debug("Output Conv2D: %s" % str(conv1.output_shape))

    pool1 = MaxPooling2D(pool_size=(params["n_pool_1"][0],
                                      params["n_pool_1"][1]))
    x = pool1(x)
    logging.debug("Output MaxPool2D: %s" % str(pool1.output_shape))

    #x = Dropout(params["dropout_factor"])(x)

    conv2 = Convolution2D(params["n_filters_2"], params["n_kernel_2"][0],
                            params["n_kernel_2"][1],
                            border_mode='valid',
                            activation='relu',
                            init="uniform")
    x = conv2(x)
    logging.debug("Output Conv2D: %s" % str(conv2.output_shape))

    pool2 = MaxPooling2D(pool_size=(params["n_pool_2"][0],
                                      params["n_pool_2"][1]))
    x = pool2(x)
    logging.debug("Output MaxPool2D: %s" % str(pool2.output_shape))

    #x = Dropout(params["dropout_factor"])(x)

    #model.add(Permute((3,2,1)))

    conv3 = Convolution2D(params["n_filters_3"],
                            params["n_kernel_3"][0],
                            params["n_kernel_3"][1],
                            activation='relu',
                            init="uniform")
    x = conv3(x)
    logging.debug("Output Conv2D: %s" % str(conv3.output_shape))


    pool3 = MaxPooling2D(pool_size=(params["n_pool_3"][0],
                                      params["n_pool_3"][1]))
    x1 = pool3(x)
    logging.debug("Output MaxPool2D: %s" % str(pool3.output_shape))
    flat = Flatten()
    x1 = flat(x1)
    logging.debug("Output Flatten: %s" % str(flat.output_shape))

    pool4 = AveragePooling2D(pool_size=(params["n_pool_3"][0],
                                      params["n_pool_3"][1]))
    x2 = pool4(x)
    logging.debug("Output AvgPool2D: %s" % str(pool4.output_shape))
    flat = Flatten()
    x2 = flat(x2)
    logging.debug("Output Flatten: %s" % str(flat.output_shape))

    flat = Flatten()
    x = flat(x)
    logging.debug("Output Flatten: %s" % str(flat.output_shape))
    lambda1 = Lambda(lambda x :l2pooling(x, 4))
    x3 = lambda1(x)
    logging.debug("Output Lambda: %s" % str(lambda1.output_shape))

    # merge
    x = merge([x1, x2, x3], mode='concat', concat_axis=1)

    #flat = Flatten()
    #x = flat(x)
    #logging.debug("Output Flatten: %s" % str(flat.output_shape))

    dense1 = Dense(output_dim=params["n_dense"], init="uniform", activation='linear')
    x = dense1(x)
    logging.debug("Output CNN: %s" % str(dense1.output_shape))
    x = Dropout(params["dropout_factor"])(x)

    dense2 = Dense(output_dim=params["n_dense"], init="uniform", activation='linear')
    x = dense2(x)
    logging.debug("Output CNN: %s" % str(dense2.output_shape))
    x = Dropout(params["dropout_factor"])(x)

    dense3 = Dense(output_dim=params["n_out"], init="uniform", activation='linear')
    xout = dense3(x)
    logging.debug("Output CNN: %s" % str(dense3.output_shape))

    #lambda1 = Lambda(lambda x :K.l2_normalize(x, axis=1))
    #xout = lambda1(xout)

    model = Model(input=inputs, output=xout)

    return model

def get_model_10(params):

    #max_features = 50001
    #maxlen = 500
    #vocab_dim = 100 # dimensionality of your word vectors

    embedding_weights = pickle.load(open("../data/datasets/train_data/embedding_weights_w2v_MSD-AG.pk","rb"))
    model = Sequential()
    model.add(Embedding(len(embedding_weights[0]), params['embedding_dim'], input_length=params['sequence_length'],
                            weights=embedding_weights))
    #model.add(Embedding(input_dim=max_features, output_dim=vocab_dim, input_length=maxlen, mask_zero=True, weights=[embedding_weights]))
    logging.debug("Input Embedding: %s" % str(model.input_shape))
    logging.debug("Output Embedding: %s" % str(model.output_shape))
    model.add(Bidirectional(LSTM(200)))
    logging.debug("Output LSTM: %s" % str(model.output_shape))
    model.add(Dropout(params['dropout_prob'][0], input_shape=(params['sequence_length'], params['embedding_dim'])))
    model.add(Dense(params['hidden_dims']))
    model.add(Dropout(params['dropout_prob'][1]))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=params["n_out"], init="uniform"))
    model.add(Activation(params['final_activation']))
    logging.debug("Output CNN: %s" % str(model.output_shape))

    if params['final_activation'] == 'linear':
        model.add(Lambda(lambda x :K.l2_normalize(x, axis=1)))

    return model

def get_model_11(params):
    # set parameters:
    max_features = 50000
    maxlen = 500
    embedding_dims = 100
    nb_filter = 1024
    filter_length = 3
    hidden_dims = 2048

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen,
                        dropout=0.2))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(output_dim=params["n_out"], activation='linear'))
    model.add(Lambda(lambda x :K.l2_normalize(x, axis=1)))

    return model

def get_model_12(params):
    # set parameters:
    max_features = 50000
    maxlen = 500
    embedding_dims = 100
    nb_filter = 250
    hidden_dims = 2048

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen,
                        dropout=0.2))
    logging.debug("Input Embedding: %s" % str(model.input_shape))
    logging.debug("Output Embedding: %s" % str(model.output_shape))

    model.add(Reshape((1, maxlen, embedding_dims)))
    logging.debug("Output Reshape: %s" % str(model.output_shape))
    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution2D(nb_filter,2,embedding_dims,
                            border_mode='valid',
                            activation='relu', init='uniform'))
    logging.debug("Output Convolution: %s" % str(model.output_shape))

    # we use max pooling:
    model.add(MaxPooling2D(pool_size=(4,1)))
    logging.debug("Output MaxPooling: %s" % str(model.output_shape))
    model.add(Flatten())
    logging.debug("Output Flatten: %s" % str(model.output_shape))

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    logging.debug("Output Dense: %s" % str(model.output_shape))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(output_dim=params["n_out"], activation='linear'))
    model.add(Lambda(lambda x :K.l2_normalize(x, axis=1)))

    return model


def main():
    pass

if __name__ == '__main__':
    main()
