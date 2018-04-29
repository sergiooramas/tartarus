from keras.layers import Dense, Dropout, Activation, Flatten, Permute, Lambda, Input, merge, BatchNormalization, Embedding, LSTM, Bidirectional, Reshape, GRU, Merge, ELU
from keras.layers import Convolution1D, GlobalMaxPooling1D, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, MaxPooling1D
from keras.regularizers import l2, l1
from keras import regularizers
from keras.models import Sequential, Model
import logging
from keras import backend as K
import keras
import theano.tensor as T
import numpy as np
import pickle
import common


params_1 = {
    # dataset params
    'dataset' : {
        'fact' : '',
        'dim' : 200,
        'dataset' : '',
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
        'n_dense_2' : 0,
        'n_filters_1' : 1024,
        'n_filters_2' : 1024,
        'n_filters_3' : 2048,
        'n_filters_4' : 2048,
        'n_filters_5' : 0,
        'n_kernel_1' : (4, 96),
        'n_kernel_2' : (4, 1),
        'n_kernel_3' : (4, 1),
        'n_kernel_4' : (1, 1),
        'n_kernel_5' : (1, 1),
        'n_out' : '',
        'n_pool_1' : (4, 1),
        'n_pool_2' : (4, 1),
        'n_pool_3' : (1, 1),
        'n_pool_4' : (1, 1),
        'n_pool_5' : (1, 1),
        'n_frames' : 322,
        'n_mel' : 96,
        'architecture' : 2,
        'batch_norm' : False,
        'dropout' : True
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

# AUDIO ARCH CNNs
def get_model_1(params):
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


    model.add(Convolution2D(params["n_filters_2"], params["n_kernel_2"][0],
                            params["n_kernel_2"][1],
                            border_mode='valid',
                            init="uniform"))
    #model.add(BatchNormalization())
    model.add(Activation("relu"))
    logging.debug("Output Conv2D: %s" % str(model.output_shape))

    model.add(MaxPooling2D(pool_size=(params["n_pool_2"][0],
                                      params["n_pool_2"][1])))
    logging.debug("Output MaxPool2D: %s" % str(model.output_shape))
    model.add(Dropout(params["dropout_factor"]))

    if params["n_filters_3"] > 0:
        model.add(Convolution2D(params["n_filters_3"],
                                params["n_kernel_3"][0],
                                params["n_kernel_3"][1],
                                border_mode='valid',
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
                                border_mode='valid',
                                init="uniform"))
        model.add(Activation("relu"))
        #model.add(BatchNormalization())
        logging.debug("Output Conv2D: %s" % str(model.output_shape))

        model.add(MaxPooling2D(pool_size=(params["n_pool_4"][0],
                                          params["n_pool_4"][1])))

        logging.debug("Output MaxPool2D: %s" % str(model.output_shape))

        model.add(Dropout(params["dropout_factor"]))

    if params["n_filters_5"] > 0:
        model.add(Convolution2D(params["n_filters_5"],
                                params["n_kernel_5"][0],
                                params["n_kernel_5"][1],
                                border_mode='valid',
                                init="uniform"))
        model.add(Activation("relu"))
        #model.add(BatchNormalization())
        logging.debug("Output Conv2D: %s" % str(model.output_shape))

        model.add(MaxPooling2D(pool_size=(params["n_pool_5"][0],
                                          params["n_pool_5"][1])))

        logging.debug("Output MaxPool2D: %s" % str(model.output_shape))

        model.add(Dropout(params["dropout_factor"]))

    model.add(Flatten())
    logging.debug("Output Flatten: %s" % str(model.output_shape))

    #model.add(Dropout(params["dropout_factor"]))

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

# Audio ARCH with graph api
def get_model_11(params):
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

# AUDIO multiple filters
def get_model_12(params):
    graph_in = Input(shape=(1, params["n_frames"],params["n_mel"]))
    convs = []
    params['filter_sizes'] = [(1,70),(5,70),(10,70),(1,35),(5,35),(10,35)]
    params['filter_widths'] = [64,32,32,64,32,32]
    for i,fsz in enumerate(params['filter_sizes']):
        conv = Convolution2D(params['filter_widths'][i],fsz[0],fsz[1],
                             border_mode='same',
                             activation='relu',
                             init='uniform')
        x = conv(graph_in)
        logging.debug("Filter size: %s,%s" % (fsz[0],fsz[1]))
        logging.debug("Output CNN: %s" % str(conv.output_shape))
        convs.append(x)

    if len(params['filter_sizes'])>1:
        merge1 = Merge(mode='concat',concat_axis=1)
        out = merge1(convs)
        logging.debug("Merge: %s" % str(merge1.output_shape))
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    model = Sequential()
    model.add(graph)

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
    if params["n_filters_3"] > 0:
        model.add(Convolution2D(params["n_filters_3"],
                                params["n_kernel_3"][0],
                                params["n_kernel_3"][1],
                                border_mode='valid',
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
                                border_mode='valid',
                                init="uniform"))
        model.add(Activation("relu"))
        #model.add(BatchNormalization())
        logging.debug("Output Conv2D: %s" % str(model.output_shape))

        model.add(MaxPooling2D(pool_size=(params["n_pool_4"][0],
                                          params["n_pool_4"][1])))

        logging.debug("Output MaxPool2D: %s" % str(model.output_shape))

        model.add(Dropout(params["dropout_factor"]))

    if params["n_filters_5"] > 0:
        model.add(Convolution2D(params["n_filters_5"],
                                params["n_kernel_5"][0],
                                params["n_kernel_5"][1],
                                border_mode='valid',
                                init="uniform"))
        model.add(Activation("relu"))
        #model.add(BatchNormalization())
        logging.debug("Output Conv2D: %s" % str(model.output_shape))

        model.add(MaxPooling2D(pool_size=(params["n_pool_5"][0],
                                          params["n_pool_5"][1])))

        logging.debug("Output MaxPool2D: %s" % str(model.output_shape))

        model.add(Dropout(params["dropout_factor"]))

    model.add(Flatten())
    logging.debug("Output Flatten: %s" % str(model.output_shape))

    #model.add(Dropout(params["dropout_factor"]))

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

# Multimodal ARCH text + audio
def get_model_2(params):
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

# METADATA ARCH
def get_model_3(params):

    # metadata
    inputs2 = Input(shape=(params["n_metafeatures"],))
    x2 = Dropout(params["dropout_factor"])(inputs2)

    if params["n_dense"] > 0:
        dense2 = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
        x2 = dense2(x2)
        logging.debug("Output CNN: %s" % str(dense2.output_shape))

        x2 = Dropout(params["dropout_factor"])(x2)

    if params["n_dense_2"] > 0:
        dense3 = Dense(output_dim=params["n_dense_2"], init="uniform", activation='relu')
        x2 = dense3(x2)
        logging.debug("Output CNN: %s" % str(dense3.output_shape))

        x2 = Dropout(params["dropout_factor"])(x2)

    dense4 = Dense(output_dim=params["n_out"], init="uniform", activation=params['final_activation'])
    xout = dense4(x2)
    logging.debug("Output CNN: %s" % str(dense4.output_shape))

    if params['final_activation'] == 'linear':
        reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
        xout = reg(xout)

    model = Model(input=inputs2, output=xout)

    return model


# Metadata 2 inputs, post-merge with dense layers
def get_model_31(params):

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

# Metadata 2 inputs, pre-merge and l2
def get_model_32(params):

    # metadata
    inputs = Input(shape=(params["n_metafeatures"],))
    reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
    x1 = reg(inputs)

    inputs2 = Input(shape=(params["n_metafeatures2"],))
    reg2 = Lambda(lambda x :K.l2_normalize(x, axis=1))
    x2 = reg2(inputs2)

    # merge
    x = merge([x1, x2], mode='concat', concat_axis=1)

    x = Dropout(params["dropout_factor"])(x)

    if params['n_dense'] > 0:
        dense2 = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
        x = dense2(x)
        logging.debug("Output CNN: %s" % str(dense2.output_shape))

    dense4 = Dense(output_dim=params["n_out"], init="uniform", activation=params['final_activation'])
    xout = dense4(x)
    logging.debug("Output CNN: %s" % str(dense4.output_shape))

    if params['final_activation'] == 'linear':
        reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
        xout = reg(xout)

    model = Model(input=[inputs,inputs2], output=xout)

    return model

# Metadata 3 inputs, pre-merge and l2
def get_model_33(params):

    # metadata
    inputs = Input(shape=(params["n_metafeatures"],))
    reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
    x1 = reg(inputs)

    inputs2 = Input(shape=(params["n_metafeatures2"],))
    reg2 = Lambda(lambda x :K.l2_normalize(x, axis=1))
    x2 = reg2(inputs2)

    inputs3 = Input(shape=(params["n_metafeatures3"],))
    reg3 = Lambda(lambda x :K.l2_normalize(x, axis=1))
    x3 = reg3(inputs3)

    # merge
    x = merge([x1, x2, x3], mode='concat', concat_axis=1)

    x = Dropout(params["dropout_factor"])(x)

    if params['n_dense'] > 0:
        dense2 = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
        x = dense2(x)
        logging.debug("Output CNN: %s" % str(dense2.output_shape))

    dense4 = Dense(output_dim=params["n_out"], init="uniform", activation=params['final_activation'])
    xout = dense4(x)
    logging.debug("Output CNN: %s" % str(dense4.output_shape))

    if params['final_activation'] == 'linear':
        reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
        xout = reg(xout)

    model = Model(input=[inputs,inputs2,inputs3], output=xout)

    return model


# Metadata 4 inputs, pre-merge and l2
def get_model_34(params):

    # metadata
    inputs = Input(shape=(params["n_metafeatures"],))
    reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
    x1 = reg(inputs)

    inputs2 = Input(shape=(params["n_metafeatures2"],))
    reg2 = Lambda(lambda x :K.l2_normalize(x, axis=1))
    x2 = reg2(inputs2)

    inputs3 = Input(shape=(params["n_metafeatures3"],))
    reg3 = Lambda(lambda x :K.l2_normalize(x, axis=1))
    x3 = reg3(inputs3)

    inputs4 = Input(shape=(params["n_metafeatures4"],))
    reg4 = Lambda(lambda x :K.l2_normalize(x, axis=1))
    x4 = reg4(inputs4)

    # merge
    x = merge([x1, x2, x3, x4], mode='concat', concat_axis=1)

    x = Dropout(params["dropout_factor"])(x)

    if params['n_dense'] > 0:
        dense2 = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
        x = dense2(x)
        logging.debug("Output CNN: %s" % str(dense2.output_shape))

    dense4 = Dense(output_dim=params["n_out"], init="uniform", activation=params['final_activation'])
    xout = dense4(x)
    logging.debug("Output CNN: %s" % str(dense4.output_shape))

    if params['final_activation'] == 'linear':
        reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
        xout = reg(xout)

    model = Model(input=[inputs,inputs2,inputs3,inputs4], output=xout)

    return model

params_w2v = {
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
        'filter_sizes' : (2, 3),
        'num_filters' : 150,
        'dropout_prob' : (0.5, 0.8),
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

# word2vec ARCH with CNNs
def get_model_4(params):
    embedding_weights = pickle.load(open(common.TRAINDATA_DIR+"/embedding_weights_w2v_%s.pk" % params['embeddings_suffix'],"rb"))
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

        pool = GlobalMaxPooling1D()
        x = pool(x)
        logging.debug("Output Pooling: %s" % str(pool.output_shape))
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
    model.add(Dense(params['n_dense']))
    model.add(Dropout(params['dropout_prob'][1]))
    model.add(Activation('relu'))

    model.add(Dense(output_dim=params["n_out"], init="uniform"))
    model.add(Activation(params['final_activation']))
    logging.debug("Output CNN: %s" % str(model.output_shape))

    if params['final_activation'] == 'linear':
        model.add(Lambda(lambda x :K.l2_normalize(x, axis=1)))

    return model

# word2vec ARCH with LSTM
def get_model_41(params):
    embedding_weights = pickle.load(open("../data/datasets/train_data/embedding_weights_w2v-google_MSD-AG.pk","rb"))
    # main sequential model
    model = Sequential()
    model.add(Embedding(len(embedding_weights[0]), params['embedding_dim'], input_length=params['sequence_length'],
                        weights=embedding_weights))
    #model.add(Dropout(params['dropout_prob'][0], input_shape=(params['sequence_length'], params['embedding_dim'])))
    model.add(LSTM(2048))
    #model.add(Dropout(params['dropout_prob'][1]))
    model.add(Dense(output_dim=params["n_out"], init="uniform"))
    model.add(Activation(params['final_activation']))
    logging.debug("Output CNN: %s" % str(model.output_shape))

    if params['final_activation'] == 'linear':
        model.add(Lambda(lambda x :K.l2_normalize(x, axis=1)))

    return model


# CRNN Arch for audio
def get_model_5(params):
    input_tensor=None
    include_top=True    

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)
    x = Permute((1, 3, 2))(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 2, 1))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3)(x)
    if include_top:
        x = Dense(params["n_out"], activation=params['final_activation'], name='output')(x)

    if params['final_activation'] == 'linear':
        reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
        x = reg(x)

    # Create model
    model = Model(melgram_input, x)
    return model

def main():
    pass

if __name__ == '__main__':
    main()
