import os
#os.chdir('F:/PhD/EMG_All/NinaProDB/Musa_NinaPro_Project')
from keras.layers import *
from keras import initializers, regularizers, constraints
from keras.models import Model
from tcn import TCN
from custom_layers import *

def TCCNet(input_shape, classes, residual_blocks=3, tcn_layers=2,
    filters=[[32, 32], [64, 64], [128, 128]], filters_size=3,
    n_dropout=0.5, n_l2=0.0005, classify_as='aot', masking=False):
    """
        Arguments:
            input_shape     : array-like, dimensions of data input (height, width, depth)
            classes         : integer, number of classification labels
            residual_blocks : integer, (see 'tcn.py')
            tcn_layers      : integer, (see 'tcn.py')
            filters         : array-like, (see 'tcn.py')
            filters_size    : integer or array-like, (see 'tcn.py')
            dropout         : float, amount of dropout
            l2              : float, amount of l_2 regularization
            classify_as     : string, one of {'aot', 'att'} corresponding to Average over Time and Attention
    """

    kernel_init = initializers.glorot_normal(seed=0)
    kernel_regl = regularizers.l2(n_l2)
    print('redidual_blocks')
    print(residual_blocks)

    model_tcn = TCN(input_shape, residual_blocks=residual_blocks, tcn_layers=tcn_layers,
                filters=filters, filters_size=filters_size,
                dropout_rate=n_dropout, weight_decay=n_l2, seed=0, masking=masking)
    print("model_tcn:.......", model_tcn.layers)
    x_input = model_tcn.layers[0].output
    x = model_tcn.layers[-1].output

    if classify_as == 'aot':
        x = MeanOverTime()(x)
        y = Dense(classes, activation='softmax', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init)(x)
    elif classify_as == 'att':
        x = AttentionWithContext(bias=False)(x)
        y = Dense(classes, activation='softmax', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init)(x)

    model = Model(x_input, y)

    return model


'''
from keras.layers import *
from keras import initializers, regularizers, constraints
from keras.models import Model
from tcn import TCN
from custom_layers import *

def TCCNet(input_shape, classes, residual_blocks=3, tcn_layers=2,
    filters=[[32, 32], [32, 32], [64, 64], [128, 128]], filters_size=2,
    n_dropout=0.5, n_l2=0.0005, classify_as='aot', masking=False):
    """
        Arguments:
            input_shape     : array-like, dimensions of data input (height, width, depth)
            classes         : integer, number of classification labels
            residual_blocks : integer, (see 'tcn.py')
            tcn_layers      : integer, (see 'tcn.py')
            filters         : array-like, (see 'tcn.py')
            filters_size    : integer or array-like, (see 'tcn.py')
            dropout         : float, amount of dropout
            l2              : float, amount of l_2 regularization
            classify_as     : string, one of {'aot', 'att'} corresponding to Average over Time and Attention
    """

    kernel_init = initializers.glorot_normal(seed=0)  # a normalizer  alies tf.initializers.GlorotNormal, tf.initializers.glorot_normal, tf.keras.initializers.glorot_normal
    print('kernel_init', type(kernel_init))
    kernel_regl = regularizers.l2(n_l2)
    print('kernel_init', type(kernel_regl))
    model_tcn = TCN(input_shape, residual_blocks=residual_blocks, tcn_layers=tcn_layers,
                filters=filters, filters_size=filters_size,
                dropout_rate=n_dropout, weight_decay=n_l2, seed=0, masking=masking)
    
    x_input = model_tcn.layers[0].output
    x = model_tcn.layers[-1].output
    print('Shape of x after TCN', x.shape)

    if classify_as == 'aot':
        x = MeanOverTime()(x)
        y = Dense(classes, activation='softmax', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init)(x)
    elif classify_as == 'att':
        x = AttentionWithContext(bias=False)(x)
        y = Dense(classes, activation='softmax', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init)(x)

    model = Model(x_input, y)

    return model

'''

def getNetwork(network):
#def getNetwork():

    #if 'TCNet' in str(network) or 'TCCNet' in str(network):
    model = MYTCC
           
    return model


#a=TCN(input_shape=(None, 10))