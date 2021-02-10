import keras
from keras import regularizers
from keras.layers import Input, Add, Dense, Activation,Dropout, BatchNormalization, Flatten, Conv3D, MaxPooling3D,AveragePooling3D
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.initializers import glorot_uniform,he_uniform
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.utils import multi_gpu_model
import keras.backend as K
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from math import ceil
import utils

import random as python_random

np.random.seed(123)
python_random.seed(123)

def resnet_with_batchnorm(input_shape):
    """
    Implementation of the residual block from the nature paper https://www.nature.com/articles/s41467-019-13163-9.pdf
    # The only difference is the input shape and Batchnormalization layer is used instead of batchrenormalization layers
    Arguments:
    X -- input tensor of shape (m, H, W, D, C)
    Returns:
    model
    """

    X_input = Input(input_shape,name='Allimages')

    ##### MAIN PATH #####

    # First component of main path
    regAmount = 0.00005
    initType='he_uniform'
    paddingType = 'same'
    X = Conv3D(filters = 8, kernel_size = (3, 3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),
                                                      kernel_initializer=initType)(X_input)
    X = BatchNormalization(axis = -1)(X)#, trainable=True)(X)
    #X = BatchRenormalization(axis = -1)(X)
    #X.trainable=True
    X = Activation('elu')(X)
    X = Conv3D(filters = 8, kernel_size = (3, 3, 3), strides=(1,1,1), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),
                                                      kernel_initializer=initType)(X)
    X = BatchNormalization(axis = -1)(X)
    #X = Activation('elu')(X)

    X_shortcut = Conv3D(8, kernel_size = (1,1,1), strides=(1,1,1), padding=paddingType,kernel_initializer=initType)(X_input)
    X = Add()([X_shortcut,X])
    outputs = Activation('elu')(X)

    pooling = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2), padding=paddingType)(outputs)


   #######################################################################################################################

    inputs = pooling
    features = 16
    hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(inputs)
    hidden = BatchNormalization(axis=-1)(hidden)
    hidden = Activation('elu')(hidden)

    hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(hidden)
    hidden = BatchNormalization(axis=-1)(hidden)

    shortcut = Conv3D(features, (1,1,1), strides=(1,1,1), padding=paddingType,kernel_initializer=initType)(inputs)
    hidden = Add()([shortcut,hidden])
    outputs = Activation('elu')(hidden)

    pooling = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2), padding=paddingType)(outputs)

   ########################################################################################################################
    inputs = pooling
    features = 32
    hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(inputs)
    hidden = BatchNormalization(axis=-1)(hidden)
    hidden = Activation('elu')(hidden)

    hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(hidden)
    hidden = BatchNormalization(axis=-1)(hidden)

    shortcut = Conv3D(features, (1,1,1), strides=(1,1,1), padding=paddingType,kernel_initializer=initType)(inputs)
    hidden = Add()([shortcut,hidden])
    outputs = Activation('elu')(hidden)

    pooling = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2), padding=paddingType)(outputs)

   ##########################################################################################################################

    inputs = pooling
    features = 64
    hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(inputs)
    hidden = BatchNormalization(axis=-1)(hidden)
    hidden = Activation('elu')(hidden)

    hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(hidden)
    hidden = BatchNormalization(axis=-1)(hidden)

    shortcut = Conv3D(features, (1,1,1), strides=(1,1,1), padding=paddingType,kernel_initializer=initType)(inputs)
    hidden = Add()([shortcut,hidden])
    outputs = Activation('elu')(hidden)

    pooling = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2), padding=paddingType)(outputs)

   ######################################################################################################################
    inputs = pooling
    features = 128
    hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(inputs)
    hidden = BatchNormalization(axis=-1)(hidden)
    hidden = Activation('elu')(hidden)

    hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(hidden)
    hidden = BatchNormalization(axis=-1)(hidden)

    shortcut = Conv3D(features, (1,1,1), strides=(1,1,1), padding=paddingType,kernel_initializer=initType)(inputs)
    hidden = Add()([shortcut,hidden])
    outputs = Activation('elu')(hidden)

    pooling = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2), padding=paddingType)(outputs)


    hidden = Flatten()(pooling)
    hidden = Dense(128,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType,name='FullyConnectedLayer')(hidden)
    hidden = Activation('elu')(hidden)
    hidden = Dropout(0.8)(hidden)#, training=True) # dropout layer was not used during testing, so had to add this,check https://github.com/keras-team/keras/issues/9412

    prediction = Dense(1,kernel_regularizer=regularizers.l2(regAmount), name='PainPrediction')(hidden)

    model = Model(inputs = X_input, outputs = prediction)

    return model

# def step_decay(epoch):
#     initial_rate = 0.01
#     factor = int(epoch / 1000)
#     return initial_rate * (0.3 ** factor)

# def start_training(X_train, y_train,X_val,y_val,model, lr, decayRate, generator,patience=100, batchSize=4, optimizer_to_use='Adam',nEpochs=1000,model_path=None,use_dynamic_LR=True):


#     if optimizer_to_use =='Adam':
#         opt = Adam(lr, beta_1=0.9, beta_2=0.999,decay=decayRate)
#     else:
#         opt = SGD(lr,decay=decayRate)
# #     adam = Adam(lr=lr, decay=decayRate)
#     model.compile(loss='mean_absolute_error',optimizer=opt)

#     if model_path==None:
#         model_path=os.getcwd()

#     early = EarlyStopping(monitor='val_loss', patience=patience,mode='min',verbose=1)
#     mc = ModelCheckpoint(filepath=os.path.join(model_path,'model.h5'),verbose=1, monitor='val_loss', save_best_only=True)

#     if use_dynamic_LR:
#    # learning schedule callback
#         lrate = LearningRateScheduler(step_decay)
#         cb = [early, mc, lrate]
#     else:
#         lrate = lr
#         cb = [early, mc]
#     print("##########model requires %s GB##########"%utils.get_model_memory_usage(batchSize, model))
#     if generator:
#         print("Using generator...")
#         hist = model.fit_generator(dataGenerator(X_train, y_train, batch_size = batchSize, meanImg=None, scaling=False,dim=X_train.shape[1:], shuffle=True, augment=False, maxAngle=40, maxShift=10),
#                       validation_data=dataGenerator(X_val, y_val, batch_size = batchSize, meanImg=None, scaling=False,dim=X_train.shape[1:], shuffle=True, augment=False, maxAngle=40, maxShift=10),
#                        epochs=nEpochs,verbose=1,max_queue_size=32,workers=4,use_multiprocessing=False,steps_per_epoch=ceil(X_train.shape[0]/batchSize),callbacks=cb)
#     else:
#         hist = model.fit(X_train,y_train, epochs=nEpochs, batch_size=batchSize,verbose = 1,
#                      # max_queue_size=32,workers=4,use_multiprocessing=False,
#                       validation_data=(X_val,y_val),shuffle=True,callbacks=cb)
    
#     return hist

