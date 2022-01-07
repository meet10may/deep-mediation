from PIL import Image
from sklearn import datasets
import numpy as np
import pandas as pd
# For repeatable results from keras
from numpy.random import seed
seed(1)
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV, cross_val_predict,cross_val_score
from sklearn.metrics import f1_score,confusion_matrix,classification_report,roc_curve, auc, mean_squared_error,make_scorer,mean_absolute_error,r2_score

from sklearn.svm import SVR
import matplotlib.pyplot as plt
import statsmodels

from scipy.stats import zscore,norm
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

from sklearn.datasets import fetch_openml
mnist_dataset = fetch_openml('mnist_784')
import os
import keras
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, Add, Dense, Activation,Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D,Conv3D, MaxPooling3D
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping


def create_train_data(X,y,testing_dataset_size=0.3):
    training_dataset_size = 1-testing_dataset_size
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=testing_dataset_size,shuffle=True,random_state=42)
    print("Dataset is randomly split into %s%% test dataset and %s%% train dataset"%(testing_dataset_size*100,training_dataset_size*100))
    print("Total number of subjects in training dataset: %s"%len(X_train))
    print("Total number of subjects in testing dataset: %s"%len(X_test))
    return X_train, X_test, y_train, y_test

def tune_params(est,parameters,train_df,train_y,cross_val=5,njobs=-1,verbose=2):
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    est_grid = GridSearchCV(est, parameters,cv=cross_val,n_jobs=njobs,verbose=verbose,scoring=scorer)
    est_grid.fit(train_df, train_y)
    return est_grid.best_estimator_

def plot_regression_results(ax, y_true, y_pred, title, scores):
    """Scatter plot of the predicted vs true targets."""

    ax.plot([y_true.min()-1, y_true.max()+1],[y_true.min()-1, y_true.max()+1],'--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', y_true.max()+1))
    ax.spines['bottom'].set_position(('outward', y_true.max()+1))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Actual Value')
    ax.set_ylabel('Predicted Value')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = title
    ax.set_title(title)

def load_pickle(filename):
    model = pickle.load(open(filename, 'rb'))
    return model

def save_pickle(model,filename):
    pickle.dump(model,open(filename, 'wb'))


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def extract_image(num,mnist_dataset,resize,new_shape,algo='svr'):
    """
    Used for data simulation.Returns the image
    num: string. The string of number and a digit is replaced with an actual image.
    """

    target = mnist_dataset.target
    mnist_image = mnist_dataset.data
    img_shape = (int(np.sqrt(mnist_image.shape[-1])) , int(np.sqrt(mnist_image.shape[-1])))
    image= []
    for j in range(len(num)):
        raw_img = mnist_image[np.argwhere(target == num[j])[0]][0]
        raw_img = raw_img.reshape(img_shape)
        new_img = Image.fromarray(raw_img)
        img = new_img.resize(size=(new_shape[0],new_shape[1]))
        img = np.asarray(img)
        if j==0:
            output = img
        else:
            output = concat_images(output, img)
        if algo == 'svr':
            output_ = output.flatten()
#         if not cnn:
#             output_ = output.flatten()
    if algo == 'svr':
        output = output_
    image.append(output)
    return np.concatenate(image)

def visualize_dataset(M,z,idx,shape):
    """
    M: multi-dimensional mediator image
    z:
    idx: the index of image to visualize
    shape: tuple of mediator shape
    """
    import matplotlib.pyplot as plt
    x,y = shape[0],shape[1]*4
    plt.imshow(M[idx].reshape(x,y))
    plt.title("The z value for this image is %s"%str(z[idx])[2:6])
    plt.show()

def simulate_dataset(num_subs, resize,new_shape,visualize,algo='svr',alpha=0,std=1):
    """
    Simulate the dataset.
    num_subs :int; number of subjects;
    resize: bool; if you want to resize the images;
    new_shape: list of int; new shape of the images
    """

    #Simulate X with fixed
    X = np.random.normal(0,1,num_subs)
    a0,e0 = -0.1,np.random.normal(0,std,num_subs)
    m = a0 + alpha*X + e0
    z = norm.cdf(m)

    #Simulate images m
    M = []
    floating_pt = [str(i)[2:6].zfill(4) for i in z] # makes sure that there are 4 digits after decimal point
    for num in floating_pt:
        image = extract_image(num,mnist_dataset,resize,new_shape,algo)
        M.append(image)
    M = np.array(M) #multi-dimensional
    if algo != 'svr':
        M = M[:,:,:,np.newaxis] # for keras to add an extra dimension
    #simulate Y ratings
    beta,gamma,b0 = 4,5,6 #4,5,6
    error = np.random.normal(0,std,num_subs)
    Y = b0 + gamma*X + beta*m + error 
    
    df = pd.DataFrame()
    df['X'] = X
    df['Y'] = Y
    df['m'] = m
    if visualize:
        idx = 0
        visualize_dataset(M,z,idx,new_shape)

    return df,M,z

def create_shallow_model2D(input_shape):
    """
    Creates the training model.
    train_df: Training dataframe
    train_y: Training labels
    hyperparams: a dictionary of hyperparameters with a list of variables as the values

    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(input_shape)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=None))
    model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(),metrics=['mse'])
    return model

def create_shallow_model3D(input_shape):
    """
    Creates the training model.
    train_df: Training dataframe
    train_y: Training labels
    hyperparams: a dictionary of hyperparameters with a list of variables as the values

    """
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3),activation='relu',input_shape=(input_shape)))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=None))
    model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(),metrics=['mse'])
    return model


def create_svr_model(M,d,tune=True):
    svr = SVR(C=1,kernel='rbf',gamma='scale',verbose=True)
    if tune:
        parameters_grid = {'gamma': [1e-6, 1e-5,1e-4, 1e-3, 0.01, 0.1],
                       'C': [10,100,500,1000]#[1, 10, 100, 1000]
                      }
#         parameters_grid = {'kernel': ['rbf'],'gamma': [1e-6, 1e-5,1e-4, 1e-3, 0.01, 0.1],
#                        'C': [1, 10, 100, 1000]
#                       }
        svr_grid = GridSearchCV(svr, parameters_grid,cv=5,n_jobs=-1,verbose=2)
        svr_grid.fit(M,d)   
        model = svr_grid.best_estimator_
    else:
        model=svr.fit(M,d)
      
    return model

def create_deep_model3D(input_shape):
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

def create_deep_model2D(input_shape):
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
    X = Conv2D(filters = 8, kernel_size = (3, 3),padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),
                                                      kernel_initializer=initType)(X_input)
    X = BatchNormalization(axis = -1)(X)#, trainable=True)(X)
    #X = BatchRenormalization(axis = -1)(X)
    #X.trainable=True
    X = Activation('elu')(X)
    X = Conv2D(filters = 8, kernel_size = (3, 3), strides=(1,1), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),
                                                      kernel_initializer=initType)(X)
    X = BatchNormalization(axis = -1)(X)
    #X = Activation('elu')(X)

    X_shortcut = Conv2D(8, kernel_size = (1,1), strides=(1,1), padding=paddingType,kernel_initializer=initType)(X_input)
    X = Add()([X_shortcut,X])
    outputs = Activation('elu')(X)

    pooling = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding=paddingType)(outputs)


   #######################################################################################################################

    inputs = pooling
    features = 16
    hidden = Conv2D(features, (3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(inputs)
    hidden = BatchNormalization(axis=-1)(hidden)
    hidden = Activation('elu')(hidden)

    hidden = Conv2D(features, (3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(hidden)
    hidden = BatchNormalization(axis=-1)(hidden)

    shortcut = Conv2D(features, (1,1), strides=(1,1), padding=paddingType,kernel_initializer=initType)(inputs)
    hidden = Add()([shortcut,hidden])
    outputs = Activation('elu')(hidden)

    pooling = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding=paddingType)(outputs)

   ########################################################################################################################
    inputs = pooling
    features = 32
    hidden = Conv2D(features, (3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(inputs)
    hidden = BatchNormalization(axis=-1)(hidden)
    hidden = Activation('elu')(hidden)

    hidden = Conv2D(features, (3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(hidden)
    hidden = BatchNormalization(axis=-1)(hidden)

    shortcut = Conv2D(features, (1,1), strides=(1,1), padding=paddingType,kernel_initializer=initType)(inputs)
    hidden = Add()([shortcut,hidden])
    outputs = Activation('elu')(hidden)

    pooling = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding=paddingType)(outputs)

   ##########################################################################################################################

    inputs = pooling
    features = 64
    hidden = Conv2D(features, (3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(inputs)
    hidden = BatchNormalization(axis=-1)(hidden)
    hidden = Activation('elu')(hidden)

    hidden = Conv2D(features, (3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(hidden)
    hidden = BatchNormalization(axis=-1)(hidden)

    shortcut = Conv2D(features, (1,1), strides=(1,1), padding=paddingType,kernel_initializer=initType)(inputs)
    hidden = Add()([shortcut,hidden])
    outputs = Activation('elu')(hidden)

    pooling = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding=paddingType)(outputs)

   ######################################################################################################################
    inputs = pooling
    features = 128
    hidden = Conv2D(features, (3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(inputs)
    hidden = BatchNormalization(axis=-1)(hidden)
    hidden = Activation('elu')(hidden)

    hidden = Conv2D(features, (3, 3), padding=paddingType,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType)(hidden)
    hidden = BatchNormalization(axis=-1)(hidden)

    shortcut = Conv2D(features, (1,1), strides=(1,1), padding=paddingType,kernel_initializer=initType)(inputs)
    hidden = Add()([shortcut,hidden])
    outputs = Activation('elu')(hidden)

    pooling = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding=paddingType)(outputs)


    hidden = Flatten()(pooling)
    hidden = Dense(128,kernel_regularizer=regularizers.l2(regAmount),kernel_initializer=initType,name='FullyConnectedLayer')(hidden)
    hidden = Activation('elu')(hidden)
    hidden = Dropout(0.8)(hidden)#, training=True) # dropout layer was not used during testing, so had to add this,check https://github.com/keras-team/keras/issues/9412

    prediction = Dense(1,kernel_regularizer=regularizers.l2(regAmount), name='PainPrediction')(hidden)

    model = Model(inputs = X_input, outputs = prediction)

    return model

def predict(model,X_test,y_test,plot=True,result_path=None, name=None):

    y_pred = model.predict(X_test)
    if len(y_pred.shape) >1 :
        y_pred = np.concatenate(y_pred)
    score = np.corrcoef(y_pred,y_test)
    score = score[0,1]
    if plot:
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))
        axs = np.ravel(axs)
        ax = axs[0]
        mae = mean_absolute_error(y_pred,y_test)
        plot_regression_results(ax, y_test, y_pred,
            name,(r'$R={:.3f}$' + '\n'+r'$MAE={:.3f}$').format(score,mae))

        plt.tight_layout(pad=2.0)
        if result_path == None:
            result_path = os.getcwd()
        fname = str('parity-plot-'+name+'.png')
        plt.savefig(os.path.join(result_path,fname), dpi=300)
        plt.show()

    return score,y_pred
