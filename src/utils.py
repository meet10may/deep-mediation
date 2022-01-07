import pickle,os
import numpy as np
import nibabel as nib
import scipy.io as sio
from pathlib import Path
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import keras
#from keras.layers import Dense
#from keras.models import Model
#from keras import regularizers

import random as python_random
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import zscore,norm
import pandas as pd
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
import tensorflow as tf
from datagenerator import dataGenerator
np.random.seed(123)
python_random.seed(123)
#import ceil

def save_as_pickle(data,filename):
    with open(filename, 'wb') as handle:
        return pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def load_existing_dataset(path):
    """
    Loads the existing train,test and validation datasets.
    """
    print('Loading dataset..')
    X_train = nib.load(os.path.join(path,'X_train.nii.gz'))
    X_train = np.rollaxis(X_train.get_fdata(), 3, 0)[...,None]
    #X_train = X_train.astype(np.int16)

    X_test = nib.load(os.path.join(path,'X_test.nii.gz'))
    X_test = np.rollaxis(X_test.get_fdata(), 3, 0)[...,None]
    #X_test = X_test.astype(np.int16)

    X_val = nib.load(os.path.join(path,'X_val.nii.gz'))
    X_val = np.rollaxis(X_val.get_fdata(), 3, 0)[...,None]
    #X_val = X_val.astype(np.int16)

    y_train = sio.loadmat(os.path.join(path,'y_train.mat'))['rate']
    y_test = sio.loadmat(os.path.join(path,'y_test.mat'))['rate']
    y_val = sio.loadmat(os.path.join(path,'y_val.mat'))['rate']

    y_train = y_train.T
    y_val = y_val.T
    y_test = y_test.T

    X_train = np.vstack([X_train,X_val])
    y_train = np.vstack([y_train,y_val])
    print("Data loaded!")
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    return X_train,y_train,X_test,y_test,X_val,y_val

def combine_images(images,outfname,padding=True,shape=(80,96,72),save=True):
    """
    Combines multiple images to make 1 image of the size (#num_images,shape,1)
    """
    tmp = []
    for i in images:
        print("Reading image %s"%i)
        if padding:
            data = pad_images(nib.load(i).get_fdata(),shape)
        else:
            data = pad_images(nib.load(i).get_fdata())
        tmp.append(data)
    all_images = np.concatenate(tmp,axis=3)
    all_images.astype(np.int16)
    img = nib.Nifti1Image(all_images, np.eye(4))
    img.get_data_dtype() == np.dtype(np.int16)
    if save:
        nib.save(img, outfname)
    img = img.get_fdata()
    img = img.reshape(img.shape+(1,))
    img = np.rollaxis(np.rollaxis(img, 3), 1, 1)
    return img

def pad_images(images,shape):
    image_shape = images.shape
    temp = np.zeros([shape[0],shape[1],shape[2],image_shape[-1]])

    temp[temp.shape[0]-images.shape[0]:,temp.shape[1]-images.shape[1]:,temp.shape[2]-images.shape[2]:,:] = images
    images = temp
    return images

def combine_rate(ratings,outfname,scaling,save):
    f0 = []
    f0 = np.array(f0)
    for f in ratings:
        print(f)
        f1 = np.concatenate(sio.loadmat(f)['rate'])
        f0 = np.hstack([f0,f1])

    if scaling:
        scaler = StandardScaler()
        f0 = scaler.fit_transform(f0.reshape(-1,1))
        f0 = np.concatenate(f0)

    # save as a mat file
    ratings ={}
    ratings['rate'] = f0
    if save:
        sio.savemat(outfname,ratings)
    return f0

def combine_temp(input_temp,outfname,scaling,save):
    f0 = []
    f0 = np.array(f0)
    for f in input_temp:
        print(f)
        f1 = np.concatenate(sio.loadmat(f)['temp'])
        f0 = np.hstack([f0,f1])

    if scaling:
        scaler = StandardScaler()
        f0 = scaler.fit_transform(f0.reshape(-1,1))
        f0 = np.concatenate(f0)

    # save as a mat file
    #temp ={}
    #temp['temp'] = f0
    if save:
        temp = {}
        temp['temp'] = f0
        sio.savemat(outfname,temp)
    return f0


def plot_train_loss(history,name,save=True):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if save:
        plt.savefig(name)

def make_prediction(model,X_train,X_test,X_val,batchSize):
    """
    Make the predictions using the model. 
    """
    y_pred_train = model.predict(X_train,batch_size=batchSize)
    y_pred_test = model.predict(X_test,batch_size=batchSize)
    y_pred_val = model.predict(X_val,batch_size=batchSize)
    # concatenate the predictions
    y_pred_train = np.concatenate(y_pred_train)
    y_pred_test = np.concatenate(y_pred_test)
    y_pred_val = np.concatenate(y_pred_val)
    return y_pred_train,y_pred_test,y_pred_val

def save_prediction_result(y_pred_train,y_pred_test,y_pred_val,y_train,y_test,y_val,result_path,fname):
    prediction= {}
    prediction['pred_train'] = y_pred_train
    prediction['pred_test'] = y_pred_test
    prediction['pred_val'] = y_pred_val
    prediction['true_train'] = y_train.T
    prediction['true_test']= y_test.T
    prediction['true_val'] = y_val
    sio.savemat(os.path.join(result_path,fname+'.mat'),prediction)
    return prediction

def plot_regression(y_test,y_pred,path=None,show=True,fname='test'):

    plt.clf()
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test,y_pred)
    sns.regplot(x=y_test,y=y_pred,line_kws={'label':"r={0:.2f}".format(r_value)})
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.legend()
    fname = fname.replace(" ","_")
    plt.title(fname)
    if path==None:
        path=os.getcwd()
    plt.savefig(os.path.join(path,'regression_plot_'+fname+'.png'))
    if show:
        plt.show()
    return(r_value)

def get_image_ratings(path,normalize=True,normalize_by='by_subject'):
    """
    Returns the list of all images and ratings
    normalize : use normalized by study or by subject.By default it normalizes by subject
    """
    imgs = []
    ratings = []
    for i in path:
        img_files = glob.glob(os.path.join(i,'*.nii'))
        img_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        if normalize:
            rating_files = glob.glob(os.path.join(i,normalize_by,'normalize*.mat'))
            rating_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        else:
            rating_files = glob.glob(os.path.join(i,'*.mat'))
            rating_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        imgs.append(img_files)
        ratings.append(rating_files)

    image_list = [item for sublist in imgs for item in sublist]
    rating_list = [item for sublist in ratings for item in sublist]

    return image_list,rating_list

def load_training_dataset(data_path,datafile,normalize_by='by_subject',scaling=True,padding=True,input_shape=(79, 95, 69)):
    """
    Loads the training dataset from datset.pickle file and returns the train, test and val dataset
    data_path: str full path of the data
    datafile: pickle file which contains the information of subject IDs in the train,test and val data
    normalize_by: str. Either 'by_subject' or 'by_study'. By default it is set to 'by_subject'
    """
    #Read pickle file
    dataset = load_pickle(datafile)

    #Read images
    dataset['train_subs_imgs'] = [os.path.join(data_path,Path(i).parent.name,Path(i).stem+'.nii')
    for i in dataset['train_subs']]
    dataset['val_subs_imgs'] = [os.path.join(data_path,Path(i).parent.name,Path(i).stem+'.nii')
    for i in dataset['val_subs']]
    dataset['test_subs_imgs'] = [os.path.join(data_path,Path(i).parent.name,Path(i).stem+'.nii')
    for i in dataset['test_subs']]

    #Read ratings
    dataset['train_subs_ratings'] = [os.path.join(data_path,Path(i).parent.name,normalize_by,
                                                  'normalized_by_subject_ratings_'+Path(i).stem+'.mat')
                                                  for i in dataset['train_subs']]
    dataset['val_subs_ratings'] = [os.path.join(data_path,Path(i).parent.name,normalize_by,
                                                'normalized_by_subject_ratings_'+Path(i).stem+'.mat')
                                                for i in dataset['val_subs']]
    dataset['test_subs_ratings'] = [os.path.join(data_path,Path(i).parent.name,normalize_by,
                                                 'normalized_by_subject_ratings_'+Path(i).stem+'.mat')
                                                 for i in dataset['test_subs']]

    #Read ratings
    dataset['train_subs_input_temp'] = [os.path.join(data_path,Path(i).parent.name,'temp_'+Path(i).stem+'.mat')
                                                  for i in dataset['train_subs']]
    dataset['val_subs_input_temp'] = [os.path.join(data_path,Path(i).parent.name,'temp_'+Path(i).stem+'.mat')
                                                for i in dataset['val_subs']]
    dataset['test_subs_input_temp'] = [os.path.join(data_path,Path(i).parent.name,'temp_'+Path(i).stem+'.mat')
                                                 for i in dataset['test_subs']]
    


    X_train = combine_images(dataset['train_subs_imgs'],'X_train',padding,shape=input_shape,save=False)
    y_train = combine_rate(dataset['train_subs_ratings'],'y_train.mat',scaling,save=False)
    temp_train = combine_temp(dataset['train_subs_input_temp'],'temp_train',scaling=False,save=False)

    X_val = combine_images(dataset['val_subs_imgs'],'X_val',padding,shape=input_shape,save=False)
    y_val = combine_rate(dataset['val_subs_ratings'],'y_val.mat',scaling,save=False)
    temp_val = combine_temp(dataset['val_subs_input_temp'],'temp_val',scaling=False,save=False)

    X_test = combine_images(dataset['test_subs_imgs'],'X_test',padding,shape=input_shape,save=False)
    y_test = combine_rate(dataset['test_subs_ratings'],'y_test.mat',scaling,save=False)
    temp_test = combine_temp(dataset['test_subs_input_temp'],'temp_test',scaling=False,save=False)

    return X_train,y_train,temp_train,X_val,y_val,temp_val,X_test,y_test,temp_test

def change_model(model, new_input_shape=(None, 79, 95, 69,1),change_last_layer=True): 
    # replace input shape of first layer 
    model.layers[0].batch_input_shape = new_input_shape 

    # rebuild model architecture by exporting and importing via json 
    new_model = keras.models.model_from_json(model.to_json()) 

    # copy weights from old model to new one 
    for layer in new_model.layers: 
        try: 
            layer.set_weights(model.get_layer(name=layer.name).get_weights()) 
            print("Loaded layer {}".format(layer.name)) 
        except: 
            print("Could not transfer weights for layer {}".format(layer.name)) 
    
    if change_last_layer:
        regAmount = 0.00005
        new_model.layers.pop()
        prediction = Dense(1,kernel_regularizer=regularizers.l2(regAmount), name='PainPrediction')(new_model.layers[-1].output)
        model_to_use = Model(inputs = new_model.input, outputs = prediction)
    else:
        model_to_use = new_model
    return model_to_use


def get_model_memory_usage(batch_size, model):
    import numpy as np
#    try:
#        from keras import backend as K
#    except:
#        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

def simulate_mediation(df_train,df_val,df_test,X_train,X_val,X_test,params_df,model,n_runs,batch_size,epochs,iterations,
                       output_path=None,use_model=None):
    """
    """
#     output_path = '/home/ubuntu/hacking/projects/deep-mediation/nov2020'
#     error = []
#     alpha = []
#     beta = []
#     theta = []
#     delta = []
#     gamma = []
#     r = []
#     hist = []
#     history = []
    # Initialize z with some phi
    
    if use_model!=None:
        print("Using already saved model...")
        z_train = model.predict(X_train) 
        z_val = model.predict(X_val)
    else:
        print("Using intial random model parameters")
        z_train = model.predict(X_train) 
        z_val = model.predict(X_val)
    try:
        z_train = np.concatenate(z_train)
        z_val = np.concatenate(z_val)
    except:
        pass
    # plot_scatter(z,df.m,labels=['z','m'])
    

    for i in range(0,iterations):
        print('Starting iteration... %s'%(i+1))
        # Check for directionality
        if np.corrcoef(z_train,df_train.Y)[0,1] < 0:
              z_train = z_train*(-1)
        if np.corrcoef(z_val,df_val.Y)[0,1] < 0:
              z_val = z_val*(-1)
        # Check for sclaing issue
        z_train = zscore(z_train)
        z_val = zscore(z_val)
        
        lm_train = smf.ols(formula='z_train ~ X', data=df_train).fit()
        alpha0_train = lm_train.params.loc['Intercept']
        alph_train = lm_train.params.loc['X']
        
        lm_val = smf.ols(formula='z_val ~ X', data=df_val).fit()
        alpha0_val = lm_val.params.loc['Intercept']
        alph_val = lm_val.params.loc['X']


        lm_train = smf.ols(formula='Y ~ z_train + X', data=df_train).fit()
        beta0_train = lm_train.params.Intercept
        bet_train = lm_train.params.z_train
        gam_train = lm_train.params.X
        resid_std_train = np.std(lm_train.resid)

        lm_val = smf.ols(formula='Y ~ z_val + X', data=df_val).fit()
        beta0_val = lm_val.params.Intercept
        bet_val = lm_val.params.z_val
        gam_val = lm_val.params.X
        resid_std_val = np.std(lm_val.resid)


        e_train = df_train.Y - beta0_train - (df_train.X*gam_train)
        h_train = alpha0_train + df_train.X*alph_train
        d_train = (((bet_train*e_train)+h_train)/((bet_train**2)+1))
        d_train = np.array(d_train)
        
        e_val = df_val.Y - beta0_val - (df_val.X*gam_val)
        h_val = alpha0_val + df_val.X*alph_val
        d_val = (((bet_val*e_val)+h_val)/((bet_val**2)+1))
        d_val = np.array(d_val)
        
        adam = Adam(lr=0.01, decay=0.03)
        model.compile(loss='mean_absolute_error',optimizer=adam)
        
        early = EarlyStopping(monitor='val_loss', patience=50,mode='min',verbose=1)
        mc = ModelCheckpoint(filepath=os.path.join(output_path,'model-'+str(i+1)+'.h5'), verbose=1, monitor='val_loss', save_best_only=True)
        cb = [early, mc] 
       
        hist = model.fit_generator(dataGenerator(X_train,d_train, batch_size, meanImg=None, scaling=False,dim=(79, 95, 69), shuffle=True, augment=False, maxAngle=40, maxShift=10),validation_data=dataGenerator(X_val, d_val, batch_size, meanImg=None, scaling=False,dim=(79, 95, 69), shuffle=True, augment=False, maxAngle=40, maxShift=10),epochs=1000,verbose=1,max_queue_size=32,workers=4,use_multiprocessing=False,steps_per_epoch=np.ceil(X_train.shape[0]/batch_size),callbacks=cb)        


        z_train = model.predict(X_train)
        z_val = model.predict(X_val)
        
# Save the params in the pandas dataframe
        params_df.loc[n_runs]['alpha0','iter_'+str(i)]=alpha0_train
        params_df.loc[n_runs]['beta0','iter_'+str(i)]=beta0_train
        params_df.loc[n_runs]['beta','iter_'+str(i)]=bet_train
        params_df.loc[n_runs]['gamma','iter_'+str(i)]=gam_train
        params_df.loc[n_runs]['alpha','iter_'+str(i)]=alph_train
#         save_as_pickle(hist,filename=os.path.join(output_path,'hist-'+str(i+1)+'.pickle'))
        params_df.to_excel(os.path.join(output_path,'params_'+str(i+1)+'.xlsx'))
        try:
            z_train = np.concatenate(z_train)
            z_val = np.concatenate(z_val)
        except:
            pass
    
        plot_train_loss(hist,os.path.join(output_path,'model_loss_'+str(i+1)+'.png'))
    return params_df,z_train,z_val,model

def extract_params_test_data(model,X_test,df_test,output_path=None):
    z_test = model.predict(X_test)
    z_test = np.concatenate(z_test)

    lm_test = smf.ols(formula='z_test ~ X', data=df_test).fit()
    delt_test = lm_test.params.loc['Intercept']        
    gam_test = lm_test.params.loc['X']        

    lm_test = smf.ols(formula='Y ~ z_test + X', data=df_test).fit()
    alph_test = lm_test.params.Intercept         
    bet_test = lm_test.params.z_test         
    thet_test = lm_test.params.X         
    resid_std_test = np.std(lm_test.resid)         

    test_results = {'Params':['alpha','beta','theta','delta','gamma'],'Value':[alph_test,bet_test,thet_test,delt_test,gam_test]}
    params_test = pd.DataFrame(test_results)
    params_test.to_excel(os.path.join(output_path,'params_test.xlsx'))

    return params_test

def create_empty_df(num_runs,num_iters):
    # Creates an empty dataFrame
    a = np.empty((num_runs,num_iters))
    a[:] = np.nan
    dataFrame = None
    parameters = ['alpha', 'beta','theta','gamma','delta']
    for params in parameters:
        iter = ['iter_'+str(i) for i in range(num_iters)]
        pdindex = pd.MultiIndex.from_product([[params], iter],
                                             names=['parameters', 'runs']) 
        frame = pd.DataFrame(a, columns = pdindex,index = range(0,num_runs))
        dataFrame = pd.concat([dataFrame,frame],axis=1)
    return dataFrame
