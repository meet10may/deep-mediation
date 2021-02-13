from PIL import Image
import scipy.io as sio
import numpy as np
import os, pickle,glob
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import load_model
# from tensorflow.keras.callbacks import EarlyStopping # specific for tf >2
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from scipy.stats import zscore
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img,load_img,index_img,resample_img,concat_imgs,load_img,resample_to_img,threshold_img,mean_img
from nilearn import plotting
from scipy.stats import zscore,norm
from sklearn.datasets import fetch_openml
mnist_dataset = fetch_openml('mnist_784')
import build_model

# import sys
# sys.path.append('/home/ubuntu/hacking/projects/deep-mediation/dec-2020/keras-vis-master')
# from vis.visualization import visualize_cam
# from vis.visualization import visualize_saliency, overlay
# from vis.utils import utils as vis_utils


def save_as_pickle(data,filename):
    with open(filename, 'wb') as handle:
        return pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
    
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
            output = concat_mnist_images(output, img)
        if algo == 'svr':
            output_ = output.flatten()
#         if not cnn:
#             output_ = output.flatten()
    if algo == 'svr':
        output = output_
    image.append(output)
    return np.concatenate(image) 

def simulate_dataset(num_subs, resize,new_shape,visualize,algo='svr',alpha=0):
    """
    Simulate the dataset.
    num_subs :int; number of subjects;
    resize: bool; if you want to resize the images;
    new_shape: list of int; new shape of the images
    """

    #Simulate X with fixed
    X = np.random.normal(0,1,num_subs)
    a0,e0 = -0.1,np.random.normal(0,1,num_subs)
    m = a0 + alpha*X + e0
    z = norm.cdf(m)

    #Simulate images m
    M = []
    floating_pt = [str(i)[2:6] for i in z]
    for num in floating_pt:
        image = extract_image(num,mnist_dataset,resize,new_shape,algo)
        M.append(image)
    M = np.array(M) #multi-dimensional
    if algo != 'svr':
        M = M[:,:,:,np.newaxis] # for keras to add an extra dimension
    #simulate Y ratings

    beta,gamma,b0 = 4,5,6 #4,5,6
    error = np.random.normal(0,1,num_subs)
    Y = b0 + gamma*X + beta*m + error 
    
    df = pd.DataFrame()
    df['X'] = X
    df['Y'] = Y
    df['m'] = m
    
    #Create train and test dataset
    X_train, X_test, y_train, y_test = create_train_data(df,M,testing_dataset_size=0.5)
    if visualize:
        idx = 0
        visualize_dataset(M,z,idx,new_shape)

    return X_train, X_test, y_train, y_test

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

def create_train_data(X,y,testing_dataset_size=0.5):
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

def get_rate_temp_img(dataset,subjs='train_subjs'):
    rate = [dataset[i][0] for i in list(dataset[subjs])]
    rate_zs = [zscore(dataset[i][0]) for i in list(dataset[subjs])]
    rate = np.array(rate)
    rate_zs = np.array(rate_zs)
    temp = [dataset[i][1] for i in list(dataset[subjs])]
    temp = np.array(temp)
    imgs = dataset[subjs]
    flat_rate = np.array([item for sublist in rate for item in sublist])
    flat_rate_zs = np.array([item for sublist in rate_zs for item in sublist])
    return rate,np.concatenate(temp),imgs,flat_rate,flat_rate_zs

# def create_empty_df(num_runs,num_iters):
#     # Creates an empty dataFrame
#     a = np.empty((num_runs,num_iters))
#     a[:] = np.nan
#     dataFrame = None
#     parameters = ['alpha', 'beta','theta','gamma','delta']
#     for params in parameters:
#         iter = ['iter_'+str(i) for i in range(num_iters)]
#         pdindex = pd.MultiIndex.from_product([[params], iter],
#                                              names=['parameters', 'runs']) 
#         frame = pd.DataFrame(a, columns = pdindex,index = range(0,num_runs))
#         dataFrame = pd.concat([dataFrame,frame],axis=1)
#     return dataFrame

def create_empty_df(num_runs,num_iters):
    # Creates an empty dataFrame
    a = np.empty((num_runs,num_iters))
    a[:] = np.nan
    dataFrame = None
    parameters = ['alpha0', 'beta0','alpha','beta','gamma']
    for params in parameters:
        iter = ['iter_'+str(i) for i in range(num_iters)]
        pdindex = pd.MultiIndex.from_product([[params], iter],
                                             names=['parameters', 'runs']) 
        frame = pd.DataFrame(a, columns = pdindex,index = range(0,num_runs))
        dataFrame = pd.concat([dataFrame,frame],axis=1)
    return dataFrame

def simulate_mediation(df_train,df_test,X_train,X_test,train_params_df,test_params_df,
                       model,suffix,n_runs,batchSize,nEpochs,iterations,pat,output_path=None,
                       use_model=None,algo='deep'):
    """
    """
    
    # Initialize z with some phi
    if use_model!=None:
        model = load_model(use_model)
        start = int(Path(use_model).stem.split('-')[-1]) 
        print("Using already saved model iteration %s..."%start)
        z_train = model.predict(X_train) 
    else:
        print("Using intial random model parameters")
        start = 0
        z_train = model.predict(X_train) 
    try:
        z_train = np.concatenate(z_train)
    except:
        pass
  
    for i in range(start,iterations):
        print('Starting iteration... %s'%(i+1))
        # Check for directionality
        if np.corrcoef(z_train,df_train.Y)[0,1] < 0:
              z_train = z_train*(-1)
        # Check for scaling issue
        z_train = zscore(z_train)
        
        lm_train = smf.ols(formula='z_train ~ X', data=df_train).fit()
        alpha0_train = lm_train.params.loc['Intercept']
        alph_train = lm_train.params.loc['X']
        
        lm_train = smf.ols(formula='Y ~ z_train + X', data=df_train).fit()
        beta0_train = lm_train.params.Intercept
        bet_train = lm_train.params.z_train
        gam_train = lm_train.params.X
        resid_std_train = np.std(lm_train.resid)

        e_train = df_train.Y - beta0_train - (df_train.X*gam_train)
        h_train = alpha0_train + df_train.X*alph_train
        d_train = (((bet_train*e_train)+h_train)/((bet_train**2)+1))
        d_train = np.array(d_train)
        
        output_file_name = algo+'-iter-'+str(i+1)+'-run-'+str(n_runs)+suffix
        
        if algo == 'svr':
            model = build_model.create_svr_model(X_train,d_train)
            z_train = model.predict(X_train)
        else:
            early = EarlyStopping(monitor='val_loss', patience=pat,mode='min',verbose=1)
            mc = ModelCheckpoint(filepath=os.path.join(output_path,output_file_name+'.h5'), 
                             verbose=1, monitor='val_loss',save_best_only=True)
            cb = [early, mc] 
            hist = model.fit(X_train,d_train,batch_size=batchSize,validation_split=0.3,
                                  epochs=nEpochs,verbose=1, shuffle=True,callbacks=cb)
            z_train = model.predict(X_train)
            plot_train_loss(hist,os.path.join(output_path,output_file_name+'.png'))
            
# Save the params in the pandas dataframe
        train_params_df.loc[n_runs]['alpha0','iter_'+str(i)]=alpha0_train
        train_params_df.loc[n_runs]['beta0','iter_'+str(i)]=beta0_train
        train_params_df.loc[n_runs]['beta','iter_'+str(i)]=bet_train
        train_params_df.loc[n_runs]['gamma','iter_'+str(i)]=gam_train
        train_params_df.loc[n_runs]['alpha','iter_'+str(i)]=alph_train
        train_params_df.to_excel(os.path.join(output_path,output_file_name+'-train.xlsx'))
        try:
            z_train = np.concatenate(z_train)
        except:
            pass
        
        alpha0_test,alph_test,beta0_test,bet_test,gam_test,z_test = predict_mediation(model,X_test,df_test)
        test_params_df.loc[n_runs]['alpha0','iter_'+str(i)]=alpha0_train
        test_params_df.loc[n_runs]['beta0','iter_'+str(i)]=beta0_train
        test_params_df.loc[n_runs]['beta','iter_'+str(i)]=bet_train
        test_params_df.loc[n_runs]['gamma','iter_'+str(i)]=gam_train
        test_params_df.loc[n_runs]['alpha','iter_'+str(i)]=alph_train
        test_params_df.to_excel(os.path.join(output_path,output_file_name+'-test.xlsx'))

    return train_params_df,test_params_df,z_train

def predict_mediation(model,test_imgs,df_test):
    z_test = model.predict(test_imgs)
    try:
        z_test = np.concatenate(z_test)
    except:
        pass
    
    if np.corrcoef(z_test,df_test.Y)[0,1] < 0:
          z_test = z_test*(-1)
    # Check for scaling issue
    z_test = zscore(z_test)
    lm_test = smf.ols(formula='z_test ~ X', data=df_test).fit()
    alpha0_test = lm_test.params.loc['Intercept']
    alph_test = lm_test.params.loc['X']
    
    lm_test = smf.ols(formula='Y ~ z_test + X', data=df_test).fit()
    beta0_test = lm_test.params.Intercept
    bet_test = lm_test.params.z_test
    gam_test = lm_test.params.X
    resid_std_test = np.std(lm_test.resid)
    
    return alpha0_test,alph_test,beta0_test,bet_test,gam_test,z_test

def concat_images(image_list,output_fname):
    image_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    all_images = concat_imgs(image_list)
    if output_fname != None:
        all_images.to_filename(output_fname)
    return all_images
    
def concat_mnist_images(imga, imgb):
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


def get_gradCAM_map(input_data,model,layer_name,penultimate_layer_name,trial,filter_indices=None,
                    backprop_modifier=None,output_path=None,resample=True,threshold=True,
                    fname='activation-3',plot=True,affine_mat=None,hdr=None):
    
    layer_idx = vis_utils.find_layer_idx(model, layer_name)
    penultimate_layer = vis_utils.find_layer_idx(model, penultimate_layer_name)
    if isinstance(input_data,str):
        data = np.array(nib.load(input_data).get_fdata())
        affine_mat = nib.load(input_data).affine
        hdr = nib.load(input_data).header
    else:
        data = input_data
    img = np.rollaxis(data, 0, 0)[...,None]
    grads = visualize_cam(model, layer_idx, filter_indices, seed_input=img, 
                          penultimate_layer_idx=penultimate_layer, backprop_modifier=None)   
#     grads = visualize_cam(model, layer_idx, filter_indices, seed_input=img[10:15,:,:,:,:], 
#                           penultimate_layer_idx=penultimate_layer, backprop_modifier=None)
    template = load_mni152_template()
    if resample:
        grads_ = nib.Nifti1Image(grads, affine_mat, hdr)
        resampled_grads_img = resample_to_img(grads_, template)
        output_fname = os.path.join(output_path,'trial-'+str(trial)+'-'+fname+'.nii')
        resampled_grads_img.to_filename(output_fname)
        if plot:
            resampled_plot_fname = os.path.join(output_path,'trial-'+str(trial)+'-'+fname)
            plotting.plot_stat_map(resampled_grads_img, display_mode='z', cut_coords=[36, -27, 60],
                           title='Unthresholded beta-map', 
                           output_file=resampled_plot_fname,colorbar=True)
    else:
        output_fname = os.path.join(output_path,'trial-'+str(trial)+'-'+fname+'.nii')
        grads_ = nib.Nifti1Image(grads, affine_mat, hdr)
        grads_.to_filename(output_fname)
        
    if threshold:
        threshold_percentile_img = threshold_img(output_fname, threshold='97%', copy=False)
        thresholded_fname = os.path.join(output_path,'trial-'+str(trial)+'-'+fname+'-threshold.nii')
        threshold_percentile_img.to_filename(thresholded_fname)
        print("Thresholded image saved!")
        if plot:
            threshold_plot_fname = os.path.join(output_path,'trial-'+str(trial)+'-'+fname+'-threshold')
            plotting.plot_stat_map(threshold_percentile_img, display_mode='z', cut_coords=[36, -27, 60],
                                   title='Thresholded beta-map with 97% percentile', 
                                   output_file=threshold_plot_fname,colorbar=True)
       
    return grads

def threshold_image(images,mean=True,output_fname='thresholded-file',threshold_value='80%'):
    if mean:
        mean_image = mean_img(images)
    threshold_percentile_img = threshold_img(mean_image,threshold=threshold_value, copy=False)
    threshold_percentile_img.to_filename(output_fname)
    plotting.plot_stat_map(threshold_percentile_img, display_mode='z', 
                           title='Thresholded beta-map with %s percentile'%threshold_value, 
                           output_file=output_fname,colorbar=True)
    print("Thresholded images saved!")
    return None

def read_sound_files(files,kind='temp'):
    
    mat_file = []
    if kind == 'nii':
        out = concat_imgs(files)
    elif kind=='rate':
        for i in files:
            mat_file.append(zscore(np.concatenate(sio.loadmat(i)[kind])))
        out = np.concatenate(mat_file)
    else:
        for i in files:
            mat_file.append(np.concatenate(sio.loadmat(i)[kind]))
        out = np.concatenate(mat_file)
        out = out + 46
    return out

def get_sound_files(path):
    sound_temp_files= glob.glob(os.path.join(path,'temps*'))
    sound_temp_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    sound_ratings_files= glob.glob(os.path.join(path,'ratings*'))
    sound_ratings_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    sound_images_files= glob.glob(os.path.join(path,'*.nii'))
    sound_images_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return sound_temp_files,sound_ratings_files,sound_images_files
    
