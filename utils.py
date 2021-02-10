import scipy.io as sio
import numpy as np
import os, pickle,glob
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from scipy.stats import zscore
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img,load_img,index_img,resample_img,concat_imgs,load_img,resample_to_img,threshold_img,mean_img
from nilearn import plotting


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
        
# def create_dataset(data_path):
#     print("Test")
#     data_info = load_pickle(os.path.join(data_path,'studyinfo.pickle'))
#     studies_to_include = load_pickle(os.path.join(data_path,'studyinfo.pickle'))['study'][1:8]
#     test_study = load_pickle(os.path.join(data_path,'studyinfo.pickle'))['study'][-2]
#     print("These are the studies to use %s"%studies_to_include)
#     print("This is the test study %s "%test_study)
#     data_to_include = [os.path.join(data_path,i+'_data') for i in studies_to_include]

#     imgs = []
#     for i in data_to_include:
#         print(i)
#         if 'ILCP' in i:
#             data = [os.path.join(data_path,'ILCP'+'_data',s+'.nii') for s in data_info.subjects[5]]
#         elif 'EXP' in i:
#             data = [os.path.join(data_path,'EXP'+'_data',s+'.nii') for s in data_info.subjects[6]]
#         elif 'BMRK3' in i:
#             data = [os.path.join(data_path,'BMRK3'+'_data','bmrk3_st_'+s+'.nii') for s in data_info.subjects[2]]
#         else:
#             data = glob.glob(os.path.join(i,'*.nii')) 
#             data.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
#         imgs = imgs+data

#     test_data = [os.path.join(data_path,'BMRK5_data',s.replace('sub','stim_bmrk5_S')+'*.nii') for s in sum(data_info.subjects[-2], [])]
#     test_data = [glob.glob(i) for i in test_data]
#     test_data = sum(test_data,[])
#     dataset= {}
#     NSF = {}
#     BMRK3 = {}
#     BMRK4 = {}
#     IE = {}
#     EXP = {}
#     ILCP = {}
#     SCEBL = {}
#     BMRK5 = {}

#     for i in range(0,26):
#         NSF[imgs[i]] = [data_info.rate[1][i],data_info.temp[1][i]]

#     for i in range(0,33):
#         BMRK3[imgs[26+i]] = [data_info.rate[2][i],data_info.temp[2][i]]
    
#     for i in range(0,28):
#         BMRK4[imgs[59+i]] = [data_info.rate[3][i],data_info.temp[3][i]]
    
#     for i in range(0,50):
#         IE[imgs[87+i]] = [data_info.rate[4][i],data_info.temp[4][i]]
    
#     for i in range(0,29):
#         ILCP[imgs[137+i]] = [data_info.rate[5][i],data_info.temp[5][i]]
    
#     for i in range(0,17):
#         EXP[imgs[166+i]] = [data_info.rate[6][i],data_info.temp[6][i]]

#     for i in range(0,26):
#         SCEBL[imgs[183+i]] = [data_info.rate[7][i],data_info.temp[7][i]]

#     for i in range(0,75):
#         BMRK5[test_data[i]] = [data_info.rate[8][i],data_info.temp[8][i]]
 
#     dataset = {**NSF, **BMRK3,**BMRK4, **IE,**ILCP, **EXP,**SCEBL,**BMRK5}


#     train_subjs, val_subjs = train_test_split(imgs,test_size = 0.15,shuffle=True,random_state=42)
#     print("Number of training subjects: %s" %len(train_subjs))

#     dataset['train_subjs'] = train_subjs
#     dataset['val_subjs'] = val_subjs
#     dataset['study'] = data_info['study'][1:9]
#     dataset['N'] = data_info['N'][1:9]

#     return dataset

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

def simulate_mediation(df_train,df_val,df_test,X_train,X_val,X_test,params_df,model,n_runs,batchSize,nEpochs,iterations,pat,
                       output_path=None,use_model=None):
    """
    """
    
    # Initialize z with some phi
    
    if use_model!=None:
        model = load_model(use_model)
        start = int(Path(use_model).stem.split('-')[-1]) #int(Path(use_model).stem[-1])
        print("Using already saved model iteration %s..."%start)
        z_train = model.predict(X_train) 
#         z_val = model.predict(X_val)
    else:
        print("Using intial random model parameters")
        start = 0
        z_train = model.predict(X_train) 
#         z_val = model.predict(X_val)
    try:
        z_train = np.concatenate(z_train)
#         z_val = np.concatenate(z_val)
    except:
        pass
  
    for i in range(start,iterations):
        print('Starting iteration... %s'%(i+1))
        # Check for directionality
        if np.corrcoef(z_train,df_train.Y)[0,1] < 0:
              z_train = z_train*(-1)
#         if np.corrcoef(z_val,df_val.Y)[0,1] < 0:
#               z_val = z_val*(-1)
        # Check for scaling issue
        z_train = zscore(z_train)
#         z_val = zscore(z_val)
        
        lm_train = smf.ols(formula='z_train ~ X', data=df_train).fit()
        alpha0_train = lm_train.params.loc['Intercept']
        alph_train = lm_train.params.loc['X']
        
#         lm_val = smf.ols(formula='z_val ~ X', data=df_val).fit()
#         alpha0_val = lm_val.params.loc['Intercept']
#         alph_val = lm_val.params.loc['X']

        lm_train = smf.ols(formula='Y ~ z_train + X', data=df_train).fit()
        beta0_train = lm_train.params.Intercept
        bet_train = lm_train.params.z_train
        gam_train = lm_train.params.X
        resid_std_train = np.std(lm_train.resid)

#         lm_val = smf.ols(formula='Y ~ z_val + X', data=df_val).fit()
#         beta0_val = lm_val.params.Intercept
#         bet_val = lm_val.params.z_val
#         gam_val = lm_val.params.X
#         resid_std_val = np.std(lm_val.resid)

        e_train = df_train.Y - beta0_train - (df_train.X*gam_train)
        h_train = alpha0_train + df_train.X*alph_train
        d_train = (((bet_train*e_train)+h_train)/((bet_train**2)+1))
        d_train = np.array(d_train)
        
#         e_val = df_val.Y - beta0_val - (df_val.X*gam_val)
#         h_val = alpha0_val + df_val.X*alph_val
#         d_val = (((bet_val*e_val)+h_val)/((bet_val**2)+1))
#         d_val = np.array(d_val)
        
        early = EarlyStopping(monitor='val_loss', patience=pat,mode='min',verbose=1)
        mc = ModelCheckpoint(filepath=os.path.join(output_path,'model-iter-'+str(i+1)+'.h5'), verbose=1, monitor='val_loss',save_best_only=True)
        cb = [early, mc] 
#         hist = model.fit(X_train,d_train,batch_size=batchSize,validation_data=(X_val,d_val),
#                                   epochs=nEpochs,verbose=1, shuffle=True,callbacks=cb)
        hist = model.fit(X_train,d_train,batch_size=batchSize,validation_split=0.3,
                                  epochs=nEpochs,verbose=1, shuffle=True,callbacks=cb)
        
        
#         hist = parallel_model.fit(X_train,d_train, batch_size, shuffle=True,validation_data=dataGenerator(X_val, d_val, batch_size, meanImg=None, scaling=False,dim=X_train.shape[1:4], shuffle=True, augment=False, maxAngle=40, maxShift=10),epochs=1000,verbose=1,max_queue_size=32,workers=4,use_multiprocessing=False,steps_per_epoch=np.ceil(X_train.shape[0]/batch_size),callbacks=cb)        


        z_train = model.predict(X_train)
#         z_val = model.predict(X_val)
        
# Save the params in the pandas dataframe
        params_df.loc[n_runs]['alpha0','iter_'+str(i)]=alpha0_train
        params_df.loc[n_runs]['beta0','iter_'+str(i)]=beta0_train
        params_df.loc[n_runs]['beta','iter_'+str(i)]=bet_train
        params_df.loc[n_runs]['gamma','iter_'+str(i)]=gam_train
        params_df.loc[n_runs]['alpha','iter_'+str(i)]=alph_train
        params_df.to_excel(os.path.join(output_path,'params_'+str(i+1)+'.xlsx'))
        try:
            z_train = np.concatenate(z_train)
#             z_val = np.concatenate(z_val)
        except:
            pass
    
        plot_train_loss(hist,os.path.join(output_path,'model_loss_'+str(i+1)+'.png'))
        z_val = None
    return params_df,z_train,z_val,model

def predict_mediation(model,test_imgs,df_test):
    z_test = model.predict(test_imgs)
    z_test = np.concatenate(z_test)
    
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
    
