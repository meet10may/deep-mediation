import utils,keras_model
import create_dataset
import numpy as np
import os
from nilearn.image import resample_img,concat_imgs,load_img,resample_to_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.keras import activations

import seaborn as sns
print(tf.__version__)

import pandas as pd


# Seed value
# Apparently you may use different seed values at each stage
seed_value= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
# from tensorflow.keras import backend as K
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# for later versions:
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)


#######################################################################################################################

data_path = '/home/ubuntu/hacking/data/stephan-data-ni-files'
input_shape = (91,109,91,1)
result_path = '/home/ubuntu/hacking/projects/deep-mediation/dec-2020/results/attempt-0'
source_img_name = None
batchSize = 64
ngpus = 4
nEpochs = 1000
decayRate = 0.01
lr = 0.001
pat = 50
num_runs = 1
iterations = 20
use_transfer_learning = False
use_dynamic_LR=False
optimizer='Adam'

######################################################################################################################

dataset = create_dataset.generate_dataset(data_path,test_data_size=0.30)
train_rate,train_temp,train_imgs_list,flat_train_rate,flat_train_rate_zs = utils.get_rate_temp_img(dataset,subjs='train_subjs')
val_rate,val_temp,val_imgs_list,flat_val_rate,flat_val_rate_zs = utils.get_rate_temp_img(dataset,subjs='val_subjs')
test_rate,test_temp,test_imgs_list,flat_test_rate,flat_test_rate_zs = utils.get_rate_temp_img(dataset,subjs='test_subjs')

df_train = pd.DataFrame()
df_test = pd.DataFrame()
df_val = pd.DataFrame()

df_train['X'] = train_temp
df_train['Y'] = flat_train_rate_zs

df_val['X'] = val_temp
df_val['Y'] = flat_val_rate_zs

df_test['X'] = test_temp
df_test['Y'] = flat_test_rate_zs

#####################################################################################################################

train_imgs_list[0:5],val_imgs_list[0:5],test_imgs_list[0:5]

#####################################################################################################################

print("Reading training images...")
train_imgs = concat_imgs(train_imgs_list)
train_imgs = np.rollaxis(train_imgs.get_fdata(), 3, 0)[...,None]

print("Reading validation images...")
val_imgs = concat_imgs(val_imgs_list)
val_imgs = np.rollaxis(val_imgs.get_fdata(), 3, 0)[...,None]

print("Reading testing images...")
test_imgs = concat_imgs(test_imgs_list)
test_imgs = np.rollaxis(test_imgs.get_fdata(), 3, 0)[...,None]

print(train_imgs.shape,val_imgs.shape,test_imgs.shape)
input_shape = train_imgs.shape[1:]

pre_model = None #'/home/ubuntu/hacking/projects/deep-mediation/dec-2020/results/model-iter-12.h5'
print("######################## Loading the model #####################")

if tf.__version__ == '1.12':
  
    strategy = tf.distribute.MirroredStrategy()
    print('###################### Number of devices: {}######################'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        print("################Creating the model###################")
        parallel_model = keras_model.resnet_with_batchnorm(input_shape)
        opt = Adam(lr, beta_1=0.9, beta_2=0.999,decay=decayRate)
        parallel_model.compile(loss='mean_absolute_error',optimizer=opt)

        print("################################ Start mediation analysis ##################################")
        output_df = utils.create_empty_df(num_runs,iterations)
        for runs in range(num_runs):
            params_df,z_final_train,z_final_val,parallel_model_final = utils.simulate_mediation(df_train,
                                        df_val,df_test,train_imgs,val_imgs,test_imgs,output_df,
                                        parallel_model,runs,batchSize,nEpochs,iterations,pat,
                                        output_path=result_path,use_model=pre_model)
#             params_df.to_excel(os.path.join(result_path,'params_'+str(runs+1)+'.xlsx'))




