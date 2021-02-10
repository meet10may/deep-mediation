import scipy.io as sio
import numpy as np
import os
import mat73
import pickle,glob
from sklearn.model_selection import train_test_split

def save_as_pickle(data,filename):
    with open(filename, 'wb') as handle:
        return pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

    
def generate_dataset(data_path,test_data_size=0.30):
    data_info = load_pickle(os.path.join(data_path,'studyinfo.pickle'))
    studies_to_include = load_pickle(os.path.join(data_path,'studyinfo.pickle'))['study'][1:8]
    test_study = load_pickle(os.path.join(data_path,'studyinfo.pickle'))['study'][-2]
    print("These are the studies to use %s"%studies_to_include)
    print("This is the test study %s "%test_study)

    data_to_include = [os.path.join(data_path,i+'_data') for i in studies_to_include]

    imgs = []
    for i in data_to_include:
        if 'ILCP' in i:
            data = [os.path.join(data_path,'ILCP'+'_data',s+'_zs.nii') for s in data_info.subjects[5]]
        elif 'EXP' in i:
            data = [os.path.join(data_path,'EXP'+'_data',s+'_zs.nii') for s in data_info.subjects[6]]
        elif 'BMRK3' in i:
            data = [os.path.join(data_path,'BMRK3'+'_data','bmrk3_st_'+s+'_zs.nii') for s in data_info.subjects[2]]
        else:
            data = glob.glob(os.path.join(i,'*zs.nii')) 
            data.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        imgs = imgs+data

    test_data = [os.path.join(data_path,'BMRK5_data',s.replace('sub','stim_bmrk5_S')+'*zs.nii') for s in sum(data_info.subjects[-2], [])]
    test_data = [glob.glob(i) for i in test_data]
    test_data = sum(test_data,[])

    dataset= {}
    NSF = {}
    BMRK3 = {}
    BMRK4 = {}
    IE = {}
    EXP = {}
    ILCP = {}
    SCEBL = {}
    BMRK5 = {}

    for i in range(0,26):
        NSF[imgs[i]] = [data_info.rate[1][i],data_info.temp[1][i]]

    for i in range(0,33):
        BMRK3[imgs[26+i]] = [data_info.rate[2][i],data_info.temp[2][i]]

    for i in range(0,28):
        BMRK4[imgs[59+i]] = [data_info.rate[3][i],data_info.temp[3][i]]

    for i in range(0,50):
        IE[imgs[87+i]] = [data_info.rate[4][i],data_info.temp[4][i]]

    for i in range(0,29):
        ILCP[imgs[137+i]] = [data_info.rate[5][i],data_info.temp[5][i]]

    for i in range(0,17):
        EXP[imgs[166+i]] = [data_info.rate[6][i],data_info.temp[6][i]]

    for i in range(0,26):
        SCEBL[imgs[183+i]] = [data_info.rate[7][i],data_info.temp[7][i]]

    for i in range(0,75):
        BMRK5[test_data[i]] = [data_info.rate[8][i],data_info.temp[8][i]]

    dataset = {**NSF, **BMRK3,**BMRK4, **IE,**ILCP, **EXP,**SCEBL,**BMRK5}

#     train_subjs, val_subjs = train_test_split(imgs,test_size = test_data_size,shuffle=True,random_state=42)
#     print("Number of training subjects: %s" %len(train_subjs))

#     dataset['train_subjs'] = train_subjs
#     dataset['val_subjs'] = val_subjs
#     dataset['test_subjs'] = test_data#list(BMRK5.keys())
#     dataset['study'] = data_info['study'][1:9]
#     dataset['N'] = data_info['N'][1:9]

    dataset['train_subjs'] = imgs
#     dataset['val_subjs'] = val_subjs
    dataset['test_subjs'] = test_data#list(BMRK5.keys())
    dataset['study'] = data_info['study'][1:9]
    dataset['N'] = data_info['N'][1:9]
    
    return dataset    

# def generate_dataset(data_path,test_data_size=0.30):
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
#             data = [os.path.join(data_path,'ILCP'+'_data',s+'_zs.nii') for s in data_info.subjects[5]]
#         elif 'EXP' in i:
#             data = [os.path.join(data_path,'EXP'+'_data',s+'_zs.nii') for s in data_info.subjects[6]]
#         elif 'BMRK3' in i:
#             data = [os.path.join(data_path,'BMRK3'+'_data','bmrk3_st_'+s+'_zs.nii') for s in data_info.subjects[2]]
#         else:
#             data = glob.glob(os.path.join(i,'*.nii')) 
#         data.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
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
    
# #     imgs = [images for images,rate_temp in dataset.items()]
#     train_subjs, val_subjs = train_test_split(imgs,test_size = test_data_size,shuffle=True,random_state=42)
#     print("Number of training subjects: %s" %len(train_subjs))
#     print("Number of validation subjects: %s" %len(val_subjs))
#     print("Number of testing subjects: %s" %len(BMRK5))

#     train = {}
#     val = {}
# #     for i in train_subjs:
# # #         print(i)
# #         train[i] = dataset[i]
# #     for i in val_subjs:
# #         val[i] = dataset[i]


#     dataset['train_subjs'] = train_subjs
#     dataset['val_subjs'] = val_subjs
#     dataset['test_subjs'] = BMRK5
#     dataset['study'] = data_info['study'][1:9]
#     dataset['N'] = data_info['N'][1:9]
    
#     return dataset


def get_matfiles(datapath, studies):
    mat_files = []
    for i in studies:
        if i == 'SCEBL':
            mat_files.append(glob.glob(os.path.join(datapath,i,'*'+i+'*.mat')))   
        else:
            mat_files.append(glob.glob(os.path.join(datapath,i,'*'+i.lower()+'*.mat')))
    return np.array(sum(mat_files,[]))

def combine_mat_files(mat_files):
    X = []
    rate = []
    temp = []
    for i in mat_files:
        try:
            X.append(mat73.loadmat(i)['X'])
            rate.append(mat73.loadmat(i)['Rate'])
            temp.append(mat73.loadmat(i)['Temp'])
        except:
            X.append(sio.loadmat(i)['X'])
            rate.append(sio.loadmat(i)['Rate'])
            temp.append(sio.loadmat(i)['Temp'])
    X = np.hstack(X)
    X = np.rollaxis(X,1, 0)
    rate = np.concatenate(rate)
    temp = np.concatenate(temp)
    #rate = np.vstack(rate)
    #temp = np.vstack(temp)
    return X,rate,temp

