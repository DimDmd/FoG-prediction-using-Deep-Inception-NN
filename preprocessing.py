# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:54:12 2022

@author: ddimoudis
"""

import pandas as pd 
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import random
import tqdm

    
def premutation(data,target, percentage):
    x_more = int((percentage * len(target) - sum(target)) // (1 - percentage))
    if x_more <= 0 : raise ValueError("Percentage is met.")
    true_indices = np.argwhere(target).reshape(-1,)
    
    oversample = [np.random.choice(true_indices) for _ in range(x_more)]
    new_data = []
    
    for i in oversample:
        
        split_arrays = np.array_split(data[i],5)
        order = random.sample(range(0,5),5)
        new_data.append(np.concatenate([split_arrays[order[0]],split_arrays[order[1]],
                                       split_arrays[order[2]],split_arrays[order[3]],
                                       split_arrays[order[4]]]))
    extra_labels = [1 for i in oversample]
    new_data = np.concatenate([data,new_data])
    new_labels = np.concatenate([target,extra_labels])    
    return new_data, pd.Series(new_labels)

def sigmas_rule(df,columns,sigmas = 3):
    for col in columns:
        mean = df[col].mean()    
        std = df[col].std()
        thresholdUp = mean + sigmas*std
        thresholdDwn = mean - sigmas*std
        df[df[col]>thresholdUp] = thresholdUp
        df[df[col]<thresholdDwn] = thresholdDwn  

def random_oversampler(data, target, percentage):
    x_more = int((percentage * len(target) - sum(target)) // (1 - percentage))
    if x_more < 0 : raise ValueError("Percentage is met.")
    indices = np.argwhere(target).reshape(-1,)
    oversample = [np.random.choice(indices) for _ in range(x_more)]
    
    extra_data = [data[i] for i in oversample]
    extra_labels = [1 for i in oversample]
    new_data = np.concatenate([data,extra_data])
    new_labels = np.concatenate([target,extra_labels])
    return new_data,pd.Series(new_labels)
    
def inverse_signal(data,target, percentage):
    '''
    Applies rotation on x axis on the three sensor readings
    '''
    x_more = int((percentage * len(target) - sum(target)) // (1 - percentage))
    
    if x_more < 0 : raise ValueError("Percentage is met.")
    true_indices = np.argwhere(target).reshape(-1,)
    
    oversample = [np.random.choice(true_indices) for _ in range(x_more)]
    rotated_singal = []
    
    rotation_matrix = np.matrix([[1,0,0,0,0,0,0,0,0],
                                [0,-1,0,0,0,0,0,0,0],
                                [0,0,-1,0,0,0,0,0,0],
                                [0,0,0,1,0,0,0,0,0],
                                [0,0,0,0,-1,0,0,0,0],
                                [0,0,0,0,0,-1,0,0,0],
                                [0,0,0,0,0,0,1,0,0],
                                [0,0,0,0,0,0,0,-1,0]
                                ,[0,0,0,0,0,0,0,0,-1]])
    
    for item in oversample:
        rotated_singal.append(np.array(np.matrix(data[item])*rotation_matrix))
    
    extra_labels = [1 for i in oversample]
    rotated_singal = np.concatenate([data,rotated_singal])
    new_labels = np.concatenate([target,extra_labels])    
    return rotated_singal, pd.Series(new_labels)    
    
def clean_data(data, binary = False, exclude0 = False):
    data_cp = data.copy()
    data_cp.drop(['time'],axis = 1,inplace = True)
    if ((not binary) and (not exclude0)):
        labels = data_cp.Annotation
        data_cp.drop(['Annotation'],axis = 1,inplace = True)
    elif ((binary) and (not exclude0)):
        labels = data_cp.Annotation
        data_cp.drop(['Annotation'],axis = 1,inplace = True)
        labels.replace(1,0,inplace = True)
        labels.replace(2,1,inplace = True)
    elif ((not binary) and exclude0):
        data_cp.Annotation.replace(0,np.nan,inplace = True)
        data_cp.dropna(inplace = True)
        labels = data_cp.Annotation
        data_cp.drop(['Annotation'],axis = 1,inplace = True)
        labels.replace(1,0,inplace = True)
        labels.replace(2,1,inplace = True)
    else:
        print("Cannot use True value in both arguments")
    return labels.reset_index(drop = True),data_cp.reset_index(drop = True)

def windowed_data(data,labels,window = 256, step = 4):
    stepInit = 0
    start,stop = 0,window
    windowed_dataset,agg_labels = [],[]
    while stop<=data.shape[0]:
        windowed_dataset.append(data.iloc[start:stop,:].values)
        agg_labels.append(labels.iloc[start:stop].value_counts().index[0])
        start, stop = start+stepInit,stop+stepInit 
        stepInit = step  
    return np.array(windowed_dataset), np.array(agg_labels)
            
def parse_data(path, patient, binary = False, exclude0 = True, window = 256, step = 4): 
    cnt = 1
    for file in os.listdir(path):
        if file.startswith(patient):
            print("\n")
            print (file)
            df1 = pd.read_csv(path + file, sep = ' ',
                     names = ['time', 'acc_ankel_hor', 'acc_ankel_ver', 'acc_ankel_hl', 
                   'acc_thigh_hor', 'acc_thigh_ver', 'acc_thigh_hl','acc_trunk_hor',
                   'acc_trunk_ver', 'acc_trunk_hl', 'Annotation' ]) 
            labels,clean_df = clean_data(df1,binary,exclude0)
            df_window, labels_agg = windowed_data(clean_df, labels, window, step)
            if cnt==1: 
                df_all = df_window
                labels_all = labels_agg
                cnt = 2
            else:
                df_all = np.concatenate([df_all,df_window])
                labels_all = np.concatenate([labels_all,labels_agg])
    labels_s = pd.Series(labels_all)
    return df_all, labels_s        
        

def sensitivity(actuals, predictions):
    if sum(actuals) == 0: return 1
    c_m = confusion_matrix(actuals,predictions)
    sensitivity = c_m[0,0]/(c_m[0,0]+c_m[0,1])
    return sensitivity  

def specificity(actuals, predictions):
    if sum(actuals) == 0: return 1
    c_m = confusion_matrix(actuals,predictions)
    specificity = c_m[1,1]/(c_m[1,0]+c_m[1,1])
    return specificity        
        