# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:04:54 2022

@author: ddimoudis
"""
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import preprocessing as pre
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import tqdm
from keras.models import Sequential
from keras.layers import Dense,GlobalMaxPooling1D, Dropout,LSTM,Conv1D,AveragePooling1D,Flatten,BatchNormalization,MaxPooling1D,GRU
import tensorflow as tf
import InceptionTs as incTS
import RTrans


def CNNlstm_creation(input_shape,layer1=64, layer2=32, layer3=16, dropout=0.1, lr = 0.001):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv1D(50, kernel_size=7, activation='relu', input_shape=input_shape, strides = 3 ))
    model.add(Conv1D(50, kernel_size=5, activation='relu',strides =3) )
    model.add(Conv1D(50, kernel_size=3, activation='relu',strides =3) )
    model.add(Dropout(dropout+0.2))
    model.add(LSTM(layer1, activation='tanh',return_sequences=True,name = 'GRU1'))
    model.add(LSTM(layer2, activation='tanh',name =  'GRU2',return_sequences=True))
    model.add(LSTM(layer2, activation='tanh',name =  'GRU3',return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout))
    model.add(Dense(64,activation = 'relu', name = 'Dense02'))
    model.add(Dropout(dropout))
    model.add(Dense(32,activation = 'relu', name = 'Dense05'))
    model.add(Dropout(dropout))
    model.add(Dense(8,activation = 'relu', name = 'Dense03'))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation = 'sigmoid', name = 'Dense04'))
    
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss = tf.keras.losses.BinaryCrossentropy())
    return model

data = ["S01","S02","S03","S05","S06","S07","S08","S09"]


sens = []
spec = []
f1 = []

indv_data_results = {}

for dataset in tqdm.tqdm(data):
    df_all, labels_s = pre.parse_data('dataset_fog_release\\dataset\\',dataset)
    
    #df_all, labels_s = pre.inverse_signal(df_all, labels_s.values, 0.3)
    
    
    # End of windowing and data operations.
    kf = StratifiedKFold(n_splits=10,shuffle = True)
    
    ind_sens = []
    ind_spec = []
    ind_f1 = []

    
    for train_index, test_index in kf.split(df_all,labels_s):
        X_train, X_test = df_all[train_index,:], df_all[test_index,:]
        y_train, y_test = labels_s.loc[train_index], labels_s.loc[test_index]
        
        #X_train, y_train = pre.inverse_signal(X_train, y_train.values, 0.3)
        
        shape = X_train[1].shape
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 50,
                                                          restore_best_weights = True)
        rnn_clf = incTS.inception_model(shape, 3, [64,32,16,8], 0.0005, 0.2, [100,64],residuals = True)
        #-------------------Fitting Stage----------------------------
        rnn_clf.fit(X_train, y_train, epochs = 1000,verbose = 0,batch_size = 1024,callbacks = [early_stopping],
                    validation_split = 0.1)
    #-------------------Classification Report-----------------------
        rnn_preds = rnn_clf.predict(X_test)
        rnn_preds = np.where(rnn_preds.reshape(-1) > 0.5, 1, 0) 
        ind_sens.append(pre.sensitivity(y_test, rnn_preds))
        ind_spec.append(pre.specificity(y_test, rnn_preds))
        ind_f1.append(f1_score(y_test, rnn_preds,zero_division= 1.))
    
    mean_sens = np.mean(ind_sens)
    mean_spec = np.mean(ind_spec)
    mean_f1 = np.mean(ind_f1)
    
    indv_data_results[dataset] = {'sensitivity':mean_sens ,'specifity': mean_spec,
                                  'f1': mean_f1}
    
    sens.append(mean_sens)
    spec.append(mean_spec)
    f1.append(mean_f1)
    
    
print( "\n")
print( 'Mean rNN sens: '+ str(np.mean(sens)))
print( 'Mean rNN spec: '+ str(np.mean(spec)))
print( 'Mean rNN f1: '+ str(np.mean(f1)))

res_df = pd.DataFrame.from_dict(indv_data_results)
res_df.to_csv("cnnlstm_r_inv_30.csv")