# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:49:41 2022

@author: ddimoudis
"""

import keras 


def Inception_module(inputs,filters):

    ### Bottleneck Layer
    bottleneck = keras.layers.Conv1D(inputs.shape[1],kernel_size=1, padding='same', activation='relu',
                                     use_bias=False)(inputs)
    #MaxPooling layer and bottlneck 2
    maxPool = keras.layers.MaxPool1D(3, strides=(1), padding='same')(inputs)
    bottleneck_1 = keras.layers.Conv1D(filters=inputs.shape[1], kernel_size=1,
                                     padding='same', activation='relu', use_bias=False)(maxPool)
    
    # 3 Filter layers
    #layer_10 = keras.layers.Conv1D(filters[0],7, padding='same', activation='relu',use_bias=False)(bottleneck)
    layer_20 = keras.layers.Conv1D(filters[0],5, padding='same', activation='relu',use_bias=False)(bottleneck)
    layer_40 = keras.layers.Conv1D(filters[1],3, padding='same', activation='relu',use_bias=False)(bottleneck)


    x = keras.layers.concatenate([layer_20, layer_40, bottleneck_1], axis = 2)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    return x

def residual_connection(prev_output, initial_input):
    residuals = keras.layers.Conv1D(int(prev_output.shape[-1]), 1, padding = 'same', use_bias=False)(initial_input)
    residuals = keras.layers.BatchNormalization()(residuals)
    
    out = keras.layers.Add()([residuals,prev_output])
    out = keras.layers.Activation('relu')(out)
    return out

def build_model(inputShape, InceptionLayers, MLPlayers, dropoutRate = 0.2, filters = [100,64],residuals = True):
    inputs = keras.Input(shape=inputShape)
    x = inputs
    init_inputs = inputs
    #x = keras.layers.BatchNormalization()(x)
    for i in range(InceptionLayers):
        x = Inception_module(x, filters)
        
        if (i %3 == 2) and (residuals == True):
            res = residual_connection(x,init_inputs)
            init_inputs = res
            
    x = keras.layers.Dropout(0.5)(x)        

    #x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    for size in MLPlayers:
        x = keras.layers.Dense(size, activation = 'relu')(x)
        x = keras.layers.Dropout(dropoutRate)(x)
    output = keras.layers.Dense(1, activation = 'sigmoid')(x)
    return keras.Model(inputs = inputs,outputs = output)

def inception_model(inputShape, InceptionLayers, MLPlayers, lr, dout, inFilters, residuals):
    inc_model = build_model(inputShape, InceptionLayers, MLPlayers, dout,inFilters)
    opt = keras.optimizers.Adam(learning_rate=lr)
    inc_model.compile(optimizer=opt, loss = keras.losses.BinaryCrossentropy())
    return inc_model
    
    
    
    