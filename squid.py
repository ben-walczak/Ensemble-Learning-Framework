# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 22:23:55 2019

@author: Benjamin
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

def get_distributions(df):

    distributions = {
        'unscaled_data':df,
        'standard_scaling':
            StandardScaler().fit_transform(df),
        'min-max_scaling':
            MinMaxScaler().fit_transform(df),
        'max-abs_scaling':
            MaxAbsScaler().fit_transform(df),
        'robust_scaling':
            RobustScaler(quantile_range=(25, 75)).fit_transform(df),
        'uniform_pdf':
            QuantileTransformer(output_distribution='uniform')
            .fit_transform(df),
        'gaussian_pdf':
            QuantileTransformer(output_distribution='normal')
            .fit_transform(df),
        'L2_normalizing':
            Normalizer().fit_transform(df)
    }
    
    return distributions

from sklearn.model_selection import cross_val_score

def get_layer_transforms(models, df, y):
    
    distributions = get_distributions(df)

    model_transformations = {}
    
    # find best transformation
    for model_name, model in models.items():
        if 'cluster' in str(type(model)): 
            model_transformations[model_name] = distributions['unscaled_data']
            continue
        
        distribution_scores = {}
        
        for distribution_name, distribution in distributions.items():
            rmse_scores = cross_val_score(model, distribution, y, cv=5, \
                                      scoring = 'neg_mean_squared_error')
            distribution_scores[distribution_name] = \
                                      rmse_scores.mean()
        
        # THIS IS UGLY (but works...)
        best_dscore = max(distribution_scores.values())
        for dname, dscore in distribution_scores.items():
            if dscore == best_dscore:
                distribution_name = dname
        
        model_transformations[model_name] = distributions[distribution_name]

    return model_transformations

def fit_predict_all_layers(layers, df, y):
    
    for index, layer in enumerate(layers):
        print('Layer: '+str(index))
        
        print('Getting input')
        # get input
        if index == 0:
            layer_input = df
        else:
            layer_input = layers[index-1]['outputs']
        
        print('Getting model transforms')
        # get model transformations
        model_transformations =  get_layer_transforms\
            (layer['models'], layer_input, y)

        print('Fitting models')
        # fit models
        for model_name, model in layer['models'].items():
            layers[index]['models'][model_name]\
                .fit(model_transformations[model_name],y)
            
        outputs = {}
        print('Predict outputs of layer')
        # predict outputs of layer
        for model_name, model in layer['models'].items():
            
            predictions = model\
                .predict(model_transformations[model_name])
            
            # if value is encapsulated by array un-array it
            if type(predictions[0]) == np.ndarray:
                predictions = predictions.tolist()
                for i in range(len(predictions)):
                    predictions[i] = predictions[i][0]
                predictions = np.asarray(predictions)
                
            outputs[model_name] = predictions
            
        layers[index]['outputs'] = pd.DataFrame.from_dict(outputs)
    
    return layers