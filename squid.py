# -*- coding: utf-8 -*-
"""
Squid

This module streamlines the process of building and testing
multiple layers of models. This module works in conjunction
with scikitlearn's models, but has the ability to work with
the outputs of other models (as long as they are converted
to the type pandas Dataframe).

Although a squid has many arms with different capabilities,
all arms of the squid have a collective goal...

Similarly, this module attempts to combine multiple models
with the collective goal of predicting one set of output...

TODO:
    [] finish add_output method
    [] test all methods
    [X] implement transforms within layers object
    [] work layers[index][transforms] into all methods
    [] method to add custom transforms into get_distributions
    [] add other standard distributions to get_distributions
    [] method to fit_predict one layer at a time
    [] option to avoid finding distributions automatically
"""

__version__ = '0.1'
__author__ = 'Ben Walczak'

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

from sklearn.model_selection import cross_val_score

class layers:
    layers = []
    
    def __init__(self, number_of_layers):
        for i in range(number_of_layers):
            self.layers.append({'models':{},'transforms':{}, 'outputs':None})
            
    def add_model(self, layer_index, model, model_name = None):
        if model_name == None:
            model_name = 'model_'+str(len(self.layers[layer_index]['models']))
        self.layers[layer_index]['models'][model_name] = model
        
    #def add_output(self, layer_index, output)
    #if output exists output len must be = to current output
    #otherwise instantiate output with given output

    def __get_distributions(self, df):
        
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
    
    
    
    def __get_layer_transforms(self, index, df, y):
        
        distributions = self.__get_distributions(df)
        models = self.layers[index]['models']
    
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
            self.layers[index]['transforms'][model_name] = model_transformations[model_name] 
            print(self.layers[index]['transforms'])
    
        return model_transformations
    
    def fit_predict_all_layers(self, X, y):
        
        for index, layer in enumerate(layers):
            print('Layer: '+str(index))
            
            print('Getting input')
            # get input
            if index == 0:
                layer_input = X
            else:
                layer_input = layers[index-1]['outputs']
                
            # convert input to float 64
            for column in layer_input.columns:
                layer_input[column] = layer_input[column].astype(np.float64)
            
            print('Getting model transforms')
            # get model transformations
            model_transformations =  self.__get_layer_transforms(
                    index, layer_input, y)
    
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
                
            self.layers[index]['outputs'] = pd.DataFrame.from_dict(outputs)