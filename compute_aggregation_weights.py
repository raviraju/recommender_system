"""Compute weights to aggregate multiple recommenders to build a hybrid recommender"""
import os
import argparse
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd
import numpy as np

import json
#from pprint import pprint

from sklearn.linear_model import LogisticRegression

def check_sanity(weights):
    """perform sanity checks on aggregation weights"""

    max_weight = weights[0]
    for weight in weights:
        if weight > max_weight:
            max_weight = weight
            
    """Ensure each weight is non-zero"""
    if 0.0 in weights:
        print("One of the weights is zero")
        print(weights)        
        new_weights = []
        for weight in weights:            
            if weight == 0:
                new_weight = weight + 0.01
            elif weight == max_weight:
                new_weight = weight - 0.01
            else:
                new_weight = weight
            new_weights.append(round(new_weight, 2))
        weights = new_weights
        print("corrected weights : ")
        print(weights)
        #input()
    
    """Sum of Weights to be 1.0"""
    sum_of_weights = round(np.sum(weights), 2)
    #print("sum_of_weights : ", sum_of_weights)    
    if sum_of_weights != 1.0:
        print("aggregation weights {}, do not sum to 1.0".format(sum_of_weights))        
        #remove the delta from highest weight
        difference = sum_of_weights - 1.0
        
        new_weights = []
        for weight in weights:
            if weight == max_weight:
                new_weight = weight - difference
            else:
                new_weight = weight
            new_weights.append(round(new_weight, 2))
        weights = new_weights
        print(weights)
        print("corrected weights sum to : ", round(np.sum(weights), 2))
        #input()

    return weights

def get_aggregation_weights(scores_aggregation_file_path):
    """train logistic regression and return normalized model coefficients as weights"""
    df = pd.read_csv(scores_aggregation_file_path)
    df.rename(index=str, columns={'Unnamed: 0':'id'}, inplace=True)
    #print(df.head())
    
    recommender_features = []
    for col in df.columns:
        if 'Recommender' in col:
            recommender_features.append(col)
           
    #print(recommender_features)
    target = 'watched'
    X = df[recommender_features]
    y = df[target]
    
    logreg = LogisticRegression(class_weight='balanced')
    logreg.fit(X, y)
    coefficients = logreg.coef_[0]
    #print("coefficients : ", coefficients)
    weights = np.round(coefficients/np.sum(coefficients), 2)
    #print("weights : ",  weights)
    weights = check_sanity(weights)
    
    aggregation_weights = dict()
    for rec, weight in zip(recommender_features, weights):
        #print(rec, weight)
        aggregation_weights[rec] = weight
    #pprint(aggregation_weights)
    return aggregation_weights

def analyse(use_case_dir):
    """analyse single experiment for recommender using evalution_results.json"""
    configs = []
    i = 1
    for root, _, files in os.walk(use_case_dir, topdown=False):
        for name in files:
            if fnmatch(name, 'all_scores_aggregation.csv'):
                scores_aggregation_file_path = (os.path.join(root, name))
                kfold_experiments_dir = Path(scores_aggregation_file_path).parent
                model_dir = Path(kfold_experiments_dir).parent
                model_parent_dir = Path(model_dir).parent
                if ('hybrid' in model_dir.name):
                #if ('all_recommenders' in model_dir.name):
                    print("Computing Aggregation Weights for Model", i)
                    model_name = model_parent_dir.name + '/' + model_dir.name.replace('equal', 'logit')
                    print(model_name)
                    #print(scores_aggregation_file_path)
                    print()
                    i+=1
                    
                    config = dict()
                    config['model_dir_name'] = model_name
                    config['recommenders'] = get_aggregation_weights(scores_aggregation_file_path)
                    configs.append(config)
    weights_config_file = os.path.join(use_case_dir, 'weights_config.json')
    with open(weights_config_file, 'w') as json_file:
        json.dump(configs, fp=json_file, indent=4)
    print("Aggregation weights config are present in  : ", weights_config_file)

                
def main():
    """load equal weighted results of hybrid recommenders and compute new weights"""
    parser = argparse.ArgumentParser(description="Compute Weights for Hybrid Recommender Models")
    parser.add_argument("use_case", help="Use Case Recommender Model Directory")
    args = parser.parse_args()
    if args.use_case:
        print("Analysing results of ", args.use_case)
        analyse(args.use_case)
    else:
        print("Specify use case recommender model directory to analyse")

if __name__ == '__main__':
    main()                