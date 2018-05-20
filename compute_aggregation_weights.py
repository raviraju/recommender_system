"""Compute weights to aggregate multiple recommenders to build a hybrid recommender"""
import os
import argparse
from fnmatch import fnmatch
from pathlib import Path
from shutil import copyfile

import pandas as pd
import numpy as np

import json
from pprint import pprint

from sklearn.linear_model import LogisticRegression

from recommender.aggregate import Aggregator
from recommender.evaluation import PrecisionRecall

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
    
    max_weight = weights[0]
    for weight in weights:
        if weight > max_weight:
            max_weight = weight
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

    return weights

def get_aggregation_weights(scores_aggregation_file_path):
    """train logistic regression and return normalized model coefficients as weights"""
    print("Train logistic regression and fetch normalized model coefficients as weights...")
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

def generate_recommendations(dest_all_items_for_evaluation_file, aggregation_results, no_of_recs):
    with open(dest_all_items_for_evaluation_file, 'r') as json_file:
        all_items_for_evaluation = json.load(json_file)
        new_all_items_for_evaluation = dict()
        for user_id in all_items_for_evaluation:
            assume_interacted_items = all_items_for_evaluation[user_id]['assume_interacted_items']
            items_interacted = all_items_for_evaluation[user_id]['items_interacted']
            new_all_items_for_evaluation[user_id] = dict()
            new_all_items_for_evaluation[user_id]['assume_interacted_items'] = assume_interacted_items
            new_all_items_for_evaluation[user_id]['items_interacted'] = items_interacted
        #pprint(new_all_items_for_evaluation)
        aggregation_results['user_id'] = aggregation_results['user_id'].astype(str)
        aggregation_results['item_id'] = aggregation_results['item_id'].astype(str)

        for user_id in new_all_items_for_evaluation:
            items_to_recommend = []
            assume_interacted_items = new_all_items_for_evaluation[user_id]['assume_interacted_items']
            user_agg_results = aggregation_results[aggregation_results['user_id'] == user_id]

            recommended_items_dict = dict()
            if user_agg_results is not None:
                rank = 1
                for _, res in user_agg_results.iterrows():
                    item_id = res['item_id']
                    user_id = res['user_id']
                    score = res['weighted_avg']
                    if item_id in assume_interacted_items:
                        #print("Skipping : ", item_id)
                        continue
                    if rank > no_of_recs:#limit no of recommendations
                        break
                    item_dict = {
                        'user_id' : user_id,
                        'item_id' : item_id,
                        'score' : score,
                        'rank' : rank
                    }
                    #print(user_id, item_id, score, rank)
                    items_to_recommend.append(item_dict)
                    recommended_items_dict[item_id] = {'score' : score, 'rank' : rank}
                    rank += 1
            res_df = pd.DataFrame(items_to_recommend)
            #print(res_df)
            recommended_items = list(res_df['item_id'].values)
            new_all_items_for_evaluation[user_id]['items_recommended'] = recommended_items

            items_interacted_set = set(new_all_items_for_evaluation[user_id]['items_interacted'])
            items_recommended_set = set(recommended_items)
            correct_recommendations = items_interacted_set & items_recommended_set
            no_of_correct_recommendations = len(correct_recommendations)
            new_all_items_for_evaluation[user_id]['no_of_correct_recommendations'] = no_of_correct_recommendations
            new_all_items_for_evaluation[user_id]['correct_recommendations'] = list(correct_recommendations)

            new_all_items_for_evaluation[user_id]['items_recommended_score'] = recommended_items_dict
            #pprint(new_all_items_for_evaluation[user_id])
            #input()
    return new_all_items_for_evaluation

def analyse(use_case_dir, data_file, item_id_col, no_of_recs, no_of_recs_to_eval):
    """analyse single experiment for recommender using evalution_results.json"""
    configs = []
    i = 1
    data = pd.read_csv(data_file)
    all_items = set(data[item_id_col].unique())
    #print(len(all_items))
    for root, _, files in os.walk(use_case_dir, topdown=False):
        for name in files:
            if fnmatch(name, 'all_scores_aggregation.csv'):
                src_scores_aggregation_file_path = (os.path.join(root, name))
                kfold_experiments_dir = Path(src_scores_aggregation_file_path).parent
                model_dir = Path(kfold_experiments_dir).parent
                model_parent_dir = Path(model_dir).parent
                if ('hybrid' in model_dir.name):
                #if ('all_recommenders' in model_dir.name):
                    print("\nComputing Aggregation Weights for Model", i)
                    model_name = model_parent_dir.name + '/' + model_dir.name.replace('equal', 'logit')
                    print(model_name)
                    logit_kfold_dir = use_case_dir + model_name + '/kfold_experiments'
                    if not os.path.exists(logit_kfold_dir):
                        os.makedirs(logit_kfold_dir)
                    src_all_items_for_evaluation_file = os.path.join(kfold_experiments_dir,
                                                                     'all_items_for_evaluation.json')
                    dest_all_items_for_evaluation_file = os.path.join(logit_kfold_dir,
                                                                     'all_items_for_evaluation.json')
                    #print(src_all_items_for_evaluation_file, dest_all_items_for_evaluation_file)
                    copyfile(src_all_items_for_evaluation_file, dest_all_items_for_evaluation_file)
                    #print(dest_all_items_for_evaluation_file)

                    dest_scores_aggregation_file_path = os.path.join(logit_kfold_dir,
                                                                     'all_scores_aggregation.csv')
                    #print(src_scores_aggregation_file_path, dest_scores_aggregation_file_path)
                    copyfile(src_scores_aggregation_file_path, dest_scores_aggregation_file_path)
                    #print(dest_scores_aggregation_file_path)

                    i+=1
                    
                    config = dict()
                    config['model_dir_name'] = model_name
                    print("Loading {} ...".format(dest_scores_aggregation_file_path))
                    aggregation_weights = get_aggregation_weights(dest_scores_aggregation_file_path)
                    aggregation_df = pd.read_csv(dest_scores_aggregation_file_path)
                    #print(aggregation_df.head())
                    res_aggregator = Aggregator(aggregation_df)
                    pprint(aggregation_weights)
                    aggregation_results = res_aggregator.weighted_avg(aggregation_weights)
                    #print(aggregation_results.head())
                    print("Storing computed weighted avg in {}...".format(dest_scores_aggregation_file_path))
                    aggregation_results.to_csv(dest_scores_aggregation_file_path, index=False)

                    print("Generating Recommendations...")
                    all_items_for_evaluation = generate_recommendations(dest_all_items_for_evaluation_file,
                                                                        aggregation_results, no_of_recs)
                    with open(dest_all_items_for_evaluation_file, 'w') as json_file:
                        json.dump(all_items_for_evaluation, fp=json_file, indent=4)

                    precision_recall_intf = PrecisionRecall()
                    evaluation_results = precision_recall_intf.compute_precision_recall(no_of_recs_to_eval,
                                                                                        all_items_for_evaluation,
                                                                                        all_items)
                    pprint(evaluation_results)
                    print("Evaluating...")
                    evaluation_file_path = os.path.join(logit_kfold_dir,
                                                        'kfold_evaluation.json')
                    with open(evaluation_file_path, 'w') as json_file:
                        json.dump(evaluation_results, fp=json_file, indent=4)

                    config['recommenders'] = aggregation_weights
                    configs.append(config)

    weights_config_file = os.path.join(use_case_dir, 'weights_config.json')
    with open(weights_config_file, 'w') as json_file:
        json.dump(configs, fp=json_file, indent=4)
    print("Aggregation weights config are present in  : ", weights_config_file)

                
def main():
    """load equal weighted results of hybrid recommenders and compute new weights"""
    parser = argparse.ArgumentParser(description="Compute Weights for Hybrid Recommender Models")
    parser.add_argument("use_case", help="Use Case Recommender Model Directory")
    parser.add_argument("data", help="All Data file")
    args = parser.parse_args()

    item_id_col = 'book_code'
    no_of_recs = 150
    no_of_recs_to_eval = [5, 6, 7, 8, 9, 10]

    if args.use_case:
        print("Analysing results of ", args.use_case)
        analyse(args.use_case, args.data, item_id_col, no_of_recs, no_of_recs_to_eval)
    else:
        print("Specify use case recommender model directory to analyse")

if __name__ == '__main__':
    main()                