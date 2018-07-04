import os
import argparse
import pickle

import json
import math
import pandas as pd
import numpy as np

from timeit import default_timer
from pprint import pprint

from sklearn.metrics import mean_squared_error


from surprise import Dataset
from surprise import Reader
from surprise import accuracy

from surprise.model_selection import PredefinedKFold
from surprise import dump

from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from collections import defaultdict

def get_hist_plots(prediction_df):
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4))
    figure.tight_layout()
    prediction_df['r_ui'].plot(kind='hist', title='Actual Rating', ax=ax1)
    prediction_df['est'].plot(kind='hist', title='Predict Rating', ax=ax2)
    prediction_df['err'].plot(kind='hist', title='Error', ax=ax3)
    return plt

def avg(numbers):
    """return average of list of numbers"""
    return sum(numbers)/len(numbers)

def convert_sec(no_of_secs):
    """return no_of_secs to min or hrs string"""
    if no_of_secs < 60:
        return "Time Taken : {:06.4f}    sec".format(no_of_secs)
    elif no_of_secs < 3600:
        return "Time Taken : {:06.4f}    min".format(no_of_secs/60)
    else:
        return "Time Taken : {:06.4f}    hr".format(no_of_secs/3600)

def get_testset_stats(testset):
    #print("Testing Data")
    test_users = []
    test_items = []
    test_ratings = []
    for (user_id, item_id, rating) in testset:
        #print((user_id, item_id, rating))
        test_users.append(user_id)
        test_items.append(item_id)
        test_ratings.append(rating)
    test_users_set = set(test_users)
    test_items_set = set(test_items)
    #print("No of users : {}, users_set : {}".format(len(test_users), len(test_users_set)))
    #print("No of items : {}, items_set : {}".format(len(test_items), len(test_items_set)))
    #print("No of ratings : {}".format(len(test_ratings)))
    return len(test_users_set),  len(test_items_set), len(test_ratings)

def get_no_of_items_rated_by_user(uid, trainset):
    """Return the number of items rated by given user, uid: The raw id of the user."""
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError:  # user was not part of the trainset
        return 0
    
def get_no_of_users_rated_item(iid, trainset):
    """Return the number of users that have rated given item, iid: The raw id of the item."""
    try:
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:  # item was not part of the trainset
        return 0

def filter_anti_testset(all_testset, testset):
    test_users = []
    known_ratings = defaultdict(int)
    for (user_id, item_id, rating) in testset:
        #print((user_id, item_id, rating))
        test_users.append(user_id)
        known_ratings[(user_id, item_id)] = rating
    test_users_set = set(test_users)
    
    req_ratings = []
    for (user_id, item_id, rating) in all_testset:
        if user_id in test_users_set:
            rating = known_ratings[(user_id, item_id)]
            req_ratings.append((user_id, item_id, rating))
    return req_ratings
    
def main():
    parser = argparse.ArgumentParser(description="Generate Recommendations")
    parser.add_argument("configs", help="config of algorithms used to generate recommendations")
    args = parser.parse_args()
    
    pickle_in = open(args.configs,"rb")
    configs = pickle.load(pickle_in)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.expanduser('train_test_data/kfolds_split/')    
    
    results_dir = os.path.join(current_dir, 'model_validation')
    train_file = files_dir + '%d_training_for_validation_uir_data.csv'
    test_file = files_dir + '%d_validation_uir_data.csv'

    #results_dir = os.path.join(current_dir, 'model_testing')
    #train_file = files_dir + '%d_training_all_uir_data.csv'
    #test_file = files_dir + '%d_testing_all_uir_data.csv'
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # folds_files is a list of tuples containing file paths:
    # [(1_train_data.csv, 1_test_data.csv), (2_train_data.csv, 2_test_data.csv), ... (10_train_data.csv, 10_test_data.csv)]
    no_of_kfolds = 10
    kfolds = range(1, no_of_kfolds+1)        
    folds_files = [(train_file % i, test_file % i) for i in kfolds]
    #folds_files

    reader = Reader(rating_scale=(1.0, 3.0), sep=',', skip_lines=1)
    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()

    '''
    for trainset, testset in pkf.split(data):
        print()
        print("trainset : ", trainset.n_users, trainset.n_items, trainset.n_ratings)
        testset_n_users, testset_n_items, testset_n_ratings = get_testset_stats(testset)
        print("testset  : ", testset_n_users, testset_n_items, testset_n_ratings)

        #all_testset = trainset.build_anti_testset()
        #req_testset = filter_anti_testset(all_testset, testset)
        #testset_n_users, testset_n_items, testset_n_ratings = get_testset_stats(req_testset)
        #print("testset  : ", testset_n_users, testset_n_items, testset_n_ratings)
        #input()
    '''

    summary_results_json_file = os.path.join(results_dir, 'summary_results.json')
    summary_results_csv_file = os.path.join(results_dir, 'summary_results.csv')
    if os.path.isfile(summary_results_json_file):
        print("Loading existing summary results")
        with open(summary_results_json_file, 'r') as json_file:
            summary = json.load(json_file)
        summary_df = pd.read_csv(summary_results_csv_file)
        summaries = list(summary_df.T.to_dict().values())
    else:
        summary = dict()
        summary_df = pd.DataFrame()
        summaries = []
    overall_start_time = default_timer()
    for config in configs:
        algo = config['algo']

        #algo_name = algo.__class__.__name__
        algo_name = config['name']
        print('*'*80)
        print("Generating Recommendations using ", algo_name)
        start_time = default_timer()
        
        summary[algo_name] = dict()
        kfolds_rmse_all_predictions = []
        kfolds_rmse_valid_predictions = []
        kfolds_rmse_default_predictions = []
        
        kfolds_no_of_all_predictions = []
        kfolds_no_of_valid_predictions = []
        kfolds_no_of_default_predictions = []
        
        model_dir = os.path.join(results_dir, algo_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)        
        
        all_predictions_df = pd.DataFrame()
        valid_all_predictions_df = pd.DataFrame()
        default_all_predictions_df = pd.DataFrame()
        
        kfold_exp = 0
        for trainset, testset in pkf.split(data):
            kfold_exp += 1
            kfold_model_dir = os.path.join(model_dir, 'kfold_experiments',
                                           'kfold_exp_' + str(kfold_exp))
            if not os.path.exists(kfold_model_dir):
                os.makedirs(kfold_model_dir)
            
            print(kfold_model_dir)
            #print("trainset : ", trainset.n_users, trainset.n_items, trainset.n_ratings)
            #testset_n_users, testset_n_items, testset_n_ratings = get_testset_stats(testset)
            #print("testset  : ", testset_n_users, testset_n_items, testset_n_ratings)
            # train and test algorithm.
            algo.fit(trainset)
            predictions = algo.test(testset)
                       
            predictions_df = pd.DataFrame(predictions, 
                                          columns=['uid', 'iid', 'r_ui', 'est', 'details'])            
            predictions_df['err'] = abs(predictions_df['est'] - predictions_df['r_ui'])
            predictions_df['no_of_items_rated_by_user'] = predictions_df['uid'].apply(get_no_of_items_rated_by_user, 
                                                                                      args=(trainset,))
            predictions_df['no_of_users_rated_item'] = predictions_df['iid'].apply(get_no_of_users_rated_item,
                                                                                   args=(trainset,))
            
            predictions_file = os.path.join(kfold_model_dir, 'predictions.csv')
            predictions_df.to_csv(predictions_file, index=False)
            all_predictions_df = all_predictions_df.append(predictions_df)
            
            valid_predictions = []
            default_predictions = []
            for prediction in predictions:
                info = dict()
                info['uid'] = prediction.uid
                info['iid'] = prediction.iid
                info['r_ui'] = prediction.r_ui
                info['est'] = prediction.est

                details = prediction.details
                if details['was_impossible']:
                    info['reason'] = details['reason']
                    default_predictions.append(info)
                else:
                    if 'actual_k' in details:
                        info['actual_k'] = details['actual_k']
                    valid_predictions.append(info)
            valid_predictions_df = pd.DataFrame(valid_predictions)
            valid_all_predictions_df = valid_all_predictions_df.append(valid_predictions_df)
            
            default_predictions_df = pd.DataFrame(default_predictions)
            default_all_predictions_df = default_all_predictions_df.append(default_predictions_df)
            
            valid_predictions_file = os.path.join(kfold_model_dir, 'valid_predictions.csv')
            valid_predictions_df.to_csv(valid_predictions_file, index=False)
            
            default_predictions_file = os.path.join(kfold_model_dir, 'default_predictions.csv')
            default_predictions_df.to_csv(default_predictions_file, index=False)

            mse = mean_squared_error(predictions_df['r_ui'], predictions_df['est'])
            kfold_rmse_all_predictions = round(math.sqrt(mse), 4)
            #print("kfold_rmse_all_predictions : ", kfold_rmse_all_predictions)            
            kfolds_rmse_all_predictions.append(kfold_rmse_all_predictions)

            kfold_rmse_valid_predictions = 0.0
            if not valid_predictions_df.empty:
                mse = mean_squared_error(valid_predictions_df['r_ui'], valid_predictions_df['est'])
                kfold_rmse_valid_predictions = round(math.sqrt(mse), 4)
            #print("kfold_rmse_valid_predictions : ", kfold_rmse_valid_predictions)
            kfolds_rmse_valid_predictions.append(kfold_rmse_valid_predictions)
            
            kfold_rmse_default_predictions = 0.0
            if not default_predictions_df.empty:
                mse = mean_squared_error(default_predictions_df['r_ui'], default_predictions_df['est'])
                kfold_rmse_default_predictions = round(math.sqrt(mse), 4)
            #print("kfold_rmse_default_predictions : ", kfold_rmse_default_predictions)
            kfolds_rmse_default_predictions.append(kfold_rmse_default_predictions)
                    
            #print(len(predictions_df))            
            kfolds_no_of_all_predictions.append(len(predictions_df))
            kfolds_no_of_valid_predictions.append(len(valid_predictions_df))
            kfolds_no_of_default_predictions.append(len(default_predictions_df))
            
        summary[algo_name]['mean_rmse_all_predictions']     = round(avg(kfolds_rmse_all_predictions), 4)
        summary[algo_name]['mean_rmse_valid_predictions']   = round(avg(kfolds_rmse_valid_predictions), 4)
        summary[algo_name]['mean_rmse_default_predictions'] = round(avg(kfolds_rmse_default_predictions), 4)
        
        summary[algo_name]['no_of_all_predictions']     = round(sum(kfolds_no_of_all_predictions), 4)
        summary[algo_name]['no_of_valid_predictions']   = round(sum(kfolds_no_of_valid_predictions), 4)
        summary[algo_name]['no_of_default_predictions'] = round(sum(kfolds_no_of_default_predictions), 4)
        
        users_rated_less_than_10_items = all_predictions_df[all_predictions_df['no_of_items_rated_by_user'] <= 10]
        users_rated_more_than_10_items = all_predictions_df[all_predictions_df['no_of_items_rated_by_user'] > 10]
        summary[algo_name]['avg_error_of_users_rated_less_than_10_items'] = users_rated_less_than_10_items['err'].mean()
        summary[algo_name]['avg_error_of_users_rated_more_than_10_items'] = users_rated_more_than_10_items['err'].mean()
        summary[algo_name]['algo_name'] = algo_name
        end_time = default_timer()
        time_taken = convert_sec(end_time - start_time)
        summary[algo_name]['time_taken'] = time_taken
        
        summaries.append(summary[algo_name])        
        
        plt = get_hist_plots(all_predictions_df)
        plt.subplots_adjust(bottom=0.1, left=0.2, wspace = 0.4, top=0.9)
        plt.savefig(os.path.join(model_dir, 'hist_ratings.png'))
        
        pprint(summary[algo_name])

        all_predictions_df.to_csv(os.path.join(model_dir, 'all_predictions.csv'), index=False)        
        default_all_predictions_df.to_csv(os.path.join(model_dir, 'default_all_predictions.csv'), index=False)
        valid_all_predictions_df.to_csv(os.path.join(model_dir, 'valid_all_predictions.csv'), index=False)
        
        best_all_predictions_df = all_predictions_df.sort_values(by='err')[:10]
        best_all_predictions_df.to_csv(os.path.join(model_dir, 'best_all_predictions.csv'), index=False)
        worst_all_predictions_df = all_predictions_df.sort_values(by='err')[-10:]
        worst_all_predictions_df.to_csv(os.path.join(model_dir, 'worst_all_predictions.csv'), index=False)

    overall_end_time = default_timer()
    overall_time_taken = convert_sec(overall_end_time - overall_start_time)
    summary['overall_time_taken'] = overall_time_taken
    
    #pprint(summary)

    with open(summary_results_json_file, 'w') as json_file:
        json.dump(summary, fp=json_file, indent=4)
    print("Summary Results of all algorithms in : ", summary_results_json_file)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(summary_results_csv_file, index=False)

if __name__ == '__main__':
    main()
