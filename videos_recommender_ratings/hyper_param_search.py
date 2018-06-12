import os
import argparse
import pickle

import json
import pandas as pd

from timeit import default_timer
from pprint import pprint

from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import GridSearchCV
from surprise import dump

from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline

def convert_sec(no_of_secs):
    """return no_of_secs to min or hrs string"""
    if no_of_secs < 60:
        return "Time Taken : {:06.4f}    sec".format(no_of_secs)
    elif no_of_secs < 3600:
        return "Time Taken : {:06.4f}    min".format(no_of_secs/60)
    else:
        return "Time Taken : {:06.4f}    hr".format(no_of_secs/3600)

def main():
    parser = argparse.ArgumentParser(description="Hyper Parameters Search Experiments")
    parser.add_argument("experiments", help="experiment of algorithms used to search for hyper params")
    args = parser.parse_args()    
    pickle_in = open(args.experiments,"rb")
    experiments = pickle.load(pickle_in)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.expanduser('train_test_data/kfolds_split/')    
    # folds_files is a list of tuples containing file paths:
    # [(1_train_data.csv, 1_test_data.csv), ... (10_train_data.csv, 10_test_data.csv)]
    no_of_kfolds = 10
    kfolds = range(1, no_of_kfolds+1)
    train_file = files_dir + '%d_training_for_validation_uir_data.csv'
    test_file = files_dir + '%d_validation_uir_data.csv'
    folds_files = [(train_file % i, test_file % i) for i in kfolds]
    reader = Reader(rating_scale=(1.0, 3.0), sep=',', skip_lines=1)
    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()

    for experiment in experiments:
        results_dir_name = experiment['results_dir_name']
        algos = experiment['algos']
        param_grid = experiment['param_grid']
        
        results_dir = os.path.join(current_dir, results_dir_name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        overall_start_time = default_timer()
        grid_search_results = dict()
        for algo in algos:        
            algo_name = algo.__name__
            grid_search_results[algo_name] = dict()

            print("Performing Grid Search on ", algo_name)
            start_time = default_timer()
            gs = GridSearchCV(algo, param_grid, 
                              measures=['rmse'],#['rmse', 'mae'], 
                              cv=pkf, 
                              return_train_measures=False, n_jobs=-1)

            gs.fit(data)
            end_time = default_timer()
            time_taken = convert_sec(end_time - start_time)
            grid_search_results[algo_name]['time_taken'] = time_taken

            best_rmse = gs.best_score['rmse']
            grid_search_results[algo_name]['best_rmse'] = best_rmse
            print("best RMSE score on test: ", best_rmse)

            print("combination of parameters that gave the best RMSE score")
            best_params = gs.best_params['rmse']
            grid_search_results[algo_name]['best_params'] = best_params
            print(best_params)

            best_estimator = gs.best_estimator['rmse']
            dump.dump(os.path.join(results_dir, algo_name + '_best_estimator.pkl'), 
                      algo=best_estimator, verbose=1)

            results_df = pd.DataFrame.from_dict(gs.cv_results)
            results_df.to_csv(os.path.join(results_dir, algo_name + '_cv_results.csv'), 
                              index=False)

        overall_end_time = default_timer()
        overall_time_taken = convert_sec(overall_end_time - overall_start_time)
        grid_search_results['overall_time_taken'] = overall_time_taken

        with open(os.path.join(results_dir, 'grid_search_results.json'), 'w') as json_file:
            json.dump(grid_search_results, fp=json_file, indent=4)
        with open(os.path.join(results_dir, 'grid_search_params.json'), 'w') as json_file:
            json.dump(param_grid, fp=json_file, indent=4)

if __name__ == '__main__':
    main()
