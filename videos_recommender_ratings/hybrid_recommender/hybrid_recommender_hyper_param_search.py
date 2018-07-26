import os
import argparse
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
import math

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import mlflow


from pprint import pprint

from timeit import default_timer
def convert_sec(no_of_secs):
    """return no_of_secs to min or hrs string"""
    if no_of_secs < 60:
        return "{:05.2f}    sec".format(no_of_secs)
    elif no_of_secs < 3600:
        return "{:05.2f}    min".format(no_of_secs/60)
    else:
        return "{:05.2f}    hr".format(no_of_secs/3600)
    
def main():
    parser = argparse.ArgumentParser(description="Hybrid Recommendor HyperParam Search")    
    parser.add_argument("configs", help="config of recommendors")
    args = parser.parse_args()
    
    pickle_file = open(args.configs, "rb")    
    selected_recommenders = pickle.load(pickle_file)
    features = []
    for config in selected_recommenders:
        selected_recommenders_prediction = config['name'] + '_est'
        features.append(selected_recommenders_prediction)
    print("The following recommenders predictions are used as features for hybrid recommender")
    for feature in features:
        print(feature)
    
    mlflow.set_tracking_uri("mlruns")
    
    target = 'like_rating'
    
    df = pd.read_csv('hybrid_recommender/all_combined_predictions.csv')
    
    estimators = [
        {
            'algo' : Ridge,
            'param_grid' : {
                'alpha' : [0.01, 1, 6, 12, 75, 100]                
            }
        },
        {
            'algo' : Lasso,
            'param_grid' : {
                'alpha' : [0.0001, 0.005, 0.001, 0.01, 0.1]                
            }
        },
        {
            'algo' : ElasticNet,
            'param_grid' : {
                'alpha' : [0.0001, 0.001, 0.01, 0.1],
                'l1_ratio' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.8],
                'max_iter' : [10000]
            }
        },
        {
            'algo' : SGDRegressor,
            'param_grid' :{
                'loss' : ['squared_loss', 'huber'],
                'penalty' : ['l1', 'l2', 'elasticnet'],
                'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'l1_ratio' : [0.1, 0.15, 0.2, 0.6, 0.8],
                'tol' : [1e-4, 1e-6]
            }
        },
#    ]
#    estimators1 = [
        {
            'algo' : RandomForestRegressor,
            'param_grid' : {
                'n_estimators' : [150, 250, 500],
                'max_depth' : [1, 7, 10, 15]
            }
        },
        {
            'algo' : GradientBoostingRegressor,
            'param_grid' : {
                'learning_rate' : [0.001, 0.01, 0.1],
                'n_estimators' : [10, 100, 150],
                'max_depth' : [1, 7, 10, 15]              
            }
        }
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'hybrid_recommender_search')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    #experiment_name = 'hybrid_recommenders'
    #exp_id = mlflow.tracking.get_experiment_id_by_name(experiment_name)
    #if exp_id is None:
    #    exp_id = mlflow.create_experiment(experiment_name)
    exp_id = None
    for estimator in estimators:
        print('*'*40)
        mlflow.start_run(experiment_id=exp_id)
        algo = estimator['algo']
        algo_name = algo.__name__
        algo_results_dir = os.path.join(results_dir, algo_name)
        if not os.path.exists(algo_results_dir):
            os.makedirs(algo_results_dir)
        
        param_grid = estimator['param_grid']
        
        start_time = default_timer()        
        mlflow.log_param('algo', algo_name)
        mlflow.log_param('param_grid', str(param_grid))
            
        grid_search = GridSearchCV(algo(), param_grid, cv=5, n_jobs=-1, return_train_score=False,
                                   scoring=['neg_mean_squared_error', 'r2'],
                                   refit='neg_mean_squared_error')
        grid_search.fit(df[features], df[target])
        
        end_time = default_timer()
        time_taken = end_time - start_time
        time_taken_str = convert_sec(time_taken)
        mlflow.log_param('time_taken', time_taken_str)
            
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        mlflow.log_param('best_params', str(best_params))
        mlflow.log_metric('best_nmse', best_score)
        best_rmse = math.sqrt(-1*best_score)
        mlflow.log_metric('best_rmse', best_rmse)
        
        results_df = pd.DataFrame.from_dict(grid_search.cv_results_)
        results_df.to_csv(os.path.join(algo_results_dir, 'cv_results.csv'), index=False)
        
        no_of_experiments = len(results_df)
        mlflow.log_param('no_of_experiments', no_of_experiments)
        time_taken_per_exp = time_taken / no_of_experiments
        time_taken_per_exp_str = convert_sec(time_taken_per_exp)
        mlflow.log_param('time_taken_per_exp', time_taken_per_exp_str)
        
        mlflow.log_artifacts(algo_results_dir)
        
        mlflow.end_run()
        print(algo)
        print("best_nmse : {:5.4f} best_rmse : {:5.4f}".format(round(best_score, 4), round(best_rmse,4)))
        print(best_params)
        
if __name__ == '__main__':
    main()        
