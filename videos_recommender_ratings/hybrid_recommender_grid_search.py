import pandas as pd
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from pprint import pprint

def main():
    features = ['BaselineOnly_SGD_Tuned_est',
                'Knn_UserBased_ZScore_MSD_Tuned_est',
                'Knn_ItemBased_ZScore_MSD_Tuned_est',
                'Knn_UserBased_Baseline_SGD_Tuned_est',
                'Knn_ItemBased_Baseline_SGD_Tuned_est', 
                'SVD_biased_Tuned_est',
                'SVDpp_biased_Tuned_est']
    target = 'r_ui'
    
    df = pd.read_csv('hybrid_recommender/training_data/all_combined_predictions.csv')
    
    estimators = [
        {
            'algo' : Ridge,
            'param_grid' : {
                'alpha' : [0.01, 0.1, 1, 10, 100]                
            }
        },
        {
            'algo' : Lasso,
            'param_grid' : {
                'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]                
            }
        }
    ]
    '''
        {
            'algo' : ElasticNet,
            'param_grid' : {
                'alpha' : [0.5, 1, 10, 100],
                'l1_ratio' : [0.2, 0.4, 0.6, 0.8]
            }
        },
        {
            'algo' : SGDRegressor,
            'param_grid' :{
                'loss' : ['squared_loss', 'huber']
            }
        }    
    ]
    '''    
    
    for estimator in estimators:
        print('*'*40)
        algo = estimator['algo']
        param_grid = estimator['param_grid']
        grid_search = GridSearchCV(algo(), param_grid, cv=5, n_jobs=-1, 
                                   scoring=['neg_mean_squared_error', 'r2'],
                                   refit='neg_mean_squared_error')
        grid_search.fit(df[features], df[target])
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(algo)
        print(best_score)
        print(best_params)      
        #pprint(grid_search.cv_results_)
        
if __name__ == '__main__':
    main()        