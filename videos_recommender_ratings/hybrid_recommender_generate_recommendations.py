import pandas as pd

from functools import reduce
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def get_prediction_metrics(y_true, y_pred):
    metrics = dict()
    metrics['r2_score'] = r2_score(y_true, y_pred)
    metrics['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = sqrt(mean_squared_error(y_true, y_pred))
    return metrics

def get_avg_test_metrics(kfold_test_metrics):
    metrics = dict()
    metrics['r2_score'] = []
    metrics['mean_absolute_error'] = []
    metrics['rmse'] = []
    
    for test_metrics in kfold_test_metrics:
        metrics['r2_score'].append(test_metrics['r2_score'])
        metrics['mean_absolute_error'].append(test_metrics['mean_absolute_error'])
        metrics['rmse'].append(test_metrics['rmse'])
    #print(metrics)
    avg_metrics = dict()
    avg_metrics['r2_score'] = round(np.mean(metrics['r2_score']), 4)
    avg_metrics['mean_absolute_error'] = round(np.mean(metrics['mean_absolute_error']), 4)
    avg_metrics['rmse'] = round(np.mean(metrics['rmse']), 4)
    return avg_metrics

def get_avg_metrics(estimator, no_of_kfolds, features, target, kfold_dfs,
                    user_id_col, item_id_col, rating_col):
    #print(estimator)
    kfold_test_metrics = []
    for kfold in range(1, no_of_kfolds+1):
        #print('*'*40)
        #print('kfold_', str(kfold))
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for kfold_id, df in kfold_dfs.items():
            if kfold == kfold_id:
                test_df = df
            else:
                train_df = train_df.append(df)
        #print("train_df : {} test_df : {}".format(train_df.shape, test_df.shape))
        train_users = set(train_df[user_id_col].unique())
        test_users = set(test_df[user_id_col].unique())
        common_users = train_users & test_users
        #print("No of train_users  : ", len(train_users))
        #print("No of test_users   : ", len(test_users))
        #print("No of common_users : ", len(common_users))
        #input()

        train_X, train_y = train_df[features], train_df[target]    
        #print(train_X.shape, train_y.shape)
        test_X, test_y = test_df[features], test_df[target]
        #print(test_X.shape, test_y.shape)

        model = estimator.fit(train_X, train_y)

        #predicted_train_y = model.predict(train_X)
        #train_metrics = get_prediction_metrics(train_y, predicted_train_y)
        #print(train_metrics)
        predicted_test_y = model.predict(test_X)
        test_metrics = get_prediction_metrics(test_y, predicted_test_y)
        #print(test_metrics)
        kfold_test_metrics.append(test_metrics)
    #print(kfold_test_metrics)
    avg_metrics = get_avg_test_metrics(kfold_test_metrics)
    return avg_metrics

def main():
    user_id_col = 'learner_id'
    item_id_col = 'media_id'
    rating_col = 'like_rating'
    
    no_of_kfolds = 10
    kfold_dfs = dict()
    for kfold in range(1, no_of_kfolds+1):
        df = pd.read_csv('hybrid_recommender/combined_predictions_' + str(kfold) + '.csv')
        kfold_dfs[kfold] = df
        #print(kfold_dfs[kfold].shape, kfold_dfs[kfold]['uid'].nunique())    
        
        
    features = ['BaselineOnly_SGD_Tuned_est',
                'Knn_UserBased_ZScore_MSD_Tuned_est',
                'Knn_ItemBased_ZScore_MSD_Tuned_est',
                'Knn_UserBased_Baseline_SGD_Tuned_est',
                'Knn_ItemBased_Baseline_SGD_Tuned_est', 
                'SVD_biased_Tuned_est',
                'SVDpp_biased_Tuned_est']
    target = rating_col
    
    
    tuned_models = [
        LinearRegression(),
        Ridge(alpha=10),
        Lasso(alpha=0.0001),
        ElasticNet(alpha=0.01, l1_ratio=0.2),
        SGDRegressor(alpha=0.0001, l1_ratio=0.1, loss='squared_loss', penalty='elasticnet', tol=0.0001),
        RandomForestRegressor(max_depth=15, n_estimators=150),
        GradientBoostingRegressor(learning_rate=0.1, max_depth=1, n_estimators=100)
    ]
    for model in tuned_models:
        model_name = type(model).__name__
        results = get_avg_metrics(model, no_of_kfolds, features, target, kfold_dfs,
                                  user_id_col, item_id_col, rating_col)
        print("{:40s} AVG RMSE : {}".format(model_name, results['rmse']))
        
        
if __name__ == '__main__':
    main() 