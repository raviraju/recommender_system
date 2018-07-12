import os
import argparse
import pickle
import pandas as pd
from functools import reduce
from collections import defaultdict

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def load_anti_test_df(configs, top_n_recs_dir, user_id_col, item_id_col, rating_col):
    recommenders = []
    for config in configs:
        algo = config['algo']        
        algo_name = config['name']        
        recommenders.append(algo_name)
    recommender_dfs = dict()
    for recommender in recommenders:
        predictions_file = os.path.join(top_n_recs_dir,
                                        recommender + '_predictions.csv')
        #print(predictions_file)
        predictions_df = pd.read_csv(predictions_file)
        #print(predictions_df.shape)
        #print(predictions_df.head())
        recommender_dfs[recommender] = predictions_df[[user_id_col, item_id_col, rating_col]].rename(columns={rating_col: recommender + '_est'})
    recommendations_df = reduce(lambda x, y: pd.merge(x, y, on=[user_id_col, item_id_col]), recommender_dfs.values())
    #print(recommendations_df.head())
    #print(recommendations_df.shape)
    
    #get actual rating from anti_test_set
    anti_test_df = pd.read_csv(os.path.join(top_n_recs_dir, 'anti_test_set.csv'))
    #print(anti_test_df.head())
    #print(anti_test_df.shape)

    combined_df = pd.merge(recommendations_df, anti_test_df, on=[user_id_col, item_id_col])
    #print(combined_df.head())
    #print(combined_df.shape)
    return combined_df
        
def get_top_n(prediction_df, user_id_col, item_id_col, rating_col, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for _, prediction in prediction_df.iterrows():
        uid = prediction[user_id_col]
        iid = prediction[item_id_col]
        est = prediction[rating_col]
        top_n[uid].append((iid, est))
    
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
    
def main():
    parser = argparse.ArgumentParser(description="Hybrid Recommender Generate Top N Recs")
    parser.add_argument("configs", help="config of algorithms used to generate Top N recommendations")
    parser.add_argument("top_n_recs_dir", help="path of generated Top N recs for each recommender")
    args = parser.parse_args()
    
    user_id_col = 'learner_id'
    item_id_col = 'media_id'
    rating_col = 'like_rating'
    top_n = 50
    
    pickle_in = open(args.configs,"rb")
    configs = pickle.load(pickle_in)
    anti_test_df = load_anti_test_df(configs, args.top_n_recs_dir, user_id_col, item_id_col, rating_col)
    
    train_df = pd.read_csv('hybrid_recommender/all_combined_predictions.csv')
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
    train_X, train_y = train_df[features], train_df[target]    
    #print(train_X.shape, train_y.shape)
    test_X, test_y = anti_test_df[features], anti_test_df[target]
    #print(test_X.shape, test_y.shape)
    for estimator in tuned_models:
        model_name = type(estimator).__name__
        model = estimator.fit(train_X, train_y)
        predicted_test_y = model.predict(test_X)
        
        prediction_df = pd.DataFrame(anti_test_df[[user_id_col, item_id_col]])
        prediction_df.loc[:, rating_col] = predicted_test_y
        #print(prediction_df.head())
        prediction_file = os.path.join(args.top_n_recs_dir, 'Hybrid_' + model_name + '_predictions.csv')
        print(prediction_file)
        prediction_df[[user_id_col, item_id_col, rating_col]].to_csv(prediction_file, index=False)
        
        top_n_recs = get_top_n(prediction_df, user_id_col, item_id_col, rating_col, n=top_n)
        
        # Collect the TOP N recommended items for each user
        top_recommendations = []
        for uid, user_ratings in top_n_recs.items():
            #print(uid, [iid for (iid, _) in user_ratings])
            for iid, rating in user_ratings:
                reco = {
                    user_id_col : uid,
                    item_id_col : iid,
                    rating_col  : float("{0:.4f}".format(rating))
                }
                #print(reco)
                top_recommendations.append(reco)

        top_recommendations_df = pd.DataFrame(top_recommendations)
        result_file = os.path.join(args.top_n_recs_dir, 'Hybrid_' + model_name + '_top_n_recs.csv')
        print(result_file)
        top_recommendations_df[[user_id_col, item_id_col, rating_col]].to_csv(result_file, index=False)

if __name__ == '__main__':
    main() 
