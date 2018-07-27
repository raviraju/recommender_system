import os
import argparse
import pickle
import pandas as pd
from functools import reduce
from collections import defaultdict
from sklearn.externals import joblib

from timeit import default_timer
def convert_sec(no_of_secs):
    """return no_of_secs to min or hrs string"""
    if no_of_secs < 60:
        return "Time Taken : {:06.4f}    sec".format(no_of_secs)
    elif no_of_secs < 3600:
        return "Time Taken : {:06.4f}    min".format(no_of_secs/60)
    else:
        return "Time Taken : {:06.4f}    hr".format(no_of_secs/3600)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def get_top_n(predictions_df, user_id_col, item_id_col, rating_col, n=10):
    top_n = predictions_df.groupby([user_id_col], sort=False, as_index=False)\
                          .apply(lambda grp: grp.nlargest(n, rating_col))\
                          .reset_index()[[user_id_col, item_id_col, rating_col]]
    return top_n
    
def main():
    parser = argparse.ArgumentParser(description="Hybrid Recommendor Generate Recommendations")    
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
    
    user_id_col = 'learner_id'
    item_id_col = 'media_id'
    rating_col = 'like_rating'
    top_n = 50
    
    target = rating_col
        
    tuned_models = [
        ElasticNet(alpha=0.0001, l1_ratio=0.1, max_iter=10000),
        GradientBoostingRegressor(learning_rate=0.1, max_depth=1, n_estimators=100)
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, '../results')
    models_dir = os.path.join(results_dir, 'models')
    
    train_file = 'data_hybrid_recommender/all_combined_predictions.csv'    
    anti_test_set_file = 'results/anti_test_set_predictions.csv'
    
    print("Loading Train Data {}...".format(train_file))
    train_df = pd.read_csv(train_file)    
    train_X, train_y = train_df[features], train_df[target]    
    print(train_X.shape, train_y.shape)
    
    print("Loading Test  Data {}...".format(anti_test_set_file))
    test_df = pd.read_csv(anti_test_set_file)
    test_X = test_df[features]
    print(test_X.shape)

    for estimator in tuned_models:
        model_name = type(estimator).__name__
        algo_start_time = default_timer()

        model = estimator.fit(train_X, train_y)
        joblib.dump(model, os.path.join(models_dir, 'Hybrid_' + model_name + '.pickle'))
        predicted_test_y = model.predict(test_X)
        
        algo_end_time = default_timer()
        algo_time_taken = convert_sec(algo_end_time - algo_start_time)
        print("Trained and Predicted {}. {}".format(model_name, algo_time_taken))

        start_time = default_timer()
        prediction_df = pd.DataFrame(test_df[[user_id_col, item_id_col]],
                                     dtype=object)
        prediction_df.loc[:, rating_col] = predicted_test_y
        prediction_df[rating_col] = prediction_df[rating_col].round(4)
        prediction_file = os.path.join(results_dir, 'Hybrid_' + model_name + '_predictions.csv')
        print("Capturing predictions in :      {}".format(prediction_file))
        prediction_df[[user_id_col, item_id_col, rating_col]].to_csv(prediction_file, index=False)
        

        print("Collect the TOP N recommended items for each user...")
        top_recommendations_df = get_top_n(prediction_df, user_id_col, item_id_col, rating_col, n=top_n)
        result_file = os.path.join(results_dir, 'Hybrid_' + model_name + '_top_n_recs.csv')
        print("Capturing TOP_N predictions in : {}".format(result_file))
        top_recommendations_df[[user_id_col, item_id_col, rating_col]].to_csv(result_file, index=False)
        print()
        end_time = default_timer()
        time_taken = convert_sec(end_time - start_time)
        print("Captured Recommender Predictions and Top N recommendations.", time_taken)

if __name__ == '__main__':
    main() 
