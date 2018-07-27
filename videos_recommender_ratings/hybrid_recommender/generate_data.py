import os
import pickle
import argparse
from functools import reduce

import pandas as pd

def generate_data(configs, data_dir, result_dir, user_id_col, item_id_col, rating_col):    
    no_of_kfolds = 10    
    
    recommenders = []
    for config in configs:
        algo = config['algo']        
        algo_name = config['name']        
        recommenders.append(algo_name)
        
        
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    all_dfs = pd.DataFrame()
    for kfold in range(1, no_of_kfolds+1):
        #print(kfold)
        recommender_dfs = dict()
        kfold_dir = 'kfold_exp_'+str(kfold)
        for recommender in recommenders:
            predictions_file = os.path.join(data_dir,
                                            recommender, 
                                            'kfold_experiments', 
                                            kfold_dir, 'predictions.csv')
            #print(predictions_file)
            predictions_df = pd.read_csv(predictions_file,
                                         dtype={'uid': object, 'iid': object})
            #print(predictions_df.shape)
            recommender_dfs[recommender] = predictions_df[['uid', 'iid', 'r_ui', 'est']].rename(columns={'est': recommender + '_est'})
        combined_df = reduce(lambda x, y: pd.merge(x, y, on=['uid', 'iid', 'r_ui']), recommender_dfs.values())
        #print(combined_df.head())
        kfold_file = os.path.join(result_dir, 'combined_predictions_' + str(kfold) + '.csv')        
        print(kfold_file)
        combined_df.rename(columns={'uid' : user_id_col, 'iid' : item_id_col, 'r_ui' : rating_col}, inplace=True)        
        combined_df.to_csv(kfold_file, index=False)
        all_dfs = all_dfs.append(combined_df)
    all_combined_predictions_file = os.path.join(result_dir, 'all_combined_predictions.csv')
    all_dfs.to_csv(all_combined_predictions_file, index=False)
    print(all_combined_predictions_file)
        
def main():
    parser = argparse.ArgumentParser(description="Hybrid Recommender")
    parser.add_argument("configs", help="config of algorithms used to generate recommendations")
    parser.add_argument("testing_data", help="path of generated recommendations for testing data")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, '../data_hybrid_recommender')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    pickle_in = open(args.configs,"rb")
    configs = pickle.load(pickle_in)
    
    user_id_col = 'learner_id'
    item_id_col = 'media_id'
    rating_col = 'like_rating'
    
    generate_data(configs, args.testing_data, results_dir, user_id_col, item_id_col, rating_col)

    
if __name__ == '__main__':
    main()
    