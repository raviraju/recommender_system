import os
import pickle
import argparse
from functools import reduce

import pandas as pd

def generate_data(configs, data_dir, result_dir):    
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
            predictions_df = pd.read_csv(predictions_file)
            #print(predictions_df.shape)
            recommender_dfs[recommender] = predictions_df[['uid', 'iid', 'r_ui', 'est']].rename(columns={'est': recommender + '_est'})
        combined_df = reduce(lambda x, y: pd.merge(x, y, on=['uid', 'iid', 'r_ui']), recommender_dfs.values())
        #print(combined_df.head())
        kfold_file = os.path.join(result_dir, 'combined_predictions_' + str(kfold) + '.csv')
        print(kfold_file)
        combined_df.to_csv(kfold_file, index=False)
        all_dfs = all_dfs.append(combined_df)
    all_dfs.to_csv(os.path.join(result_dir, 'all_combined_predictions.csv'), index=False)
        
def main():
    parser = argparse.ArgumentParser(description="Hybrid Recommender")
    parser.add_argument("configs", help="config of algorithms used to generate recommendations")        
    parser.add_argument("validation_data", help="path of generated recommendations for validation data")
    parser.add_argument("testing_data", help="path of generated recommendations for testing data")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'hybrid_recommender')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    pickle_in = open(args.configs,"rb")
    configs = pickle.load(pickle_in)
    
    
    training_data_dir = os.path.join(results_dir, 'training_data')
    generate_data(configs, args.validation_data, training_data_dir)
    
    testing_data_dir = os.path.join(results_dir, 'testing_data')
    generate_data(configs, args.testing_data, testing_data_dir)
        
            
    
    
if __name__ == '__main__':
    main()
    