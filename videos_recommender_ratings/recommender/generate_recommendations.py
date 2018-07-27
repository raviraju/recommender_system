import os
import pickle
import argparse
import pandas as pd

from collections import defaultdict

from surprise import Dataset
from surprise import Reader
from surprise.dump import dump, load

from timeit import default_timer
def convert_sec(no_of_secs):
    """return no_of_secs to min or hrs string"""
    if no_of_secs < 60:
        return "Time Taken : {:06.4f}    sec".format(no_of_secs)
    elif no_of_secs < 3600:
        return "Time Taken : {:06.4f}    min".format(no_of_secs/60)
    else:
        return "Time Taken : {:06.4f}    hr".format(no_of_secs/3600)

def get_top_n(predictions_df, user_id_col, item_id_col, rating_col, n=10):
    top_n = predictions_df.groupby([user_id_col], sort=False, as_index=False)\
                          .apply(lambda grp: grp.nlargest(n, rating_col))\
                          .reset_index()[[user_id_col, item_id_col, rating_col]]
    return top_n

       
def get_testset_stats(testset):
    """Collect Stats for testset"""
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

def load_training_data(data, user_id_col, item_id_col, rating_col, event_col, min_no_of_items):
    print("Loading Training Data...")   
    events_df = pd.read_csv(data,
                            usecols=[user_id_col, item_id_col, rating_col, event_col],
                            dtype={user_id_col: object, item_id_col: object},
                            parse_dates=[event_col]) 
    print("{:30} : {:20} : {}".format("Data", "No of records", len(events_df)))
    print("{:30} : {:20} : {}".format("Data", "No of users", len(events_df[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Data", "No of items", len(events_df[item_id_col].unique())))
    events_df.sort_values(by=event_col, inplace=True)
    
    #filtering data to be imported
    print('*'*80)
    print("Filtering Data...: ", "Min no of items : ", min_no_of_items)
    user_items_df = events_df.groupby([user_id_col])\
                             .agg({item_id_col: (lambda x: len(x.unique()))})\
                             .rename(columns={item_id_col : 'no_of_items'})\
                             .reset_index()
    user_min_items_df = user_items_df[user_items_df['no_of_items'] >= min_no_of_items]        
    filtered_events_df = pd.merge(user_min_items_df, events_df, how='inner', on=user_id_col)
    print("{:30} : {:20} : {}".format("Filtered Data", "No of records", len(filtered_events_df)))
    print("{:30} : {:20} : {}".format("Filtered Data", "No of users", len(filtered_events_df[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Filtered Data", "No of items", len(filtered_events_df[item_id_col].unique())))        
    user_item_rating_df = filtered_events_df[[user_id_col, item_id_col, rating_col]]
    
    print('*'*80)
    reader = Reader(rating_scale=(1, 3))
    data = Dataset.load_from_df(user_item_rating_df, reader=reader)
    trainset = data.build_full_trainset()
    return trainset

def load_anti_test_set(trainset, user_id_col, item_id_col, rating_col):
    start_time = default_timer()
    testset = trainset.build_anti_testset()
    testset_n_users, testset_n_items, testset_n_ratings = get_testset_stats(testset)
    print("testset  n_users : {}, n_users : {}, n_ratings : {}".format(testset_n_users, testset_n_items, testset_n_ratings))   
    user_item_ratings = []
    for (user_id, item_id, rating) in testset:
        user_item_rating = dict()
        user_item_rating[user_id_col] = user_id
        user_item_rating[item_id_col] = item_id
        user_item_rating[rating_col] = float("{0:.4f}".format(rating))
        user_item_ratings.append(user_item_rating)
        #print((user_id, item_id, rating))
    anti_test_set_df = pd.DataFrame(user_item_ratings)
    end_time = default_timer()
    time_taken = convert_sec(end_time - start_time)
    print("Built Anti Testset.", time_taken)
    return anti_test_set_df
    
def main():
    parser = argparse.ArgumentParser(description="Generate Recommendations")    
    parser.add_argument("configs", help="config of recommendors")
    parser.add_argument("--train", help="data used to train recommendor")
    parser.add_argument("--predict", help="generate predictions for anti-testset", action='store_true')
    args = parser.parse_args()
    
    pickle_in = open(args.configs, "rb")
    configs = pickle.load(pickle_in)
    
    user_id_col = 'learner_id'
    item_id_col = 'media_id'
    rating_col = 'like_rating'
    event_col = 'event_time'
    min_no_of_items = 10
    n = 50
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, '../results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    models_dir = os.path.join(results_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    if args.train:
        trainset = load_training_data(args.train,
                                      user_id_col, item_id_col, rating_col, event_col,
                                      min_no_of_items)
        print("trainset n_users : {}, n_users : {}, n_ratings : {}".format(trainset.n_users,
                                                                           trainset.n_items,
                                                                           trainset.n_ratings))
        start_time = default_timer()
        for config in configs:
            algo_name = config['name']
            algo = config['algo']
            model_file = os.path.join(models_dir, algo_name+'.pickle')

            print("Training {}".format(algo_name))

            algo_start_time = default_timer()

            algo.fit(trainset)                
            dump(model_file, algo=algo)

            algo_end_time = default_timer()
            algo_time_taken = convert_sec(algo_end_time - algo_start_time)
            print("Trained {}. {}".format(algo_name, algo_time_taken))

        end_time = default_timer()
        time_taken = convert_sec(end_time - start_time)
        print("Trained All Recommenders.", time_taken)
        ##############################################################################################################
        start_time = default_timer()
        print("Preparing Anti TestSet...")
        anti_test_set_df = load_anti_test_set(trainset, user_id_col, item_id_col, rating_col)
        anti_test_set_file = os.path.join(results_dir, 'anti_test_set.csv')
        print("Saving Anti TestSet...")
        anti_test_set_df[[user_id_col, item_id_col, rating_col]].to_csv(anti_test_set_file,
                                                                        chunksize=10000,
                                                                        index=False)
        print(anti_test_set_file)
        end_time = default_timer()
        time_taken = convert_sec(end_time - start_time)
        print("Generated Anti TestSet.", time_taken)
        print('*'*80)
        ##############################################################################################################
        
    if args.predict:
        anti_test_set_file = os.path.join(results_dir, 'anti_test_set.csv')
        anti_test_set_prediction_file = os.path.join(results_dir, 'anti_test_set_predictions.csv')

        trained_recommenders = dict()
        for config in configs:
            algo_name = config['name']

            algo = config['algo']
            model_file = os.path.join(models_dir, algo_name+'.pickle')
                        
            print("Loading {}".format(algo_name))
            _, model = load(model_file)
            trained_recommenders[algo_name] = model
            
        print("Loading Anti TestSet : ", anti_test_set_file)
        
        if os.path.exists(anti_test_set_prediction_file):
            os.remove(anti_test_set_prediction_file)
        
        start_time = default_timer()
        reader = pd.read_csv(anti_test_set_file,
                             dtype={user_id_col: object, item_id_col: object},
                             chunksize=100000)
        total_no_of_rows = 0
        first_set = True
        for df in reader:
            anti_test_predictions = []
            for _, (user_id, item_id, actual_rating) in df.iterrows():
                
                anti_test_prediction = dict()
                anti_test_prediction[user_id_col] = user_id
                anti_test_prediction[item_id_col] = item_id
                    
                for algo_name, algo in trained_recommenders.items():
                    prediction = algo.predict(user_id, item_id, actual_rating)
                    predicted_rating = prediction.est
                    #print(algo_name, actual_rating, predicted_rating)
                    anti_test_prediction[algo_name + '_est'] = float("{0:.4f}".format(predicted_rating))
                #print(anti_test_prediction)
                anti_test_predictions.append(anti_test_prediction)
            anti_test_predictions_df = pd.DataFrame(anti_test_predictions)
            #print(anti_test_predictions_df.head())
            no_of_rows = anti_test_predictions_df.shape[0]
            total_no_of_rows += no_of_rows
            print("Generated predictions for {} rows...".format(total_no_of_rows))
            if first_set:
                anti_test_predictions_df.to_csv(anti_test_set_prediction_file, index=False)
                first_set = False
            else:
                anti_test_predictions_df.to_csv(anti_test_set_prediction_file, mode='a', header=False, index=False)
        print(anti_test_set_prediction_file)

        end_time = default_timer()
        time_taken = convert_sec(end_time - start_time)
        print("Generated Anti TestSet Predictions.", time_taken)
        
        ##############################################################################################################
        start_time = default_timer()
        for config in configs:
            algo_name = config['name']
            algo_prediction_col = algo_name + '_est'
            print("Loading anti_test_set predictions of {} ...".format(algo_prediction_col))
            
            algo_predictions_df = pd.read_csv(anti_test_set_prediction_file, usecols=[user_id_col,
                                                                                      item_id_col,
                                                                                      algo_prediction_col])
            
            algo_predictions_df.rename(columns={algo_prediction_col:rating_col}, inplace=True)
            algo_predictions_file = os.path.join(results_dir, algo_name + '_predictions.csv')            
            print("Extracting predictions into : ", algo_predictions_file)
            algo_predictions_df[[user_id_col, item_id_col, rating_col]].to_csv(algo_predictions_file, index=False)

            print("Collect the TOP N recommended items for each user...")
            top_recommendations_df = get_top_n(algo_predictions_df, user_id_col, item_id_col, rating_col, n)            
            print(top_recommendations_df.head())
            algo_top_n_predictions_file = os.path.join(results_dir, algo_name + '_top_n_recs.csv')
            print("Capturing TOP_N predictions into : ", algo_top_n_predictions_file)
            top_recommendations_df[[user_id_col, item_id_col, rating_col]].to_csv(algo_top_n_predictions_file, index=False)

        end_time = default_timer()
        time_taken = convert_sec(end_time - start_time)
        print("Captured Recommender Predictions and Top N recommendations.", time_taken)

if __name__ == '__main__':
    main()
