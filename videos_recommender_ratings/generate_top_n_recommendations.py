import os
import pickle
import argparse
import pandas as pd

from collections import defaultdict

from surprise import Dataset
from surprise import Reader

from timeit import default_timer
def convert_sec(no_of_secs):
    """return no_of_secs to min or hrs string"""
    if no_of_secs < 60:
        return "Time Taken : {:06.4f}    sec".format(no_of_secs)
    elif no_of_secs < 3600:
        return "Time Taken : {:06.4f}    min".format(no_of_secs/60)
    else:
        return "Time Taken : {:06.4f}    hr".format(no_of_secs/3600)

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

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


def main():
    parser = argparse.ArgumentParser(description="Generate Top N Recommendations")
    parser.add_argument("data", help="data used to generate recommendations")
    parser.add_argument("configs", help="config of algorithms used to generate recommendations")
    args = parser.parse_args()
    
    pickle_in = open(args.configs, "rb")
    configs = pickle.load(pickle_in)
    
    user_id_col = 'learner_id'
    item_id_col = 'media_id'
    rating_col = 'like_rating'
    min_no_of_items = 10
    top_n = 50
    
    print("Loading Data...")    
    events_df = pd.read_csv(args.data) 
    print("{:30} : {:20} : {}".format("Data", "No of records", len(events_df)))
    print("{:30} : {:20} : {}".format("Data", "No of users", len(events_df[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Data", "No of items", len(events_df[item_id_col].unique())))    
    events_df = events_df.drop(['Unnamed: 0'], axis=1)
    
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
    print("trainset n_users : {}, n_users : {}, n_ratings : {}".format(trainset.n_users, trainset.n_items, trainset.n_ratings))
    
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
    print('*'*80)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'top_n_recs')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    anti_test_set_df.to_csv(os.path.join(results_dir, 'anti_test_set.csv'), index=False)
   
    start_time = default_timer()
    for config in configs:
        algo_name = config['name']
        print("Training using {}".format(algo_name))
        algo = config['algo']        
        algo.fit(trainset)

        # Predict ratings for all pairs (u, i) that are NOT in the training set.
        predictions = algo.test(testset)
        top_n_recs = get_top_n(predictions, n=top_n)

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
        result_file = os.path.join(results_dir, algo_name + '_top_n_recs.csv')
        top_recommendations_df[[user_id_col, item_id_col, rating_col]].to_csv(result_file, index=False)
        print("Recommendations Generated")
        print()

    end_time = default_timer()
    time_taken = convert_sec(end_time - start_time)
    print("All Recommendations Generated.", time_taken)
    
if __name__ == '__main__':
    main()