"""Module for Popularity Based Songs Recommender"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.popularity_based import PopularityBasedRecommender

def load_train_test(user_songs_file):
    """Loads data and returns training and test set"""
    # Read user_id-song-listen_count triplets
    user_songs_df = pd.read_csv(user_songs_file)
    user_songs_df = user_songs_df[user_songs_df['listen_count'] > 50]
    user_songs_df = user_songs_df.head(1000)
    train_data, test_data = train_test_split(user_songs_df,
                                             test_size=0.20,
                                             random_state=0)
    return train_data, test_data

def main():
    """Method for Popularity Based Recommender"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.join(current_dir, 'preprocessed_data')
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_dir = os.path.join(current_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Loading Training and Test Data")
    user_songs_file = os.path.join(preprocessed_dir, 'user_songs.csv')
    train_data, test_data = load_train_test(user_songs_file)
    # print(train_data.head(5))
    # print(test_data.head(5))
    train_data_file = os.path.join(model_dir, 'train_data.csv')
    train_data.to_csv(train_data_file)
    test_data_file = os.path.join(model_dir, 'test_data.csv')
    test_data.to_csv(test_data_file)    
    print('*' * 80)

    print("Training Recommender...")
    model = PopularityBasedRecommender(results_dir, model_dir,
                                       train_data, test_data,
                                       user_id_col='user_id',
                                       item_id_col='song')
    model.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    no_of_recs_to_eval = [10, 20]
    sample_test_users_percentage = 1
    results = model.eval(sample_test_users_percentage, no_of_recs_to_eval)
    pprint(results)
    print('*' * 80)

    print("Testing Recommendation for an User")
    users = test_data['user_id'].unique()
    user_id = users[0]
    print("Items recommended for a user with user_id : {}".format(user_id))
    recommended_items = model.recommend(user_id)
    for item in recommended_items:
        print(item)
    print('*' * 80)

if __name__ == '__main__':
    main()
