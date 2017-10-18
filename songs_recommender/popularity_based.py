"""Module for Popularity Based Songs Recommender"""
import os
import sys
import pandas as pd
from sklearn.cross_validation import train_test_split
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.popularity_based import PopularityBasedRecommender
from recommender.evaluation import PrecisionRecall

def load_train_test(user_songs_file):
    """Loads data and returns training and test set"""
    #Read user_id-song-listen_count triplets
    user_songs_df = pd.read_csv(user_songs_file)
    user_songs_df = user_songs_df[user_songs_df['listen_count'] > 50]

    train_data, test_data = train_test_split(user_songs_df, test_size = 0.20, random_state=0)

    return train_data, test_data

def main():
    """Method for Popularity Based Recommender"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.join(current_dir, 'preprocessed_data')
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("Loading Training and Test Data")
    user_songs_file = os.path.join(preprocessed_dir, 'user_songs.csv')
    train_data, test_data = load_train_test(user_songs_file)
    # print(train_data.head(5))
    # print(test_data.head(5))

    model = PopularityBasedRecommender(results_dir)
    model.train(train_data, user_id_col='user_id', item_id_col='song')
    print('*'*40)
    
    #users = test_data['user_id'].unique()
    #user_id = users[0]
    user_id = '97e48f0f188e04dcdb0f8e20a29dacb881d80c9e'
    print("Testing Recommendation for a user with user_id : {}".format(user_id))
    print("Items recommended :")
    recommended_items = model.recommend(user_id)
    for item in recommended_items:
        print(item)
    print('*'*40)
    #print("Popularity Based Recommendation Results are found in results/")

    print("Evaluating Recommender System")
    precision_recall_intf = PrecisionRecall(train_data, test_data, model, user_id_col='user_id', item_id_col='song')
    no_of_items_to_predict_list = [10, 20]
    test_users_percentage = 0.02
    results = precision_recall_intf.compute_measures(no_of_items_to_predict_list, test_users_percentage)
    pprint(results)

if __name__ == '__main__':
    main()
