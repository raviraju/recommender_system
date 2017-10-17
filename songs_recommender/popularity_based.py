"""Module for Popularity Based Songs Recommender"""
import os
import sys
import pandas
from sklearn.cross_validation import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.popularity_based import PopularityBasedRecommender
from recommender.evaluation import PrecisionRecall

def load_data(data_dir):
    """Loads data and returns training and test set"""
    #Read userid-songid-listen_count triplets
    triplets_file = os.path.join(data_dir, '10000.txt')
    songs_metadata_file = os.path.join(data_dir, 'song_data.csv')

    song_df_1 = pandas.read_table(triplets_file, header=None)
    song_df_1.columns = ['user_id', 'song_id', 'listen_count']

    #Read song  metadata
    song_df_2 = pandas.read_csv(songs_metadata_file)

    #Merge the two dataframes above to create input dataframe for recommender systems
    song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

    song_df = song_df.head(10000)

    #Merge song title and artist_name columns to make a merged column
    song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']

    train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)

    return train_data, test_data

def main():
    """Method for Popularity Based Recommender"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train_data, test_data = load_data(data_dir)
    # print(train_data.head(5))
    # print(test_data.head(5))

    model = PopularityBasedRecommender(results_dir)
    model.train(train_data, user_id_col='user_id', item_id_col='song')

    #users = test_data['user_id'].unique()
    #user_id = users[0]
    user_id = '97e48f0f188e04dcdb0f8e20a29dacb881d80c9e'
    recommended_items = model.recommend(user_id)
    print(recommended_items)
    #print("Popularity Based Recommendation Results are found in results/")

    precision_recall_intf = PrecisionRecall(train_data, test_data, model, user_id_col='user_id', item_id_col='song')
    results = precision_recall_intf.compute_measures(test_users_percentage=0.1)
    print(results)

if __name__ == '__main__':
    main()
