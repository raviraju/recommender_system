"""Module for Item Based CF Songs Recommender"""
import os
import sys
import pandas
from sklearn.cross_validation import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.item_based_cf import ItemBasedCFRecommender

def load_data():
    """Loads data and returns training and test set"""
    #Read userid-songid-listen_count triplets
    triplets_file = 'data/10000.txt'
    songs_metadata_file = 'data/song_data.csv'

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
    """Method for Item Based Recommender"""
    train_data, test_data = load_data()
    #print(train_data.head(5))
    #print(test_data.head(5))
    item_based_cf_reco = ItemBasedCFRecommender()
    item_based_cf_reco.train(train_data, 'user_id', 'song')

    #users = test_data['user_id'].unique()
    #user_id = users[0]
    user_id = '97e48f0f188e04dcdb0f8e20a29dacb881d80c9e'
    recommendations = item_based_cf_reco.recommend(user_id)
    print(recommendations)

if __name__ == '__main__':
    main()
