"""Module for Preprocessing data for Song Recommendations"""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_user_songs(user_song_triplets_file, songs_metadata_file):
    """Extract userid-song-listen_count triplets"""

    song_df_1 = pd.read_table(user_song_triplets, header=None)
    song_df_1.columns = ['user_id', 'song_id', 'listen_count']
    #eliminate users with null id
    song_df_1 = song_df_1[song_df_1['user_id'].notnull()]
    #eliminate songs with null code
    song_df_1 = song_df_1[song_df_1['song_id'].notnull()]

    #Read songs metadata
    song_df_2 = pd.read_csv(songs_metadata_file)

    #Merge the two dataframes above to create input dataframe for recommender systems
    song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

    #Merge song title and artist_name columns to make a merged column
    song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']
    user_song_df = song_df[['user_id', 'song', 'listen_count']]
    user_song_df = user_song_df.sort_values('listen_count', ascending=False)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_data_dir = os.path.join(current_dir, 'preprocessed_data')
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)
    user_songs_file = os.path.join(preprocessed_data_dir, 'user_songs.csv')
    user_song_df.to_csv(user_songs_file)
    print("Preprocessed data available in preprocessed_data/")

if __name__ == '__main__':
    user_song_triplets = os.path.join('data/', '10000.txt')
    songs_file = os.path.join('data/', 'song_data.csv')
    extract_user_songs(user_song_triplets, songs_file)
