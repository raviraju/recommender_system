"""Module to split data into train and test"""
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_random_split(train_test_dir, data, test_size=0.2):
    """Loads data and returns training and test set"""
    print("Generate Training and Test Data")
    # Read user_id-song-listen_count triplets
    users_songs_df = pd.read_csv(data)
    print("{:10} : {:20} : {}".format("Total", "No of records",
                                      len(users_songs_df)))
    #2000000 records
    #filtering data to be imported
    users_songs_df = users_songs_df[users_songs_df['listen_count'] > 50]
    #4782 records
    #users_songs_df = users_songs_df.head(1000)
    print("{:10} : {:20} : {}".format("Filtered", "No of records",
                                      len(users_songs_df)))
    train_data, test_data = train_test_split(users_songs_df,
                                             test_size=test_size,
                                             random_state=None)
                                             #random_state=0)
                                             #If int, random_state is the seed used
                                             #by the random number generator
    print("{:10} : {:20} : {}".format("Train Data", "No of records", len(train_data)))
    print("{:10} : {:20} : {}".format("Test Data", "No of records", len(test_data)))
    train_data_file = os.path.join(train_test_dir, 'train_data.csv')
    train_data.to_csv(train_data_file)
    test_data_file = os.path.join(train_test_dir, 'test_data.csv')
    test_data.to_csv(test_data_file)
    print("Train and Test Data are in ", train_test_dir)

def generate_users_split(train_test_dir, data, test_size=0.2):
    """Loads data and returns training and test set"""
    print("Generate Training and Test Data")
    # Read user_id-song-listen_count triplets
    users_songs_df = pd.read_csv(data)
    print("{:10} : {:20} : {}".format("Total", "No of records",
                                      len(users_songs_df)))
    #2000000 records
    #filtering data to be imported
    user_songs_df = users_songs_df.groupby('user_id')\
                                  .agg({'song' : 'count'})\
                                  .rename(columns={'song' : 'no_of_songs'})\
                                  .reset_index()
    dist = user_songs_df['no_of_songs'].describe()
    no_of_songs_list = [dist['min'], dist['25%'], dist['50%'], dist['75%'], dist['max']]
    for no_of_songs in no_of_songs_list:
        print("No of Songs : ", no_of_songs,
              " No of listerners : ",
              len(user_songs_df[user_songs_df['no_of_songs'] == no_of_songs]))

    min_no_of_songs = 10
    min_user_songs_df = user_songs_df[user_songs_df['no_of_songs'] >= min_no_of_songs]
    print("Min no of Songs : ", min_no_of_songs,
          " No of listerners : ",
          len(min_user_songs_df))

    train_users, test_users = train_test_split(min_user_songs_df,
                                               test_size=test_size,
                                               random_state=None)
                                               #random_state=0)
                                               #If int, random_state is the seed used
                                               #by the random number generator
    train_data = pd.merge(train_users, users_songs_df, how='inner', on='user_id')
    test_data = pd.merge(test_users, users_songs_df, how='inner', on='user_id')

    # min_listen_count = 50
    # train_data = train_data[train_data['listen_count'] > min_listen_count]
    # test_data = test_data[test_data['listen_count'] > min_listen_count]

    print("{:10} : {:20} : {}".format("Train Data", "No of records", len(train_data)))
    print("{:10} : {:20} : {}".format("Test Data", "No of records", len(test_data)))
    train_data_file = os.path.join(train_test_dir, 'train_data.csv')
    train_data.to_csv(train_data_file)
    test_data_file = os.path.join(train_test_dir, 'test_data.csv')
    test_data.to_csv(test_data_file)
    print("Train and Test Data are in ", train_test_dir)

def main():
    """interface to load and split data into train and test"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_test_dir = os.path.join(current_dir, 'train_test_data')
    if not os.path.exists(train_test_dir):
        os.makedirs(train_test_dir)

    data_dir = os.path.join(current_dir, 'preprocessed_data')
    data = os.path.join(data_dir, 'user_songs.csv')

    parser = argparse.ArgumentParser(description="Split train and test data")
    parser.add_argument("--random_split",
                        help="Random split data into train and test",
                        action="store_true")
    parser.add_argument("--users_split",
                        help="split users into train and test",
                        action="store_true")
    parser.add_argument("test_size",
                        help="test_size ratio", type=float)

    args = parser.parse_args()
    if args.random_split and args.test_size:
        generate_random_split(train_test_dir, data, args.test_size)
    elif args.users_split and args.test_size:
        generate_users_split(train_test_dir, data, args.test_size)
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()
