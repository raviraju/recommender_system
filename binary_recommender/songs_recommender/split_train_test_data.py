"""Module to split data into train and test"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def generate_random_split(train_test_dir, data, test_size=0.2, min_no_of_listen_count=10):
    """Loads data and returns training and test set"""
    print("Generate Training and Test Data")
    # Read user_id-song-listen_count triplets
    users_songs_df = pd.read_csv(data)
    print("{:10} : {:20} : {}".format("Total", "No of records",
                                      len(users_songs_df)))
    #2000000 records
    #filtering data to be imported
    data = users_songs_df[users_songs_df['listen_count'] > min_no_of_listen_count]
    print("{:10} : {:20} : {}".format("Filtered", "No of records",
                                      len(data)))
    train_data, test_data = train_test_split(users_songs_df,
                                             test_size=test_size,
                                             random_state=None)
                                             #random_state=0)
                                             #If int, random_state is the seed used
                                             #by the random number generator
    print()
    print("{:10} : {:20} : {}".format("Data", "No of records", len(data)))
    print("{:10} : {:20} : {}".format("Data", "No of listeners", len(data['user_id'].unique())))
    print("{:10} : {:20} : {}".format("Data", "No of books", len(data['song'].unique())))
    print()
    print("{:10} : {:20} : {}".format("Train Data", "No of records", len(train_data)))
    print("{:10} : {:20} : {}".format("Train Data", "No of listeners",
                                      len(train_data['user_id'].unique())))
    print("{:10} : {:20} : {}".format("Train Data", "No of books",
                                      len(train_data['song'].unique())))
    print()
    print("{:10} : {:20} : {}".format("Test Data", "No of records", len(test_data)))
    print("{:10} : {:20} : {}".format("Test Data", "No of listeners",
                                      len(test_data['user_id'].unique())))
    print("{:10} : {:20} : {}".format("Test Data", "No of books",
                                      len(test_data['song'].unique())))

    common_listeners = set(train_data['user_id'].unique()) & set(test_data['user_id'].unique())
    common_books = set(train_data['song'].unique()) & set(test_data['song'].unique())
    print()
    print("{:10} : {:20} : {}".format("Common ", "No of listeners", len(common_listeners)))
    print("{:10} : {:20} : {}".format("Common ", "No of books", len(common_books)))

    train_data_file = os.path.join(train_test_dir, 'train_data.csv')
    train_data.to_csv(train_data_file)
    test_data_file = os.path.join(train_test_dir, 'test_data.csv')
    test_data.to_csv(test_data_file)
    print("Train and Test Data are in ", train_test_dir)

def generate_users_split(train_test_dir, data, test_size=0.2, min_no_of_songs=10):
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

    min_user_songs_df = user_songs_df[user_songs_df['no_of_songs'] >= min_no_of_songs]
    print("Min no of Songs : ", min_no_of_songs,
          " No of listerners : ",
          len(min_user_songs_df))

    data = pd.merge(min_user_songs_df, users_songs_df, how='inner', on='user_id')

    listeners = data['user_id'].unique()
    no_of_listeners = len(listeners)
    no_of_test_listeners = int(no_of_listeners * test_size)
    #no_of_train_listeners = no_of_listeners - no_of_test_listeners

    listeners_set = set(listeners)
    test_listeners_set = set(np.random.choice(listeners, no_of_test_listeners, replace=False))
    train_listeners_set = listeners_set - test_listeners_set
    common_listeners = train_listeners_set & test_listeners_set
    print("No of listeners : {}".format(len(listeners_set)))
    print("No of train listeners : {}".format(len(train_listeners_set)))
    print("No of test listeners : {}".format(len(test_listeners_set)))
    print("No of common listeners : {}".format(len(common_listeners)))

    test_data = data[data['user_id'].isin(test_listeners_set)]
    train_data = data[data['user_id'].isin(train_listeners_set)]

    common_listeners = set(train_data['user_id'].unique()) & set(test_data['user_id'].unique())
    common_books = set(train_data['song'].unique()) & set(test_data['song'].unique())

    print()
    print("{:10} : {:20} : {}".format("Train Data", "No of records", len(train_data)))
    print("{:10} : {:20} : {}".format("Train Data", "No of listeners",
                                      len(train_data['user_id'].unique())))
    print("{:10} : {:20} : {}".format("Train Data", "No of books",
                                      len(train_data['song'].unique())))
    print()
    print("{:10} : {:20} : {}".format("Test Data", "No of records", len(test_data)))
    print("{:10} : {:20} : {}".format("Test Data", "No of listeners",
                                      len(test_data['user_id'].unique())))
    print("{:10} : {:20} : {}".format("Test Data", "No of books",
                                      len(test_data['song'].unique())))
    print()
    print("{:10} : {:20} : {}".format("Common ", "No of listeners", len(common_listeners)))
    print("{:10} : {:20} : {}".format("Common ", "No of books", len(common_books)))

    train_data_file = os.path.join(train_test_dir, 'train_data.csv')
    train_data.to_csv(train_data_file)
    test_data_file = os.path.join(train_test_dir, 'test_data.csv')
    test_data.to_csv(test_data_file)
    print("Train and Test Data are in ", train_test_dir)

def generate_kfolds_split(train_test_dir, data, kfolds=10, min_no_of_songs=10):
    """Loads data and returns training and test sets by kfolds selection of users"""
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

    min_user_songs_df = user_songs_df[user_songs_df['no_of_songs'] >= min_no_of_songs]
    print("Min no of Songs : ", min_no_of_songs,
          " No of listerners : ",
          len(min_user_songs_df))

    data = pd.merge(min_user_songs_df, users_songs_df, how='inner', on='user_id')

    listeners = np.array(data['user_id'].unique())
    no_of_listeners = len(listeners)
    print("No of listeners : {}".format(no_of_listeners))
    kfolds = KFold(n_splits=kfolds)
    i = 1
    experiments = dict()
    experiments['train'] = dict()
    experiments['test'] = dict()
    for train_indices, test_indices in kfolds.split(listeners):
        #print("%s %s" % (train_indices, test_indices))
        train_listeners_set = set(listeners[train_indices])
        test_listeners_set = set(listeners[test_indices])
        experiments['train'][i] = train_listeners_set
        experiments['test'][i] = test_listeners_set

        train_data = data[data['user_id'].isin(train_listeners_set)]
        test_data = data[data['user_id'].isin(test_listeners_set)]

        train_data_listeners = set(train_data['user_id'].unique())
        test_data_listeners = set(test_data['user_id'].unique())
        common_listeners = train_data_listeners & test_data_listeners

        train_data_books = set(train_data['song'].unique())
        test_data_books = set(test_data['song'].unique())
        common_books = train_data_books & test_data_books

        print()
        print("{} {:10} : {:20} : {}".format(i, "Train Data",
                                             "No of records", len(train_data)))
        print("{} {:10} : {:20} : {}".format(i, "Train Data",
                                             "No of listeners", len(train_data_listeners)))
        print("{} {:10} : {:20} : {}".format(i, "Train Data",
                                             "No of books", len(train_data_books)))
        print()
        print("{} {:10} : {:20} : {}".format(i, "Test Data",
                                             "No of records", len(test_data)))
        print("{} {:10} : {:20} : {}".format(i, "Test Data",
                                             "No of listeners", len(test_data_listeners)))
        print("{} {:10} : {:20} : {}".format(i, "Test Data",
                                             "No of books", len(test_data_books)))
        print()
        print("{} {:10} : {:20} : {}".format(i, "Common ",
                                             "No of listeners", len(common_listeners)))
        print("{} {:10} : {:20} : {}".format(i, "Common ",
                                             "No of books", len(common_books)))
        train_data_file = os.path.join(train_test_dir, str(i) + '_train_data.csv')
        train_data.to_csv(train_data_file, index=False)
        test_data_file = os.path.join(train_test_dir, str(i) + '_test_data.csv')
        test_data.to_csv(test_data_file, index=False)

        i += 1
        print('*'*30)
    print("Train and Test Data are in ", train_test_dir)
    print("Validation of kfold splits:")
    all_listeners = set(listeners)
    all_test_listeners = set()
    for i in experiments['test']:
        all_test_listeners |= set(experiments['test'][i])
    print(all_listeners - all_test_listeners)
    all_train_listeners = set()
    for i in experiments['train']:
        all_train_listeners |= set(experiments['train'][i])
    print(all_listeners - all_train_listeners)

def main():
    """interface to load and split data into train and test"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_test_dir = os.path.join(current_dir, 'train_test_data')
    if not os.path.exists(train_test_dir):
        os.makedirs(train_test_dir)

    # data_dir = os.path.join(current_dir, 'preprocessed_data')
    # data = os.path.join(data_dir, 'user_songs.csv')

    parser = argparse.ArgumentParser(description="Split train and test data")
    parser.add_argument("--random_split",
                        help="Random split data into train and test",
                        action="store_true")
    parser.add_argument("--users_split",
                        help="split users into train and test",
                        action="store_true")
    parser.add_argument("--kfold_split",
                        help="generate cross validation train and test",
                        action="store_true")
    parser.add_argument("--min_no_of_songs",
                        help="min_no_of_songs", type=int)
    parser.add_argument("--test_size",
                        help="test_size ratio", type=float)
    parser.add_argument("--kfolds",
                        help="no of k folds", type=int)
    parser.add_argument("data", help="data used to split into train and test")

    args = parser.parse_args()
    if args.random_split and args.test_size and args.data:
        generate_random_split(train_test_dir, args.data, args.test_size)
    elif args.users_split and args.test_size and args.data:
        if args.min_no_of_songs:
            generate_users_split(train_test_dir, args.data, args.test_size, args.min_no_of_songs)
        else:
            generate_users_split(train_test_dir, args.data, args.test_size)
    elif args.kfold_split and args.kfolds and args.data:
        if args.min_no_of_songs:
            generate_kfolds_split(train_test_dir, args.data, args.kfolds, args.min_no_of_songs)
        else:
            generate_kfolds_split(train_test_dir, args.data, args.kfolds)
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()
