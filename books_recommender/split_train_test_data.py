"""Module to split data into train and test"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def generate_random_split(train_test_dir, data, min_no_of_books=10, test_size=0.2):
    """Loads data and returns training and test set by random split of data"""
    print("Generate Training and Test Data")
    #Read learner_id-book_code-events
    events_df = pd.read_csv(data)
    print("{:10} : {:20} : {}".format("Total", "No of records",
                                      len(events_df)))
    #filtering data to be imported
    events_df = events_df[(events_df['age'] >= 5.0) & (events_df['age'] <= 20.0)]

    #filtering data to be imported
    learner_books_df = events_df.groupby('learner_id')\
                                        .agg({'book_code' : 'count'})\
                                        .rename(columns={'book_code' : 'no_of_books'})\
                                        .reset_index()
    dist = learner_books_df['no_of_books'].describe()
    no_of_books_list = [dist['min'], dist['25%'], dist['50%'], dist['75%'], dist['max']]
    print("Distribution of book_counts (min, 25%, 50%, 75%, max)")
    for no_of_books in no_of_books_list:
        no_of_learners = len(learner_books_df[learner_books_df['no_of_books'] == no_of_books])
        print("No of Books : ", no_of_books,
              " No of learners : ", no_of_learners)
    print()

    learner_books_df = learner_books_df[learner_books_df['no_of_books'] >= min_no_of_books]
    print("Min no of Books : ", min_no_of_books,
          " No of learners : ", len(learner_books_df))
    #split train and test data
    print("{:10} : {:20} : {}".format("Filtered", "No of records",
                                      len(learner_books_df)))

    data = pd.merge(learner_books_df, events_df, how='inner', on='learner_id')
    train_data, test_data = train_test_split(data,
                                             test_size=test_size,
                                             random_state=None)
                                             #random_state=0)
                                             #If int, random_state is the seed used
                                             #by the random number generator
    print()
    print("{:10} : {:20} : {}".format("Data", "No of records", len(data)))
    print("{:10} : {:20} : {}".format("Data", "No of learners", len(data['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Data", "No of books", len(data['book_code'].unique())))
    print()
    print("{:10} : {:20} : {}".format("Train Data", "No of records", len(train_data)))
    print("{:10} : {:20} : {}".format("Train Data", "No of learners",
                                      len(train_data['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Train Data", "No of books",
                                      len(train_data['book_code'].unique())))
    print()
    print("{:10} : {:20} : {}".format("Test Data", "No of records", len(test_data)))
    print("{:10} : {:20} : {}".format("Test Data", "No of learners",
                                      len(test_data['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Test Data", "No of books",
                                      len(test_data['book_code'].unique())))

    common_learners = set(train_data['learner_id'].unique()) & set(test_data['learner_id'].unique())
    common_books = set(train_data['book_code'].unique()) & set(test_data['book_code'].unique())
    print()
    print("{:10} : {:20} : {}".format("Common ", "No of learners", len(common_learners)))
    print("{:10} : {:20} : {}".format("Common ", "No of books", len(common_books)))

    train_data_file = os.path.join(train_test_dir, 'train_data.csv')
    train_data.to_csv(train_data_file, index=False)
    test_data_file = os.path.join(train_test_dir, 'test_data.csv')
    test_data.to_csv(test_data_file, index=False)
    print("Train and Test Data are in ", train_test_dir)

def generate_users_split(train_test_dir, data, test_size=0.2):
    """Loads data and returns training and test set by random selection of users"""
    print("Generate Training and Test Data")
    #Read learner_id-book_code-events
    events_df = pd.read_csv(data)

    #filtering data to be imported
    print("Considering learners whose age range lies in 5-20")
    valid_ages_df = events_df[(events_df['age'] >= 5.0) & (events_df['age'] <= 20.0)]

    learners = valid_ages_df['learner_id'].unique()
    no_of_learners = len(learners)
    no_of_test_learners = int(no_of_learners * test_size)
    #no_of_train_learners = no_of_learners - no_of_test_learners

    learners_set = set(learners)
    test_learners_set = set(np.random.choice(learners, no_of_test_learners, replace=False))
    train_learners_set = learners_set - test_learners_set
    common_learners = train_learners_set & test_learners_set
    print("No of learners : {}".format(len(learners_set)))
    print("No of train learners : {}".format(len(train_learners_set)))
    print("No of test learners : {}".format(len(test_learners_set)))
    print("No of common learners : {}".format(len(common_learners)))

    test_data = valid_ages_df[valid_ages_df['learner_id'].isin(test_learners_set)]
    train_data = valid_ages_df[valid_ages_df['learner_id'].isin(train_learners_set)]

    common_learners = set(train_data['learner_id'].unique()) & set(test_data['learner_id'].unique())
    common_books = set(train_data['book_code'].unique()) & set(test_data['book_code'].unique())

    print()
    print("{:10} : {:20} : {}".format("Train Data", "No of records", len(train_data)))
    print("{:10} : {:20} : {}".format("Train Data", "No of learners",
                                      len(train_data['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Train Data", "No of books",
                                      len(train_data['book_code'].unique())))
    print()
    print("{:10} : {:20} : {}".format("Test Data", "No of records", len(test_data)))
    print("{:10} : {:20} : {}".format("Test Data", "No of learners",
                                      len(test_data['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Test Data", "No of books",
                                      len(test_data['book_code'].unique())))
    print()
    print("{:10} : {:20} : {}".format("Common ", "No of learners", len(common_learners)))
    print("{:10} : {:20} : {}".format("Common ", "No of books", len(common_books)))
    train_data_file = os.path.join(train_test_dir, 'train_data.csv')
    train_data.to_csv(train_data_file, index=False)
    test_data_file = os.path.join(train_test_dir, 'test_data.csv')
    test_data.to_csv(test_data_file, index=False)
    print("Train and Test Data are in ", train_test_dir)

def generate_kfolds_split(train_test_dir, data, kfolds=10):
    """Loads data and returns training and test sets by kfolds selection of users"""
    print("Generate Training and Test Data")
    #Read learner_id-book_code-events
    events_df = pd.read_csv(data)

    #filtering data to be imported
    print("Considering learners whose age range lies in 5-20")
    valid_ages_df = events_df[(events_df['age'] >= 5.0) & (events_df['age'] <= 20.0)]

    learners = np.array(valid_ages_df['learner_id'].unique())
    no_of_learners = len(learners)
    print("No of learners : {}".format(no_of_learners))
    kfolds = KFold(n_splits=kfolds)
    i = 1
    # experiments = dict()
    # experiments['train'] = dict()
    # experiments['test'] = dict()
    for train_indices, test_indices in kfolds.split(learners):
        #print("%s %s" % (train_indices, test_indices))
        train_learners_set = set(learners[train_indices])
        test_learners_set = set(learners[test_indices])
        #print(train, test)
        # experiments['train'][i] = train_learners_set
        # experiments['test'][i] = test_learners_set

        train_data = valid_ages_df[valid_ages_df['learner_id'].isin(train_learners_set)]
        test_data = valid_ages_df[valid_ages_df['learner_id'].isin(test_learners_set)]

        train_data_learners = set(train_data['learner_id'].unique())
        test_data_learners = set(test_data['learner_id'].unique())
        common_learners = train_data_learners & test_data_learners

        train_data_books = set(train_data['book_code'].unique())
        test_data_books = set(test_data['book_code'].unique())
        common_books = train_data_books & test_data_books

        print()
        print("{} {:10} : {:20} : {}".format(i, "Train Data",
                                             "No of records", len(train_data)))
        print("{} {:10} : {:20} : {}".format(i, "Train Data",
                                             "No of learners", len(train_data_learners)))
        print("{} {:10} : {:20} : {}".format(i, "Train Data",
                                             "No of books", len(train_data_books)))
        print()
        print("{} {:10} : {:20} : {}".format(i, "Test Data",
                                             "No of records", len(test_data)))
        print("{} {:10} : {:20} : {}".format(i, "Test Data",
                                             "No of learners", len(test_data_learners)))
        print("{} {:10} : {:20} : {}".format(i, "Test Data",
                                             "No of books", len(test_data_books)))
        print()
        print("{} {:10} : {:20} : {}".format(i, "Common ",
                                             "No of learners", len(common_learners)))
        print("{} {:10} : {:20} : {}".format(i, "Common ",
                                             "No of books", len(common_books)))
        train_data_file = os.path.join(train_test_dir, str(i) + '_train_data.csv')
        train_data.to_csv(train_data_file, index=False)
        test_data_file = os.path.join(train_test_dir, str(i) + '_test_data.csv')
        test_data.to_csv(test_data_file, index=False)

        i += 1
        print('*'*30)
    print("Train and Test Data are in ", train_test_dir)
    #Validation of kfold splits
    # all_learners = set(learners)
    # all_test_learners = set()
    # for i in experiments['test']:
    #     all_test_learners |= set(experiments['test'][i])
    # print(all_learners - all_test_learners)
    # all_train_learners = set()
    # for i in experiments['train']:
    #     all_train_learners |= set(experiments['train'][i])
    # print(all_learners - all_train_learners)

def main():
    """interface to load and split data into train and test"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_test_dir = os.path.join(current_dir, 'train_test_data')
    if not os.path.exists(train_test_dir):
        os.makedirs(train_test_dir)

    #data_dir = os.path.join(current_dir, 'preprocessed_metadata')
    #data = os.path.join(data_dir, 'learner_books_info_close_events.csv')
    #data = os.path.join(data_dir, 'learner_books_info_close_min_3_events.csv')
    #data = os.path.join(data_dir, 'learner_books_info_close_min_10_events.csv')

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
    parser.add_argument("--min_no_of_books",
                        help="min_no_of_books", type=int)
    parser.add_argument("--test_size",
                        help="test_size ratio", type=float)
    parser.add_argument("--kfolds",
                        help="no of k folds", type=int)
    parser.add_argument("data", help="data used to split into train and test")

    args = parser.parse_args()
    if args.random_split and args.test_size and args.data:
        generate_random_split(train_test_dir, args.data, args.min_no_of_books, args.test_size)
    elif args.users_split and args.test_size and args.data:
        generate_users_split(train_test_dir, args.data, args.test_size)
    elif args.kfold_split and args.kfolds and args.data:
        generate_kfolds_split(train_test_dir, args.data, args.kfolds)
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()
