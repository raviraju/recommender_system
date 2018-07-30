"""Module to split data into train and test"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def get_filtered_data(events_df, min_no_of_books):
    """apply filtering on data"""
    print("{:10} : {:20} : {}".format("Data", "No of records", len(events_df)))
    print("{:10} : {:20} : {}".format("Data", "No of learners", len(events_df['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Data", "No of books", len(events_df['book_code'].unique())))

    #filtering data to be imported on age
    print("Considering learners whose age range lies in 5-20")
    events_df = events_df[(events_df['age'] >= 5.0) & (events_df['age'] <= 20.0)]
    print("{:10} : {:20} : {}".format("Filtered age", "No of records", len(events_df)))
    print("{:10} : {:20} : {}".format("Filtered age", "No of learners", len(events_df['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Filtered age", "No of books", len(events_df['book_code'].unique())))

    #filtering data to be imported on min_no_of_books
    user_items_df = events_df.groupby('learner_id')\
                             .agg({'book_code' : 'count'})\
                             .rename(columns={'book_code' : 'no_of_books'})\
                             .reset_index()
    user_items_df = user_items_df[user_items_df['no_of_books'] >= min_no_of_books]
    filtered_data = pd.merge(user_items_df, events_df, how='inner', on='learner_id')
    print("No of Books Distribution")
    print(filtered_data['no_of_books'].describe())
    print()
    print("{:10} : {:20} : {}".format("Filtered min_no_of_books>=" + str(min_no_of_books), "No of records", len(filtered_data)))
    print("{:10} : {:20} : {}".format("Filtered min_no_of_books>=" + str(min_no_of_books), "No of learners", len(filtered_data['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Filtered min_no_of_books>=" + str(min_no_of_books), "No of books", len(filtered_data['book_code'].unique())))

    return filtered_data

def generate_users_split(train_test_dir, data, test_size=0.2, min_no_of_books=10):
    """Loads data and returns training and test set by random selection of users"""
    print("Generate Training and Test Data")
    #Read learner_id-book_code-events
    events_df = pd.read_csv(data)

    filtered_data = get_filtered_data(events_df, min_no_of_books)

    learners = filtered_data['learner_id'].unique()
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

    test_data = events_df[events_df['learner_id'].isin(test_learners_set)]
    train_data = events_df[events_df['learner_id'].isin(train_learners_set)]

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

def generate_kfolds_split(train_test_dir, data, kfolds=10, min_no_of_books=10):
    """Loads data and returns training and test sets by kfolds selection of users"""
    print("Generate Training and Test Data")
    #Read learner_id-book_code-events
    events_df = pd.read_csv(data)

    filtered_data = get_filtered_data(events_df, min_no_of_books)

    learners = np.array(filtered_data['learner_id'].unique())
    no_of_learners = len(learners)
    print("No of learners : {}".format(no_of_learners))
    kfolds = KFold(n_splits=kfolds)
    i = 1
    experiments = dict()
    experiments['train'] = dict()
    experiments['test'] = dict()
    for train_indices, test_indices in kfolds.split(learners):
        #print("%s %s" % (train_indices, test_indices))
        train_learners_set = set(learners[train_indices])
        test_learners_set = set(learners[test_indices])
        #print(train, test)
        experiments['train'][i] = train_learners_set
        experiments['test'][i] = test_learners_set

        train_data = events_df[events_df['learner_id'].isin(train_learners_set)]
        test_data = events_df[events_df['learner_id'].isin(test_learners_set)]
        
        '''
        train_data_stats = train_data.groupby(['learner_id'])\
                                     .agg({'book_code' : 'count'})\
                                     .rename(columns={'book_code' : 'no_of_items'})\
                                     .reset_index()
        #print(train_data_stats.head())
        print("train_data : no of items stats:")
        print(train_data_stats['no_of_items'].describe())
        
        test_data_stats = test_data.groupby(['learner_id'])\
                                   .agg({'book_code' : 'count'})\
                                   .rename(columns={'book_code' : 'no_of_items'})\
                                   .reset_index()
        #print(test_data_stats.head())
        print("test_data : no of items stats:")
        print(test_data_stats['no_of_items'].describe())
        '''
        train_data_learners = set(train_data['learner_id'].unique())
        test_data_learners = set(test_data['learner_id'].unique())
        common_learners = train_data_learners & test_data_learners

        train_data_books = set(train_data['book_code'].unique())
        test_data_books = set(test_data['book_code'].unique())
        common_books = train_data_books & test_data_books

        print()
        print("{} {:10} : {:20} : {}".format(i, "Train Data", "No of records", len(train_data)))
        print("{} {:10} : {:20} : {}".format(i, "Train Data", "No of learners", len(train_data_learners)))
        print("{} {:10} : {:20} : {}".format(i, "Train Data", "No of books", len(train_data_books)))
        print()
        print("{} {:10} : {:20} : {}".format(i, "Test Data", "No of records", len(test_data)))
        print("{} {:10} : {:20} : {}".format(i, "Test Data", "No of learners", len(test_data_learners)))
        print("{} {:10} : {:20} : {}".format(i, "Test Data", "No of books", len(test_data_books)))
        print()
        print("{} {:10} : {:20} : {}".format(i, "Common ", "No of learners", len(common_learners)))
        print("{} {:10} : {:20} : {}".format(i, "Common ", "No of books", len(common_books)))
        train_data_file = os.path.join(train_test_dir, str(i) + '_train_data.csv')
        train_data.to_csv(train_data_file, index=False)
        test_data_file = os.path.join(train_test_dir, str(i) + '_test_data.csv')
        test_data.to_csv(test_data_file, index=False)

        i += 1
        print('*'*30)
    print("Train and Test Data are in ", train_test_dir)
    #Validation of kfold splits")
    print("Validation of kfold splits")
    all_learners = set(learners)
    all_test_learners = set()
    for i in experiments['test']:
        all_test_learners |= set(experiments['test'][i])
    print("Learners not considered in Test Data : ", all_learners - all_test_learners)
    all_train_learners = set()
    for i in experiments['train']:
        all_train_learners |= set(experiments['train'][i])
    print("Learners not considered in Train Data : ",all_learners - all_train_learners)

def main():
    """interface to load and split data into train and test"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_test_dir = os.path.join(current_dir, 'train_test_data')
    if not os.path.exists(train_test_dir):
        os.makedirs(train_test_dir)

    parser = argparse.ArgumentParser(description="Split train and test data")
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

    if not args.min_no_of_books:
        min_no_of_books = 10
    else:
        min_no_of_books = args.min_no_of_books

    if args.users_split and args.test_size and args.data:
        train_test_dir = os.path.join(current_dir, '../train_test_data/users_split')
        if not os.path.exists(train_test_dir):
            os.makedirs(train_test_dir)
        generate_users_split(train_test_dir, args.data, args.test_size, min_no_of_books)
    elif args.kfold_split and args.kfolds and args.data:
        train_test_dir = os.path.join(current_dir, '../train_test_data/kfold_split')
        if not os.path.exists(train_test_dir):
            os.makedirs(train_test_dir)
        generate_kfolds_split(train_test_dir, args.data, args.kfolds, min_no_of_books)
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()
