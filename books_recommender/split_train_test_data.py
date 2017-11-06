"""Module to split data into train and test"""
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_random_split(train_test_dir, data, min_no_of_books = 10, test_size=0.2):
    """Loads data and returns training and test set"""
    print("Generate Training and Test Data")
    #Read learner_id-book_code-view_count triplets
    learners_books_df = pd.read_csv(data)
    print("{:10} : {:20} : {}".format("Total", "No of records",
                                      len(learners_books_df)))
    #filtering data to be imported
    learner_books_df = learners_books_df.groupby('learner_id')\
                                        .agg({'book_code' : 'count'})\
                                        .rename(columns={'book_code' : 'no_of_books'})\
                                        .reset_index()
    dist = learner_books_df['no_of_books'].describe()
    no_of_books_list = [dist['min'], dist['25%'], dist['50%'], dist['75%'], dist['max']]
    print("Distribution of book_counts (min, 25%, 50%, 75%, max)")
    for no_of_books in no_of_books_list:
        print("No of Books : ", no_of_books,
              " No of learners : ", len(learner_books_df[learner_books_df['no_of_books'] == no_of_books]))
    print()
    
    learner_books_df = learner_books_df[learner_books_df['no_of_books'] >= min_no_of_books]
    print("Min no of Books : ", min_no_of_books,
          " No of learners : ", len(learner_books_df))
    #split train and test data
    print("{:10} : {:20} : {}".format("Filtered", "No of records",
                                      len(learner_books_df)))
    
    data = pd.merge(learner_books_df, learners_books_df, how='inner', on='learner_id')
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
    print("{:10} : {:20} : {}".format("Train Data", "No of learners", len(train_data['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Train Data", "No of books", len(train_data['book_code'].unique())))
    print()
    print("{:10} : {:20} : {}".format("Test Data", "No of records", len(test_data)))
    print("{:10} : {:20} : {}".format("Test Data", "No of learners", len(test_data['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Test Data", "No of books", len(test_data['book_code'].unique())))

    train_data_file = os.path.join(train_test_dir, 'train_data.csv')
    train_data.to_csv(train_data_file, index=False)
    test_data_file = os.path.join(train_test_dir, 'test_data.csv')
    test_data.to_csv(test_data_file, index=False)
    print("Train and Test Data are in ", train_test_dir)

def generate_users_split(train_test_dir, data, min_no_of_books = 10, test_size=0.2):
    """Loads data and returns training and test set"""
    print("Generate Training and Test Data")
    #Read learner_id-book_code-view_count triplets
    learners_books_df = pd.read_csv(data)
    print("{:10} : {:20} : {}".format("Total", "No of records",
                                      len(learners_books_df)))
    #filtering data to be imported
    learner_books_df = learners_books_df.groupby('learner_id')\
                                        .agg({'book_code' : 'count'})\
                                        .rename(columns={'book_code' : 'no_of_books'})\
                                        .reset_index()
    dist = learner_books_df['no_of_books'].describe()
    no_of_books_list = [dist['min'], dist['25%'], dist['50%'], dist['75%'], dist['max']]
    print("Distribution of book_counts (min, 25%, 50%, 75%, max)")
    for no_of_books in no_of_books_list:
        print("No of Books : ", no_of_books,
              " No of learners : ", len(learner_books_df[learner_books_df['no_of_books'] == no_of_books]))
    print()
    
    learner_books_df = learner_books_df[learner_books_df['no_of_books'] >= min_no_of_books]
    print("Min no of Books : ", min_no_of_books,
          " No of learners : ", len(learner_books_df))
    #split train and test data
    print("{:10} : {:20} : {}".format("Filtered", "No of records",
                                      len(learner_books_df)))
    
    #data = pd.merge(learner_books_df, learners_books_df, how='inner', on='learner_id')
    train_users, test_users = train_test_split(learners_books_df,
                                               test_size=test_size,
                                               random_state=None)
                                               #random_state=0)
                                               #If int, random_state is the seed used
                                               #by the random number generator
    train_data = pd.merge(train_users, learners_books_df, how='inner', on='learner_id')
    test_data = pd.merge(test_users, learners_books_df, how='inner', on='learner_id')
    
    print()
    print("{:10} : {:20} : {}".format("Train Data", "No of records", len(train_data)))
    print("{:10} : {:20} : {}".format("Train Data", "No of learners", len(train_data['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Train Data", "No of books", len(train_data['book_code'].unique())))
    print()
    print("{:10} : {:20} : {}".format("Test Data", "No of records", len(test_data)))
    print("{:10} : {:20} : {}".format("Test Data", "No of learners", len(test_data['learner_id'].unique())))
    print("{:10} : {:20} : {}".format("Test Data", "No of books", len(test_data['book_code'].unique())))

    train_data_file = os.path.join(train_test_dir, 'train_data.csv')
    train_data.to_csv(train_data_file, index=False)
    test_data_file = os.path.join(train_test_dir, 'test_data.csv')
    test_data.to_csv(test_data_file, index=False)
    print("Train and Test Data are in ", train_test_dir)

def main():
    """interface to load and split data into train and test"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_test_dir = os.path.join(current_dir, 'train_test_data')
    if not os.path.exists(train_test_dir):
        os.makedirs(train_test_dir)

    data_dir = os.path.join(current_dir, 'preprocessed_data')
    data = os.path.join(data_dir, 'learner_books_close_min_events.csv')
   
    parser = argparse.ArgumentParser(description="Split train and test data")
    parser.add_argument("--random_split",
                        help="Random split data into train and test",
                        action="store_true")
    parser.add_argument("--users_split",
                        help="split users into train and test",
                        action="store_true")
    parser.add_argument("min_no_of_books",
                        help="min_no_of_books", type=int)    
    parser.add_argument("test_size",
                        help="test_size ratio", type=float)

    args = parser.parse_args()
    if args.random_split and args.test_size:
        generate_random_split(train_test_dir, data, args.min_no_of_books, args.test_size)
    elif args.users_split and args.test_size:
        generate_users_split(train_test_dir, data, args.test_size)
        pass
    else:
        print("Invalid option")    

if __name__ == "__main__":
    main()
