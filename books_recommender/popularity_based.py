"""Module for Popularity Based Books Recommender"""
import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.popularity_based import *

def generate_train_test(model_dir, learner_books_file):
    """Loads data and returns training and test set"""
    #Read learner_id-book_code-view_count triplets
    learner_books_df = pd.read_csv(learner_books_file)
    #filtering data to be imported
    #split train and test data
    print("{:30} : {}".format("No of records in data", len(learner_books_df)))
    train_data, test_data = train_test_split(learner_books_df,
                                             test_size=0.20,
                                             random_state=None)
                                             #random_state=0)
                                             #If int, random_state is the seed used
                                             #by the random number generator
    print("{:30} : {}".format("No of records in train_data", len(train_data)))
    print("{:30} : {}".format("No of records in test_data", len(test_data)))
    train_data_file = os.path.join(model_dir, 'train_data.csv')
    train_data.to_csv(train_data_file)
    test_data_file = os.path.join(model_dir, 'test_data.csv')
    test_data.to_csv(test_data_file)
    return train_data, test_data

def main():
    """Popularity based recommender interface"""
    parser = argparse.ArgumentParser(description="Popularity Based Recommender")
    parser.add_argument("--train",
                        help="Train Model",
                        action="store_true")
    parser.add_argument("--eval",
                        help="Evaluate Trained Model",
                        action="store_true")
    parser.add_argument("--recommend",
                        help="Recommend Items for a User",
                        action="store_true")
    parser.add_argument("--user_id",
                        help="User Id to recommend items")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'preprocessed_data')
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_dir = os.path.join(current_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data = os.path.join(data_dir, 'learner_books.csv')
    if args.train:
        train_data, test_data = generate_train_test(model_dir, data)
        train(train_data, test_data,
              'learner_id', 'book_code',
              results_dir, model_dir)
    elif args.eval:
        no_of_recs_to_eval = [10, 20]
        sample_test_users_percentage = 1
        evaluate('learner_id', 'book_code',
                 results_dir, model_dir,
                 no_of_recs_to_eval, sample_test_users_percentage)
    elif args.recommend and args.user_id:
        recommend(args.user_id, 'learner_id', 'book_code',
                  results_dir, model_dir)
    else:
        train_data, test_data = generate_train_test(model_dir, data)
        no_of_recs_to_eval = [10, 20]
        sample_test_users_percentage = 1
        train_eval_recommend(train_data, test_data,
                             'learner_id', 'book_code',
                             results_dir, model_dir,
                             no_of_recs_to_eval,
                             sample_test_users_percentage)

if __name__ == '__main__':
    main()
