"""Module for Random Based Songs Recommender"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import random_based

def main():
    """Random based recommender interface"""
    parser = argparse.ArgumentParser(description="Random Based Recommender")
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
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_dir = os.path.join(current_dir, 'model/random_based')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    no_of_recs = 10
    user_id_col = 'user_id'
    item_id_col = 'song'
    train_test_dir = os.path.join(current_dir, 'train_test_data')

    if args.train:
        random_based.train(results_dir, model_dir, train_test_dir,
                               user_id_col, item_id_col, no_of_recs=no_of_recs)
    elif args.eval:
        no_of_recs_to_eval = [1, 2, 5]
        random_based.evaluate(results_dir, model_dir, train_test_dir,
                                  user_id_col, item_id_col,
                                  no_of_recs_to_eval, dataset='test', no_of_recs=no_of_recs)
    elif args.recommend and args.user_id:
        random_based.recommend(results_dir, model_dir, train_test_dir,
                                   user_id_col, item_id_col,
                                   args.user_id, no_of_recs=no_of_recs)
    else:
        no_of_recs_to_eval = [1, 2, 5]
        random_based.train_eval_recommend(results_dir, model_dir, train_test_dir,
                                              user_id_col, item_id_col,
                                              no_of_recs_to_eval,
                                              dataset='test', no_of_recs=no_of_recs)

if __name__ == '__main__':
    main()
