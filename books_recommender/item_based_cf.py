"""Module for Item Based CF Books Recommender"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import item_based_cf_opt

def main():
    """Item based recommender interface"""
    parser = argparse.ArgumentParser(description="Item Based Recommender")
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
    model_dir = os.path.join(current_dir, 'model/item_based')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    no_of_recs = 10
    user_id_col = 'learner_id'
    item_id_col = 'book_code'
    train_test_dir = os.path.join(current_dir, 'train_test_data')

    if args.train:
        item_based_cf_opt.train(results_dir, model_dir, train_test_dir,
                                user_id_col, item_id_col, no_of_recs=no_of_recs)
    elif args.eval:
        no_of_recs_to_eval = [1, 2, 5, 10]
        item_based_cf_opt.evaluate(results_dir, model_dir, train_test_dir,
                                   user_id_col, item_id_col,
                                   no_of_recs_to_eval, dataset='test',
                                   no_of_recs=no_of_recs, hold_out_ratio=0.5)
    elif args.recommend and args.user_id:
        item_based_cf_opt.recommend(results_dir, model_dir, train_test_dir,
                                    user_id_col, item_id_col,
                                    args.user_id, no_of_recs=no_of_recs)
    else:
        no_of_recs_to_eval = [1, 2, 5, 10]
        item_based_cf_opt.train_eval_recommend(results_dir, model_dir, train_test_dir,
                                               user_id_col, item_id_col,
                                               no_of_recs_to_eval, dataset='test',
                                               no_of_recs=no_of_recs, hold_out_ratio=0.5)

if __name__ == '__main__':
    main()
