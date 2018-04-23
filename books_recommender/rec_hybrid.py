"""Module for Hybrid of Books Recommenders"""
import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender import rec_interface as generic_rec_interface
from books_recommender import rec_item_based_cf as books_rec_item_based_cf
from books_recommender import rec_user_based_cf as books_rec_user_based_cf
from books_recommender import rec_popularity_based as books_rec_popularity_based
from rec_hybrid_user_based_cf_age_itp import Hybrid_UserBased_CF_AgeItp_Recommender
from rec_content_based import ContentBasedRecommender

def main():
    """Hybrid of Songs Recommenders interface"""
    parser = argparse.ArgumentParser(description="Hybrid of Songs Recommender")
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
    parser.add_argument("--cross_eval",
                        help="Cross Evaluate Trained Model",
                        action="store_true")
    parser.add_argument("--kfolds",
                        help="No of kfold datasets to consider",
                        type=int)
    parser.add_argument("train_data",
                        help="Train Data")
    parser.add_argument("test_data",
                        help="Test Data")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    
    user_id_col = 'learner_id'
    item_id_col = 'book_code'

    kwargs = dict()
    kwargs['no_of_recs'] = 150 # max no_of_books read is 144

    # kwargs['hold_out_strategy'] = 'hold_out_ratio'
    # kwargs['hold_out_ratio'] = 0.5

    # kwargs['hold_out_strategy'] = 'assume_first_n'
    # kwargs['first_n'] = 5 #each user has atleast 10 items interacted, so there shall be equal split if no_of_items = 10

    kwargs['hold_out_strategy'] = 'hold_last_n'
    kwargs['last_n'] = 5 #each user has atleast 10 items interacted, so there shall be equal split if no_of_items = 10

    no_of_recs_to_eval = [5, 6, 7, 8, 9, 10]

    configs = [
        {
            'model_dir_name' : 'model/hybrid_item_cf_user_cf',
            'recommenders' : {
                books_rec_item_based_cf.ItemBasedCFRecommender : 0.5,
                books_rec_user_based_cf.UserBasedCFRecommender : 0.5}
        },
        {
            'model_dir_name' : 'model/hybrid_item_cf_content',
            'recommenders' : {
                books_rec_item_based_cf.ItemBasedCFRecommender : 0.5,
                ContentBasedRecommender : 0.5}
        },
        {
            'model_dir_name' : 'model/hybrid_user_cf_content',
            'recommenders' : {
                books_rec_user_based_cf.UserBasedCFRecommender : 0.5,
                ContentBasedRecommender : 0.5}
        },
        {
            'model_dir_name' : 'model/hybrid_item_cf_user_age',
            'recommenders' : {
                books_rec_item_based_cf.ItemBasedCFRecommender : 0.5,
                Hybrid_UserBased_CF_AgeItp_Recommender : 0.5}
        },
        {
            'model_dir_name' : 'model/hybrid_item_cf_content_user_age',
            'recommenders' : {
                books_rec_item_based_cf.ItemBasedCFRecommender : 0.35,
                ContentBasedRecommender : 0.35,
                Hybrid_UserBased_CF_AgeItp_Recommender : 0.3}
        }
    ]
    kwargs['age_or_itp'] = 'age'
    
    for config in configs:
        model_dir = os.path.join(current_dir, config['model_dir_name'])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        recommenders = config['recommenders']

    
        if args.cross_eval and args.kfolds:
            generic_rec_interface.hybrid_kfold_evaluation(recommenders,
                                                          args.kfolds,
                                                          results_dir, model_dir,
                                                          args.train_data, args.test_data,
                                                          user_id_col, item_id_col,
                                                          no_of_recs_to_eval, **kwargs)
            continue
        if args.train:
            generic_rec_interface.hybrid_train(recommenders,
                                               results_dir, model_dir,
                                               args.train_data, args.test_data,
                                               user_id_col, item_id_col,
                                               **kwargs)
        elif args.eval:
            generic_rec_interface.hybrid_evaluate(recommenders,
                                                  results_dir, model_dir,
                                                  args.train_data, args.test_data,
                                                  user_id_col, item_id_col,
                                                  no_of_recs_to_eval,
                                                  eval_res_file='evaluation_results.json',
                                                  **kwargs)
        elif args.recommend and args.user_id:
            generic_rec_interface.hybrid_recommend(recommenders,
                                                   results_dir, model_dir,
                                                   args.train_data, args.test_data,
                                                   user_id_col, item_id_col,
                                                   args.user_id, **kwargs)
        else:
            generic_rec_interface.hybrid_train_eval_recommend(recommenders,
                                                              results_dir, model_dir,
                                                              args.train_data, args.test_data,
                                                              user_id_col, item_id_col,
                                                              no_of_recs_to_eval, **kwargs)

if __name__ == '__main__':
    main()
