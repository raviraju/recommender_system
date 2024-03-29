"""Module for Item Based CF Articles Recommender"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                #binary_rec     #articles_recommender
import argparse

import logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

import rec_interface as articles_rec_interface
from recommender import rec_interface as generic_rec_interface
from recommender import rec_item_based_cf as generic_rec_item_based_cf

import pandas as pd
pd.set_option('display.max_colwidth', 150)

class ItemBasedCFRecommender(articles_rec_interface.ArticlesRecommender,
                             generic_rec_item_based_cf.ItemBasedCFRecommender):
    """Item based colloborative filtering recommender system model for Articles"""
    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)

def main():
    """Item based colloborative filtering recommender interface"""
    parser = argparse.ArgumentParser(description="Item Based CF Recommender")
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

    parser.add_argument("train_data", help="Train Data")
    parser.add_argument("test_data", help="Test Data")
    parser.add_argument("--meta_data", help="Meta Data")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')

    user_id_col = 'personId'
    item_id_col = 'contentId'
    
    kwargs = dict()
    kwargs['no_of_recs'] = 10
    kwargs['hold_out_strategy'] = 'hold_all'

    # kwargs['hold_out_strategy'] = 'assume_ratio'
    # kwargs['assume_ratio'] = 0.5

    # kwargs['hold_out_strategy'] = 'assume_first_n'
    # kwargs['assume_first_n'] = 5

    # kwargs['hold_out_strategy'] = 'hold_last_n'
    # kwargs['hold_last_n'] = 5

    no_of_recs_to_eval = [5, 10]
    recommender_obj = ItemBasedCFRecommender

    if args.meta_data:
        kwargs['meta_data_file'] = args.meta_data
        kwargs['meta_data_fields'] = ['url']#, 'title']
    
    model_name = 'models/' + kwargs['hold_out_strategy'] + '_item_based_cf'
    model_dir = os.path.join(current_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if args.train:
        generic_rec_interface.train(recommender_obj,
                                    results_dir, model_dir,
                                    args.train_data, args.test_data,
                                    user_id_col, item_id_col,
                                    **kwargs)
    elif args.eval:
        generic_rec_interface.evaluate(recommender_obj,
                                        results_dir, model_dir,
                                        args.train_data, args.test_data,
                                        user_id_col, item_id_col,
                                        no_of_recs_to_eval,
                                        eval_res_file='evaluation_results.json',
                                        **kwargs)
    elif args.recommend and args.user_id:
        generic_rec_interface.recommend(recommender_obj,
                                        results_dir, model_dir,
                                        args.train_data, args.test_data,
                                        user_id_col, item_id_col,
                                        args.user_id, **kwargs)
    else:
        generic_rec_interface.train_eval_recommend(recommender_obj,
                                                results_dir, model_dir,
                                                args.train_data, args.test_data,
                                                user_id_col, item_id_col,
                                                no_of_recs_to_eval, **kwargs)
if __name__ == '__main__':
    main()
