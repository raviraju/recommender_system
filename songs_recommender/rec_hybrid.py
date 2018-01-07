"""Module for Hybrid of Songs Recommenders"""
import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender import rec_interface as generic_rec_interface
from recommender import rec_item_based_cf as generic_rec_item_based_cf
from recommender import rec_user_based_cf as generic_rec_user_based_cf
from recommender import rec_popularity_based as generic_rec_popularity_based

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

    model_dir = os.path.join(current_dir, 'model/hybrid_based')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    recommenders = {generic_rec_item_based_cf.ItemBasedCFRecommender : 0.5,
                    generic_rec_user_based_cf.UserBasedCFRecommender : 0.5}

    user_id_col = 'user_id'
    item_id_col = 'song'

    no_of_recs = 10
    hold_out_ratio = 0.5
    kwargs = {'no_of_recs':no_of_recs,
              'hold_out_ratio':hold_out_ratio
             }
    no_of_recs_to_eval = [1, 2, 5, 10]

    if args.cross_eval and args.kfolds:
        generic_rec_interface.hybrid_kfold_evaluation(recommenders,
                                                      args.kfolds,
                                                      results_dir, model_dir,
                                                      args.train_data, args.test_data,
                                                      user_id_col, item_id_col,
                                                      no_of_recs_to_eval, **kwargs)
        return
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