"""Module for Popularity Based Books Recommender"""
import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import rec_interface as generic_rec_interface
from recommender import rec_popularity_based as generic_rec_popularity_based
import rec_interface as books_rec_interface

class PopularityBasedRecommender(books_rec_interface.BooksRecommender,
                                 generic_rec_popularity_based.PopularityBasedRecommender):
    """Popularity based recommender system model for Books"""
    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)

def main():
    """Popularity based recommender interface"""
    parser = argparse.ArgumentParser(description="User Based Recommender")
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
    parser.add_argument("hold_out_strategy",
                        help="assume_ratio/assume_first_n/hold_last_n")
    parser.add_argument("hold_out_value",
                        help="assume_ratio=0.5/assume_first_n=5/hold_last_n=5")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')

    user_id_col = 'learner_id'
    item_id_col = 'book_code'

    kwargs = dict()
    kwargs['no_of_recs'] = 150 # max no_of_books read is 144

    kwargs['hold_out_strategy'] = args.hold_out_strategy
    if kwargs['hold_out_strategy'] == 'assume_ratio':
        kwargs['assume_ratio'] = float(args.hold_out_value)
    elif kwargs['hold_out_strategy'] == 'assume_first_n':
        kwargs['first_n'] = int(args.hold_out_value)
    elif kwargs['hold_out_strategy'] == 'hold_last_n':
        kwargs['last_n'] = int(args.hold_out_value)
    else:
        print("Invalid hold_out_strategy {} chosen".format(args.hold_out_strategy))
        exit(-1)

    model_name = 'model/' + kwargs['hold_out_strategy'] + '_item_based_cf'
    model_dir = os.path.join(current_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    no_of_recs_to_eval = [5, 6, 7, 8, 9, 10]
    recommender_obj = PopularityBasedRecommender

    model_name_prefix = 'model/' + kwargs['hold_out_strategy']
    user_features_configs = [[], ['learner_gender'], ['age'], ['learner_gender', 'age']]
    for user_features in user_features_configs:
        user_features_str = '_'.join(user_features)
        model_dir = os.path.join(current_dir, model_name_prefix + '_pop_based_' + user_features_str)
        print(model_dir)
        kwargs['user_features'] = user_features
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if args.cross_eval and args.kfolds:
            generic_rec_interface.kfold_evaluation(recommender_obj,
                                                   args.kfolds,
                                                   results_dir, model_dir,
                                                   args.train_data, args.test_data,
                                                   user_id_col, item_id_col,
                                                   no_of_recs_to_eval, **kwargs)
            continue
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
            # generic_rec_interface.recommend(recommender_obj,
            #                                 results_dir, model_dir,
            #                                 args.train_data, args.test_data,
            #                                 user_id_col, item_id_col,
            #                                 args.user_id, **kwargs)
            # metadata_fields = None
            metadata_fields = ['T_BOOK_NAME', 'T_KEYWORD', 'T_AUTHOR']
            books_rec_interface.recommend(recommender_obj,
                                          results_dir, model_dir,
                                          args.train_data, args.test_data,
                                          user_id_col, item_id_col,
                                          args.user_id, metadata_fields, **kwargs)
        else:
            generic_rec_interface.train_eval_recommend(recommender_obj,
                                                       results_dir, model_dir,
                                                       args.train_data, args.test_data,
                                                       user_id_col, item_id_col,
                                                       no_of_recs_to_eval, **kwargs)

if __name__ == '__main__':
    main()
