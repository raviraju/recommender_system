"""Module for Popularity Based Books Recommender"""
import os
import sys
import argparse
import logging
from pprint import pprint

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
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
    """User based recommender interface"""
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
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)

    user_id_col = 'learner_id'
    item_id_col = 'book_code'

    no_of_recs = 10
    hold_out_ratio = 0.5
    kwargs = {'no_of_recs':no_of_recs,
              'hold_out_ratio':hold_out_ratio
             }

    #metadata_fields = None
    metadata_fields = ['T_BOOK_NAME', 'T_KEYWORD', 'T_AUTHOR']

    user_features_configs = [[], ['learner_gender'], ['age'], ['learner_gender', 'age']]
    for user_features in user_features_configs:
        user_features_str = '_'.join(user_features)
        model_dir = os.path.join(current_dir, 'model/pop_based_' + user_features_str)
        print(model_dir)
        kwargs['user_features'] = user_features
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if args.cross_eval and args.kfolds:
            kfold_experiments = dict()
            for kfold in range(args.kfolds):
                kfold_exp = kfold+1
                train_data_file = os.path.join(args.train_data, str(kfold_exp) + '_train_data.csv')
                test_data_file = os.path.join(args.train_data, str(kfold_exp) + '_test_data.csv')
                print("Loading...")
                print(train_data_file)
                print(test_data_file)
                train_data, test_data = generic_rec_interface.load_train_test(train_data_file,
                                                                              test_data_file,
                                                                              user_id_col,
                                                                              item_id_col)
                kfold_model_dir = os.path.join(model_dir,
                                               'kfold_experiments',
                                               'kfold_exp_' + str(kfold_exp))
                if not os.path.exists(kfold_model_dir):
                    os.makedirs(kfold_model_dir)
                recommender = PopularityBasedRecommender(results_dir, kfold_model_dir,
                                                         train_data, test_data,
                                                         user_id_col, item_id_col, **kwargs)
                generic_rec_interface.train(recommender)
                no_of_recs_to_eval = [1, 2, 5, 10]
                kfold_eval_file = 'kfold_exp_' + str(kfold_exp) + '_evaluation.json'
                evaluation_results = generic_rec_interface.evaluate(recommender,
                                                                    no_of_recs_to_eval,
                                                                    eval_res_file=kfold_eval_file)
                kfold_experiments[kfold_exp] = evaluation_results

            avg_kfold_exp_res = generic_rec_interface.get_avg_kfold_exp_res(kfold_experiments)
            print('average of kfold evaluation results')
            pprint(avg_kfold_exp_res)
            results_file = os.path.join(model_dir, 'kfold_experiments', 'kfold_evaluation.json')
            utilities.dump_json_file(avg_kfold_exp_res, results_file)
            continue#proceed for next user_features config

        train_data, test_data = generic_rec_interface.load_train_test(args.train_data,
                                                                      args.test_data,
                                                                      user_id_col,
                                                                      item_id_col)
        recommender = PopularityBasedRecommender(results_dir, model_dir,
                                                 train_data, test_data,
                                                 user_id_col, item_id_col, **kwargs)
        if args.train:
            generic_rec_interface.train(recommender)
        elif args.eval:
            no_of_recs_to_eval = [1, 2, 5, 10]
            generic_rec_interface.evaluate(recommender, no_of_recs_to_eval)
        elif args.recommend and args.user_id:
            #generic_rec_interface.recommend(recommender, model_dir, args.user_id)
            books_rec_interface.recommend(recommender, model_dir, args.user_id,
                                          train_data, test_data,
                                          item_id_col, metadata_fields)
        else:
            no_of_recs_to_eval = [1, 2, 5, 10]
            generic_rec_interface.train_eval_recommend(recommender, model_dir, no_of_recs_to_eval)

if __name__ == '__main__':
    main()
