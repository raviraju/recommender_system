"""Module for Random Based Books Recommender"""
import os
import sys
import argparse
import logging
from pprint import pprint

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rec_interface as books_rec_interface
from lib import utilities
from recommender.rec_interface import load_train_test
from recommender import rec_random_based

class RandomBasedRecommender(books_rec_interface.BooksRecommender,
                             rec_random_based.RandomBasedRecommender):
    """Random based recommender system model for Books"""

    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)

def train(results_dir, model_dir, train_test_dir,
          user_id_col, item_id_col, **kwargs):
    """train recommender"""
    train_data, test_data = load_train_test(train_test_dir,
                                            user_id_col,
                                            item_id_col)

    print("Training Recommender...")
    model = RandomBasedRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col, **kwargs)
    model.train()
    print('*' * 80)

def evaluate(results_dir, model_dir, train_test_dir,
             user_id_col, item_id_col, **kwargs):
    """evaluate recommender"""
    train_data, test_data = load_train_test(train_test_dir,
                                            user_id_col,
                                            item_id_col)

    print("Evaluating Recommender System...")
    model = RandomBasedRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col, **kwargs)
    evaluation_results = model.evaluate(kwargs['no_of_recs_to_eval'])
    pprint(evaluation_results)
    print('*' * 80)

def train_eval_recommend(results_dir, model_dir, train_test_dir,
                         user_id_col, item_id_col, **kwargs):
    """Train Evaluate and Recommend for User Based Recommender"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    print("Training Recommender...")
    model = RandomBasedRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col, **kwargs)
    model.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    results = model.evaluate(kwargs['no_of_recs_to_eval'])
    pprint(results)
    print('*' * 80)

    print("Testing Recommendation for an User")
    eval_items_file = os.path.join(model_dir, 'items_for_evaluation.json')
    eval_items = utilities.load_json_file(eval_items_file)
    users = list(eval_items.keys())
    user_id = users[0]
    recommended_items = model.recommend_items(user_id)
    print("Items recommended for a user with user_id : {}".format(user_id))
    if recommended_items:
        for item in recommended_items:
            print(item)
    else:
        print("No items to recommend")
    print('*' * 80)

def recommend(results_dir, model_dir, train_test_dir,
              user_id_col, item_id_col, user_id, **kwargs):
    """recommend items for user"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    model = RandomBasedRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col, **kwargs)

    metadata_fields = kwargs['metadata_fields']

    eval_items_file = os.path.join(model_dir, 'items_for_evaluation.json')
    eval_items = utilities.load_json_file(eval_items_file)
    if user_id in eval_items:
        assume_interacted_items = eval_items[user_id]['assume_interacted_items']
        items_interacted = eval_items[user_id]['items_interacted']

        print("Assumed Item interactions for a user with user_id : {}".format(user_id))
        for item in assume_interacted_items:
            print(item)
            if metadata_fields is not None:
                item_profile = books_rec_interface.get_item_profile(test_data, item)
                item_name_tokens, item_author, item_keywords = item_profile
                print("\t item_name_tokens : {}".format(item_name_tokens))
                print("\t item_author : {}".format(item_author))
                print("\t item_keywords : {}".format(item_keywords))
                print()

                record = test_data[test_data[item_id_col] == item]
                if not record.empty:
                    for field in metadata_fields:
                        print("\t {} : {}".format(field, record[field].values[0]))
                print('\t '+ '#'*30)

        print()
        print("Items to be interacted for a user with user_id : {}".format(user_id))
        for item in items_interacted:
            print(item)
            if metadata_fields is not None:
                item_profile = books_rec_interface.get_item_profile(test_data, item)
                item_name_tokens, item_author, item_keywords = item_profile
                print("\t item_name_tokens : {}".format(item_name_tokens))
                print("\t item_author : {}".format(item_author))
                print("\t item_keywords : {}".format(item_keywords))
                print()

                record = test_data[test_data[item_id_col] == item]
                if not record.empty:
                    for field in metadata_fields:
                        print("\t {} : {}".format(field, record[field].values[0]))
                print('\t '+ '#'*30)

        print()
        print("Items recommended for a user with user_id : {}".format(user_id))
        recommended_items = model.recommend_items(user_id)
        print()
        if recommended_items:
            for recommended_item in recommended_items:
                print(recommended_item)
                if metadata_fields is not None:
                    item_profile = books_rec_interface.get_item_profile(train_data,
                                                                        recommended_item)
                    item_name_tokens, item_author, item_keywords = item_profile
                    print("\t item_name_tokens : {}".format(item_name_tokens))
                    print("\t item_author : {}".format(item_author))
                    print("\t item_keywords : {}".format(item_keywords))
                    print()

                    record = train_data[train_data[item_id_col] == recommended_item]
                    if not record.empty:
                        for field in metadata_fields:
                            print("\t {} : {}".format(field, record[field].values[0]))
                    for interacted_item in items_interacted:
                        score = books_rec_interface.get_similarity_score(train_data,
                                                                         test_data,
                                                                         recommended_item,
                                                                         interacted_item)
                        print("\t {:20s} | {:20s} | {}".format('recommended_item',
                                                               'interacted_item',
                                                               'score'))
                        print("\t {:20s} | {:20s} | {}".format(recommended_item,
                                                               interacted_item,
                                                               score))
                        print()
                    print('\t '+ '#'*30)
        else:
            print("No items to recommend")
        print('*' * 80)
    else:
        print("""Cannot generate recommendations as either
              items assumed to be interacted or items held out are None""")

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
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_dir = os.path.join(current_dir, 'model/random_based')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    user_id_col = 'learner_id'
    item_id_col = 'book_code'
    train_test_dir = os.path.join(current_dir, 'train_test_data')

    no_of_recs = 10
    hold_out_ratio = 0.5
    kwargs = {'no_of_recs':no_of_recs,
              'hold_out_ratio':hold_out_ratio
             }

    #metadata_fields = None
    metadata_fields = ['T_BOOK_NAME', 'T_KEYWORD', 'T_AUTHOR']

    if args.train:
        train(results_dir, model_dir, train_test_dir,
              user_id_col, item_id_col, **kwargs)
    elif args.eval:
        kwargs['no_of_recs_to_eval'] = [1, 2, 5, 10]
        evaluate(results_dir, model_dir, train_test_dir,
                 user_id_col, item_id_col, **kwargs)
    elif args.recommend and args.user_id:
        kwargs['metadata_fields'] = metadata_fields
        recommend(results_dir, model_dir, train_test_dir,
                  user_id_col, item_id_col, args.user_id, **kwargs)
    else:
        kwargs['no_of_recs_to_eval'] = [1, 2, 5, 10]
        train_eval_recommend(results_dir, model_dir, train_test_dir,
                             user_id_col, item_id_col, **kwargs)

if __name__ == '__main__':
    main()
