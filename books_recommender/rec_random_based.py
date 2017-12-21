"""Module for Random Based Books Recommender"""
import os
import sys
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pprint import pprint
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

from lib import utilities
from recommender.reco_interface import load_train_test
from recommender import random_based

class Books_RandomBasedRecommender(random_based.RandomBasedRecommender):
    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col,
                 no_of_recs=10):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col,
                         no_of_recs)
        self.book_access_time_train = dict()
        self.book_access_time_test = dict()
        
    def derive_stats(self):
        """derive use case specific stats"""
        super().derive_stats()

        LOGGER.debug("Train Data :: Getting Access Time for each User-Item")
        self.book_access_time_train = dict()
        for index, row in self.train_data.iterrows():             
            user = row['learner_id']
            item = row['book_code']
            if user in self.book_access_time_train:
                self.book_access_time_train[user][item] = row['first_access_time']
            else:
                self.book_access_time_train[user] = {item : row['first_access_time']}
        #pprint(self.book_access_time_train)
        book_access_time_train_file = os.path.join(self.model_dir, 'book_access_time_train.json')
        utilities.dump_json_file(self.book_access_time_train, book_access_time_train_file)
        
        LOGGER.debug("Test Data :: Getting Access Time for each User-Item")
        for index, row in self.test_data.iterrows(): 
            user = row['learner_id']
            item = row['book_code']
            if user in self.book_access_time_test:
                self.book_access_time_test[user][item] = row['first_access_time']
            else:
                self.book_access_time_test[user] = {item : row['first_access_time']}        
        #pprint(self.book_access_time_test)
        book_access_time_test_file = os.path.join(self.model_dir, 'book_access_time_test.json')
        utilities.dump_json_file(self.book_access_time_test, book_access_time_test_file)
        
    def load_stats(self):
        """load use case specific stats"""
        super().load_stats()
        
        LOGGER.debug("Train Data :: Loading Access Time for each User-Item")
        book_access_time_train_file = os.path.join(self.model_dir, 'book_access_time_train.json')
        self.book_access_time_train = utilities.load_json_file(book_access_time_train_file)
        
        LOGGER.debug("Test Data :: Loading Access Time for each User-Item")
        book_access_time_test_file = os.path.join(self.model_dir, 'book_access_time_test.json')
        self.book_access_time_test = utilities.load_json_file(book_access_time_test_file)
    
    def get_items(self, user_id, dataset='train'):
        """Get unique items for a given user in sorted order of access time """
        if dataset == "train":
            items_access = self.book_access_time_train[user_id]
        else:#test
            items_access = self.book_access_time_test[user_id]
        #pprint(items_access)
        sorted_items = sorted(items_access.items(), key=lambda p: p[1])
        #pprint(sorted_items)
        items_access_ordered = []
        for item, access in sorted_items:
            items_access_ordered.append(item)
        #print(items_access_ordered)
        #input()
        return items_access_ordered

def train_eval_recommend(results_dir, model_dir, train_test_dir,
                         user_id_col, item_id_col,
                         no_of_recs_to_eval, dataset='test',
                         no_of_recs=10, hold_out_ratio=0.5):
    """Train Evaluate and Recommend for Random Based Recommender"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    print("Training Recommender...")
    model = Books_RandomBasedRecommender(results_dir, model_dir,
                                         train_data, test_data,
                                         user_id_col, item_id_col, no_of_recs)
    model.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    results = model.evaluate(no_of_recs_to_eval, dataset, hold_out_ratio)
    pprint(results)
    print('*' * 80)

    print("Testing Recommendation for an User")
    users = test_data[user_id_col].unique()
    user_id = users[0]
    print("Items recommended for a user with user_id : {}".format(user_id))
    recommended_items = model.recommend_items()
    print()
    if recommended_items:
        for item in recommended_items:
            print(item)
    else:
        print("No items to recommend")
    print('*' * 80)

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
    user_id_col = 'learner_id'
    item_id_col = 'book_code'
    train_test_dir = os.path.join(current_dir, 'train_test_data')

    if args.train:
        random_based.train(results_dir, model_dir, train_test_dir,
                               user_id_col, item_id_col, no_of_recs=no_of_recs)
    elif args.eval:
        no_of_recs_to_eval = [1, 2, 5, 10]
        random_based.evaluate(results_dir, model_dir, train_test_dir,
                                  user_id_col, item_id_col,
                                  no_of_recs_to_eval, dataset='test', no_of_recs=no_of_recs)
    elif args.recommend and args.user_id:
        random_based.recommend(results_dir, model_dir, train_test_dir,
                                   user_id_col, item_id_col,
                                   args.user_id, no_of_recs=no_of_recs)
    else:
        no_of_recs_to_eval = [1, 2, 5, 10]
        train_eval_recommend(results_dir, model_dir, train_test_dir,
                             user_id_col, item_id_col,
                             no_of_recs_to_eval,
                             dataset='test', no_of_recs=no_of_recs)

if __name__ == '__main__':
    main()