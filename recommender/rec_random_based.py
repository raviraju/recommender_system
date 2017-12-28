"""Module for Random Based Recommender"""
import os
import sys
import logging
from timeit import default_timer
from pprint import pprint
import joblib

import random
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.rec_interface import Recommender
from recommender.rec_interface import load_train_test
from recommender.evaluation import PrecisionRecall

class RandomBasedRecommender(Recommender):
    """Random based recommender system model"""

    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)
        self.model_file = os.path.join(self.model_dir,
                                       'random_based_model.pkl')
    #######################################
    def train(self):
        """train the random based recommender system model"""
        super().train()
    #######################################
    def __generate_top_recommendations(self, user_id, user_interacted_items):
        """pick items in random from train data with equal probability as score"""
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']

        items_train = self.get_all_items(dataset='train')
        random_items = random.sample(items_train,
                                     self.no_of_recs)        
        score = round(1/self.no_of_recs, 2)
        rank = 1
        for item_id in random_items:
            if item_id in user_interacted_items:#to avoid items which user has already aware
                continue
            item_dict = {
                self.user_id_col : user_id,
                self.item_id_col : item_id,
                'score' : score,
                'rank' : rank
            }
            items_to_recommend.append(item_dict)
            rank += 1
        res_df = pd.DataFrame(items_to_recommend, columns=columns)
        # Handle the case where there are no recommendations
        # if res_df.shape[0] == 0:
        #     return None
        # else:
        #     return res_df
        return res_df

    def recommend_items(self, user_id):
        """recommend items for given user_id from test dataset"""
        super().recommend_items(user_id)
        #pprint(self.items_for_evaluation[user_id])

        start_time = default_timer()
        assume_interacted_items = self.items_for_evaluation[user_id]['assume_interacted_items']
        user_recommendations = self.__generate_top_recommendations(user_id,
                                                                   assume_interacted_items)
        recommended_items = list(user_recommendations[self.item_id_col].values)
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated. ",
                                   utilities.convert_sec(end_time - start_time)))
        return recommended_items
    #######################################
    def __recommend_items_to_evaluate(self):
        """recommend items for all users from test dataset"""
        for user_id in self.items_for_evaluation:
            assume_interacted_items = self.items_for_evaluation[user_id]['assume_interacted_items']
            user_recommendations = self.__generate_top_recommendations(user_id,
                                                                       assume_interacted_items)

            recommended_items = list(user_recommendations[self.item_id_col].values)
            self.items_for_evaluation[user_id]['items_recommended'] = recommended_items
        return self.items_for_evaluation

    def evaluate(self, no_of_recs_to_eval):
        """Evaluate trained model for different no of ranked recommendations"""
        super().evaluate(no_of_recs_to_eval)

        start_time = default_timer()
        #Generate recommendations for the users
        self.items_for_evaluation = self.__recommend_items_to_evaluate()
        self.save_items_for_evaluation()

        precision_recall_intf = PrecisionRecall()
        evaluation_results = precision_recall_intf.compute_precision_recall(
            no_of_recs_to_eval, self.items_for_evaluation)
        end_time = default_timer()
        print("{:50}    {}".format("Evaluation Completed. ",
                                   utilities.convert_sec(end_time - start_time)))

        results_file = os.path.join(self.model_dir, 'evaluation_results.json')
        utilities.dump_json_file(evaluation_results, results_file)

        return evaluation_results
    #######################################

def train(results_dir, model_dir, train_test_dir,
          user_id_col, item_id_col,
          no_of_recs=10, hold_out_ratio=0.5):
    """train recommender"""
    train_data, test_data = load_train_test(train_test_dir,
                                            user_id_col,
                                            item_id_col)

    print("Training Recommender...")
    model = RandomBasedRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col,
                                   no_of_recs, hold_out_ratio)
    model.train()
    print('*' * 80)

def evaluate(results_dir, model_dir, train_test_dir,
             user_id_col, item_id_col,
             no_of_recs_to_eval,
             no_of_recs=10, hold_out_ratio=0.5):
    """evaluate recommender"""
    train_data, test_data = load_train_test(train_test_dir,
                                            user_id_col,
                                            item_id_col)

    print("Evaluating Recommender System...")
    model = RandomBasedRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col,
                                   no_of_recs, hold_out_ratio)
    evaluation_results = model.evaluate(no_of_recs_to_eval)
    pprint(evaluation_results)
    print('*' * 80)

def recommend(results_dir, model_dir, train_test_dir,
              user_id_col, item_id_col,
              user_id,
              no_of_recs=10, hold_out_ratio=0.5):
    """recommend items for user"""
    train_data, test_data = load_train_test(train_test_dir,
                                            user_id_col,
                                            item_id_col)

    model = RandomBasedRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col,
                                   no_of_recs, hold_out_ratio)

    recommended_items = model.recommend_items(user_id)
    print("Items recommended for a user with user_id : {}".format(user_id))
    if recommended_items:
        for item in recommended_items:
            print(item)
    else:
        print("No items to recommend")
    print('*' * 80)

def train_eval_recommend(results_dir, model_dir, train_test_dir,
                         user_id_col, item_id_col,
                         no_of_recs_to_eval,
                         no_of_recs=10, hold_out_ratio=0.5):
    """Train Evaluate and Recommend for Item Based Recommender"""
    train_data, test_data = load_train_test(train_test_dir,
                                            user_id_col,
                                            item_id_col)

    print("Training Recommender...")
    model = RandomBasedRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col,
                                   no_of_recs, hold_out_ratio)
    model.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    evaluation_results = model.evaluate(no_of_recs_to_eval)
    pprint(evaluation_results)
    print('*' * 80)

    print("Testing Recommendation for an User")
    items_for_evaluation_file = os.path.join(model_dir, 'items_for_evaluation.json')
    items_for_evaluation = utilities.load_json_file(items_for_evaluation_file)
    users = list(items_for_evaluation.keys())
    user_id = users[0]
    recommended_items = model.recommend_items(user_id)
    print("Items recommended for a user with user_id : {}".format(user_id))
    if recommended_items:
        for item in recommended_items:
            print(item)
    else:
        print("No items to recommend")
    print('*' * 80)
