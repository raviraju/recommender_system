"""Module for Random Based Recommender"""
import os
import sys
import logging
from timeit import default_timer
from pprint import pprint

import random
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.rec_interface import Recommender
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

        print()
        print("*"*80)
        print("\tRandom Based : Recommending Random items ...")
        print("*"*80)
    #######################################
    def __generate_top_recommendations(self, user_id, known_interacted_items):
        """pick items in random from train data with equal probability as score"""
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']

        items_train = self.get_all_items(dataset='train')
        random_items = random.sample(items_train, self.no_of_recs)        
        score = round(1/self.no_of_recs, 2)
        rank = 1
        for item_id in random_items:
            if not self.allow_recommending_known_items and item_id in known_interacted_items:#to avoid items which user has already aware
                continue
            if rank > self.no_of_recs:#limit no of recommendations
                break
            item_dict = {
                self.item_id_col : item_id,
                'score' : round(score, 3),
                'rank' : rank
            }
            items_to_recommend.append(item_dict)
            rank += 1
        if len(items_to_recommend) > 0:
            items_to_recommend_df = pd.DataFrame(items_to_recommend)
        else:
            items_to_recommend_df = pd.DataFrame(columns = [self.item_id_col, 'score', 'rank'])
        return items_to_recommend_df

    def recommend_items(self, user_id):
        """recommend items for given user_id from test dataset"""
        super().recommend_items(user_id)

        start_time = default_timer()
        known_interacted_items = self.items_for_evaluation[user_id]['known_interacted_items']        
        items_to_recommend_df = self.__generate_top_recommendations(user_id, known_interacted_items)
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated. ",
                                   utilities.convert_sec(end_time - start_time)))
        return items_to_recommend_df
    #######################################
    def __recommend_items_to_evaluate(self):
        """recommend items for all users from test dataset"""
        for user_id in self.items_for_evaluation:
            known_interacted_items = self.items_for_evaluation[user_id]['known_interacted_items']            
            items_to_recommend_df = self.__generate_top_recommendations(user_id, known_interacted_items)
            recommended_items_dict = items_to_recommend_df.set_index(self.item_id_col).to_dict('index')
            
            self.items_for_evaluation[user_id]['items_recommended'] = list(recommended_items_dict.keys())
            self.items_for_evaluation[user_id]['items_recommended_score'] = recommended_items_dict

            items_to_be_interacted_set = set(self.items_for_evaluation[user_id]['items_to_be_interacted'])
            items_recommended_set = set(self.items_for_evaluation[user_id]['items_recommended'])
            correct_recommendations = items_to_be_interacted_set.intersection(items_recommended_set)
            no_of_correct_recommendations = len(correct_recommendations)
            self.items_for_evaluation[user_id]['no_of_correct_recommendations'] = no_of_correct_recommendations
            self.items_for_evaluation[user_id]['correct_recommendations'] = list(correct_recommendations)
        return self.items_for_evaluation

    def evaluate(self, no_of_recs_to_eval, eval_res_file='evaluation_results.json'):
        """Evaluate trained model for different no of ranked recommendations"""
        super().evaluate(no_of_recs_to_eval, eval_res_file)

        start_time = default_timer()
        #Generate recommendations for the users
        self.items_for_evaluation = self.__recommend_items_to_evaluate()
        self.save_items_for_evaluation()

        precision_recall_intf = PrecisionRecall()
        evaluation_results = precision_recall_intf.compute_precision_recall(
            no_of_recs_to_eval, self.items_for_evaluation, self.items_all)
        end_time = default_timer()
        print("{:50}    {}".format("Evaluation Completed. ",
                                   utilities.convert_sec(end_time - start_time)))

        results_file = os.path.join(self.model_dir, eval_res_file)
        utilities.dump_json_file(evaluation_results, results_file)

        return evaluation_results
    #######################################
