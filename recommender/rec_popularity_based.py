"""Module for Popularity Based Recommender"""
import os
import sys
import logging
from timeit import default_timer

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.rec_interface import Recommender
from recommender.evaluation import PrecisionRecall

class PopularityBasedRecommender(Recommender):
    """Popularity based recommender system model"""

    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)
        self.model_file = os.path.join(self.model_dir,
                                       'popularity_based_model.pkl')
        #print(self.model_file)
        self.user_features = kwargs['user_features']
        #print(self.user_features)
        self.data_groups = None
    #######################################
    def train(self):
        """train the popularity based recommender system model"""
        super().train()

        start_time = default_timer()
        # Get a count of user_ids for each unique item as popularity score
        self.data_groups = self.train_data.groupby(self.user_features + [self.item_id_col])\
                                            .agg({self.user_id_col: 'count'})\
                                            .reset_index()
        self.data_groups.rename(columns={self.user_id_col:'no_of_users',
                                         self.item_id_col:self.item_id_col},
                                inplace=True)
        #print(self.data_groups.head())
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))
        joblib.dump(self.data_groups, self.model_file)
        LOGGER.debug("Saved Model")
    #######################################
    def __get_feature_val(self, identifiers, user_feature):
        """retrieve value for user_feature using identifiers from test_data"""
        for identifier, val in identifiers.items():
            #print(identifier, val, type(val))
            data = self.test_data[self.test_data[identifier] == val]
        return data[user_feature].values[0]

    def __generate_top_recommendations(self, user_id, user_interacted_items):
        """Generate top popularity recommendations"""
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']
        #print(user_id)
        identifiers = {self.user_id_col:user_id}
        feature_vals = dict()
        for user_feature in self.user_features:
            feature_vals[user_feature] = self.__get_feature_val(identifiers, user_feature)
        #pprint(feature_vals)
        for feature, val in feature_vals.items():
            self.data_groups = self.data_groups[self.data_groups[feature] == val]

        #Sort the items based upon popularity score : no_of_users
        data_groups_sort = self.data_groups.sort_values(['no_of_users', self.item_id_col],
                                                        ascending=[0, 1])

        total_no_of_users = len(self.get_all_users(dataset='train'))
        if total_no_of_users == 0:
            total_no_of_users = 1#to avoid division by zero
        data_groups_sort['users_percent'] = data_groups_sort['no_of_users']/total_no_of_users

        data_groups_sort.reset_index(drop=True, inplace=True)
        #print(data_groups_sort.head())
        rank = 1
        for _, reco in data_groups_sort.iterrows():
            item_id = reco[self.item_id_col]
            score = reco['users_percent']
            if item_id in user_interacted_items:#to avoid items which user has already aware
                continue
            if rank > self.no_of_recs:#limit no of recommendations
                break
            item_dict = {
                self.user_id_col : user_id,
                self.item_id_col : item_id,
                'score' : score,
                'rank' : rank
            }
            #print(user_id, item_id, score, rank)
            items_to_recommend.append(item_dict)
            rank += 1
        res_df = pd.DataFrame(items_to_recommend, columns=columns)
        #print(res_df)
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

        if os.path.exists(self.model_file):
            self.data_groups = joblib.load(self.model_file)
            LOGGER.debug("Loaded Trained Model")

            start_time = default_timer()
            assume_interacted_items = self.items_for_evaluation[user_id]['assume_interacted_items']
            user_recommendations = self.__generate_top_recommendations(user_id,
                                                                       assume_interacted_items)
            recommended_items = list(user_recommendations[self.item_id_col].values)
            end_time = default_timer()
            print("{:50}    {}".format("Recommendations generated. ",
                                       utilities.convert_sec(end_time - start_time)))
            return user_recommendations
        else:
            print("Trained Model not found !!!. Failed to generate recommendations")
            return None
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

    def evaluate(self, no_of_recs_to_eval, eval_res_file='evaluation_results.json'):
        """evaluate trained model for different no of ranked recommendations"""
        super().evaluate(no_of_recs_to_eval, eval_res_file)

        if os.path.exists(self.model_file):
            self.data_groups = joblib.load(self.model_file)
            LOGGER.debug("Loaded Trained Model")

            start_time = default_timer()
            #Generate recommendations for the users
            self.items_for_evaluation = self.__recommend_items_to_evaluate()
            self.save_items_for_evaluation()

            precision_recall_intf = PrecisionRecall()
            evaluation_results = precision_recall_intf.compute_precision_recall(
                no_of_recs_to_eval, self.items_for_evaluation, self.items_train)
            end_time = default_timer()
            print("{:50}    {}".format("Evaluation Completed. ",
                                       utilities.convert_sec(end_time - start_time)))

            results_file = os.path.join(self.model_dir, eval_res_file)
            utilities.dump_json_file(evaluation_results, results_file)

            return evaluation_results
        else:
            print("Trained Model not found !!!. Failed to evaluate")
            evaluation_results = {'status' : "Trained Model not found !!!. Failed to evaluate"}

            results_file = os.path.join(self.model_dir, eval_res_file)
            utilities.dump_json_file(evaluation_results, results_file)

            return evaluation_results
    #######################################
