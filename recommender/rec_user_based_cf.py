"""Module for User Based Colloborative Filtering Recommender"""
import os
import sys
import logging
from timeit import default_timer

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.rec_interface import Recommender
from recommender.evaluation import PrecisionRecall

class UserBasedCFRecommender(Recommender):
    """User based colloborative filtering recommender system model"""

    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)
        self.model_file = os.path.join(self.model_dir,
                                       'user_based_cf_model.pkl')

        self.user_similarity_matrix_df = None
        self.similar_users = None
        self.uim_df = None
    #######################################
    def save_uim(self, uim_df):
        """save user item interaction matrix"""
        uim_df_fname = os.path.join(self.model_dir, 'uim.csv')
        uim_df = uim_df.reset_index()#so that user_id col is added as first col
        uim_df.to_csv(uim_df_fname, index=False)#do not write the default index,
                                                #so that on read first col is picked as index col

    def load_uim(self):
        """load user item interaction matrix"""
        uim_df_fname = os.path.join(self.model_dir, 'uim.csv')
        uim_df = pd.read_csv(uim_df_fname, index_col=[self.user_id_col])
        uim_df.index = uim_df.index.map(str)
        return uim_df
    #######################################
    def compute_uim(self):
        """Compute User Item Matrix"""
        start_time = default_timer()
        print()
        print("Combining train_data with test_data_for_evaluation...")
        train_test_data = self.train_data.append(self.test_data_for_evaluation,
                                                 ignore_index=True)
        print("Computing User Item Matrix...")
        uim_df = pd.get_dummies(train_test_data[self.item_id_col])\
                   .groupby(train_test_data[self.user_id_col])\
                   .apply(max)
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))
        uim = uim_df.as_matrix()
        non_zero_count = np.count_nonzero(uim)
        count = uim.size
        density = non_zero_count/count
        print("Density of User Item Matrix : ", density)
        return uim_df

    def compute_user_similarity(self):
        """construct matrix using cooccurence of items"""
        #Compute User Item Matrix
        self.uim_df = self.compute_uim()
        self.save_uim(self.uim_df)
        uim = self.uim_df.as_matrix()
        #         Item1   Item2   Item3   Item4
        # User1       1       1       0       0
        # User2       0       1       1       0
        # User3       0       1       1       1
        # uim = np.array([
        #     [1,1,0,0],
        #     [0,1,1,0],
        #     [0,1,1,1]
        # ])

        #stats
        print()
        items = [str(col) for col in self.uim_df.columns]
        no_of_items = len(items)
        users = [str(idx) for idx in self.uim_df.index]
        no_of_users = len(users)
        print("No of Items : ", no_of_items)
        print("No of Users : ", no_of_users)
        #Compute User-User Similarity Matrix with intersection of items interacted
        print()
        print("Computing User-User Similarity Matrix with intersection of items interacted...")
        start_time = default_timer()
        #intersection is like the `&` operator,
        #i.e., item A has user X and item B has user X -> intersection
        #multiplication of 1s and 0s is equivalent to the `&` operator
        intersection = np.dot(uim, uim.T)   #3*4 x 4*3 --> 3*3 User-User
        intersection_df = pd.DataFrame(intersection,
                                       columns=users,
                                       index=users)
        intersection_df_fname = os.path.join(self.model_dir,
                                             'intersection.csv')
        intersection_df.to_csv(intersection_df_fname, index=False)
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))

        #Compute User-User Similarity Matrix with union of items interacted
        print()
        print("Computing User-User Similarity Matrix with union of items interacted...")
        start_time = default_timer()
        #union is like the `|` operator, i.e., item A has user X or item B has user X -> union
        #`0*0=0`, `0*1=0`, and `1*1=1`, so `*` is equivalent to `|` if we consider `0=T` and `1=F`
        #Hence we obtain flip_uim
        flip_uim = 1-uim    #3*4
        items_left_out_of_union = np.dot(flip_uim, flip_uim.T)  #3*4 x 4*3 --> 3*3 User-User
        union = no_of_items - items_left_out_of_union
        union_df = pd.DataFrame(union,
                                columns=users,
                                index=users)
        union_df_fname = os.path.join(self.model_dir,
                                      'union.csv')
        union_df.to_csv(union_df_fname, index=False)
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))

        #Compute User-User Similarity Matrix with Jaccard Similarity of items
        print()
        print("Computing User-User Similarity Matrix with Jaccard Similarity of items interacted...")
        start_time = default_timer()
        jaccard = intersection/union
        jaccard_df = pd.DataFrame(jaccard,
                                  columns=users,
                                  index=users)
        jaccard_df_fname = os.path.join(self.model_dir,
                                        'jaccard.csv')
        jaccard_df.to_csv(jaccard_df_fname, index=False)
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))
        return jaccard_df

    def train(self):
        """train the user similarity based recommender system model"""
        super().train()

        # Compute user similarity matrix of size, len(users) X len(users)
        print("Compute user similarity matrix...")
        start_time = default_timer()
        self.user_similarity_matrix_df = self.compute_user_similarity()
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))
        #print(self.user_similarity_matrix_df.shape)
        joblib.dump(self.user_similarity_matrix_df, self.model_file)
        LOGGER.debug("Saved Model : " + self.model_file)
    #######################################
    def __get_similar_users(self, user_id):
        """retrieve similar users for a given user_id"""
        #print(user_id)
        similar_users = self.user_similarity_matrix_df[user_id]
        sorted_similar_users = similar_users.sort_values(ascending=False)
        #print(len(sorted_similar_users))
        most_similar_users = (sorted_similar_users.drop(user_id))
        most_similar_users = most_similar_users[most_similar_users > 0]#score > 0
        #print(len(most_similar_users))
        #print(most_similar_users)
        #input()
        return most_similar_users

    def __generate_top_recommendations(self, user_id, user_interacted_items):
        """Use the cooccurence matrix to make top recommendations"""
        # Calculate a weighted average of the scores in cooccurence matrix for
        # all user items.
        #print(user_id)
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']

        similar_users_weights = self.__get_similar_users(user_id)
        similar_user_ids = similar_users_weights.index
        #print(similar_user_ids)
        #top_10_users = list(similar_users_weights.head(10).index)
        #print(top_10_users)
        #input()
        sub_uim_df = self.uim_df.loc[similar_user_ids]
        weighted_sub_uim_df = sub_uim_df.mul(similar_users_weights, axis='index')
        no_of_similar_users = weighted_sub_uim_df.shape[0]
        if no_of_similar_users != 0:
            item_scores = weighted_sub_uim_df.sum(axis=0) / float(no_of_similar_users)
            item_scores.sort_values(inplace=True, ascending=False)
            #print(item_scores)
            #item_scores = item_scores[item_scores > 0]

            rank = 1
            for item_id, score in item_scores.items():
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
        self.uim_df = self.load_uim()

        if os.path.exists(self.model_file):
            self.user_similarity_matrix_df = joblib.load(self.model_file)
            #print(self.user_similarity_matrix_df.shape)
            LOGGER.debug("Loaded Trained Model")
            # Use the cooccurence matrix to make recommendations
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
        self.uim_df = self.load_uim()

        if os.path.exists(self.model_file):
            self.user_similarity_matrix_df = joblib.load(self.model_file)
            #print(self.user_similarity_matrix_df.shape)
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

            results_file = os.path.join(self.model_dir, 'evaluation_results.json')
            utilities.dump_json_file(evaluation_results, results_file)

            return evaluation_results
    #######################################
