"""Module for User Based Colloborative Filtering Recommender"""
import os
import sys
import logging
from timeit import default_timer
from pprint import pprint

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

        self.debug = False
        if 'debug' in kwargs:
            self.debug = kwargs['debug']

        self.user_similarity_matrix_df = None
        self.similar_users = None
        self.uim_df = None
    #######################################
    def save_uim(self, uim_df):
        """save user item interaction matrix"""        
        uim_df = uim_df.reset_index()#so that user_id col is added as first col
        uim_df_fname = os.path.join(self.model_dir, 'uim.csv')
        uim_df.to_csv(uim_df_fname, index=False)#do not write the default index,
                                                #so that on read first col is picked as index col

    def load_uim(self):
        """load user item interaction matrix"""
        uim_df_fname = os.path.join(self.model_dir, 'uim.csv')
        uim_df = pd.read_csv(uim_df_fname, index_col=[self.user_id_col])
        uim_df.index = uim_df.index.map(str)
        return uim_df
    #######################################
    def __compute_uim(self):
        """Compute User Item Matrix"""
        start_time = default_timer()
        print("\t\tNo of Users : ", len(self.train_data[self.user_id_col].unique()))
        print("\t\tNo of Items : ", len(self.train_data[self.item_id_col].unique()))
        print("\tCombining train_data with interactions known from test...")
        train_with_known_test_data = self.train_data.append(self.known_interactions_from_test_df,
                                                            ignore_index=True)
        print("\t\tNo of Users : ", len(train_with_known_test_data[self.user_id_col].unique()))
        print("\t\tNo of Items : ", len(train_with_known_test_data[self.item_id_col].unique()))
        print("\tComputing User Item Matrix of Users and Items in Train & Known Test Data...")
        uim_df = pd.get_dummies(train_with_known_test_data[self.item_id_col])\
                   .groupby(train_with_known_test_data[self.user_id_col])\
                   .apply(max)
        print("\t\tUser Item Matrix Shape :", uim_df.shape)
        end_time = default_timer()   

        uim = uim_df.values
        non_zero_count = np.count_nonzero(uim)
        count = uim.size
        density = non_zero_count/count
        print("\t\tDensity of User Item Matrix : {:0.4f} %".format(density*100))
        print("{:50}    {}".format("\tCompleted. ",
                                   utilities.convert_sec(end_time - start_time)))
        return uim_df

    def __compute_user_similarity(self):
        """Compute matrix using cooccurence of items"""
        #Compute User Item Matrix
        # self.uim_df = self.__compute_uim()
        uim_df = self.__compute_uim()
        items_sorted = sorted(uim_df.columns)        #Sort Items
        self.uim_df = uim_df[items_sorted]
        self.uim_df.sort_index(axis=0, inplace=True) #Sort Users
        self.save_uim(self.uim_df)
        uim = self.uim_df.values

        #stats
        users = [str(idx) for idx in self.uim_df.index]
        # no_of_users = len(users)
        items = [str(col) for col in self.uim_df.columns]
        no_of_items = len(items)
        # print("\tNo of Users : ", no_of_users)
        # print("\tNo of Items : ", no_of_items)
       
        #for ex
        #         Item1   Item2   Item3   Item4
        # User1       1       1       0       0
        # User2       0       1       1       0
        # User3       0       1       1       1
        # uim = np.array([
        #     [1,1,0,0],
        #     [0,1,1,0],
        #     [0,1,1,1]
        # ])

        #No of Items which are interacted by both users(u and v)
        #Compute User Similarity Matrix with intersection of items interacted
        print()
        print("\tFinding No of Items which are interacted by both   users (u and v)")
        # print("\tComputing User Similarity Matrix with intersection of items interacted...")
        start_time = default_timer()
        #intersection is like the `&` operator,
        #i.e., user A has item X and user B has item X -> intersection
        #multiplication of 1s and 0s is equivalent to the `&` operator
        intersection = np.dot(uim, uim.T)   #3*4 x 4*3 --> 3*3 User-User
        intersection_df = pd.DataFrame(intersection,
                                       columns=users,
                                       index=users)

        if self.debug:
            intersection_df_fname = os.path.join(self.model_dir, 'intersection.csv')
            intersection_df.to_csv(intersection_df_fname, index=False)
        end_time = default_timer()
        print("{:50}    {}".format("\tCompleted. ",
                                   utilities.convert_sec(end_time - start_time)))

        #No of Items which are interacted by either users(u or v)
        #Compute User Similarity Matrix with union of items interacted
        print()
        print("\tFinding No of Items which are interacted by either users (u or v)")
        # print("\tComputing User Similarity Matrix with union of items interacted...")
        start_time = default_timer()
        #union is like the `|` operator, i.e., item A has user X or item B has user X -> union
        #`0*0=0`, `0*1=0`, and `1*1=1`, so `*` is equivalent to `|` if we consider `0=T` and `1=F`
        #Hence we obtain flip_uim
        flip_uim = 1-uim    #3*4
        items_left_out_of_union = np.dot(flip_uim, flip_uim.T)  #3*4 x 4*3 --> 3*3 User
        union = no_of_items - items_left_out_of_union
        union_df = pd.DataFrame(union,
                                columns=users,
                                index=users)
        if self.debug:
            union_df_fname = os.path.join(self.model_dir, 'union.csv')
            union_df.to_csv(union_df_fname, index=False)
        end_time = default_timer()
        print("{:50}    {}".format("\tCompleted. ",
                                   utilities.convert_sec(end_time - start_time)))

        #Compute User Similarity Matrix with Jaccard Similarity of items
        print()
        print("\tComputing User Similarity Matrix using Jaccard Similarity of Items Interacted...")
        start_time = default_timer()
        jaccard_df = intersection_df.div(union_df)
        np.fill_diagonal(jaccard_df.values, 0.0) #set diagonal elements to 0
        jaccard_df.fillna(value=0, inplace=True) #handle Nan values filled due to 0/0
        if self.debug:
            jaccard_df_fname = os.path.join(self.model_dir, 'jaccard.csv')
            jaccard_df.to_csv(jaccard_df_fname, index=False)
        end_time = default_timer()
        print("\t\tUser Similarity Matrix Shape :", jaccard_df.shape)
        print("{:50}    {}".format("\tCompleted. ",
                                   utilities.convert_sec(end_time - start_time)))
        return jaccard_df

    def train(self):
        """Train the user similarity based recommender system model"""
        super().train()

        print()
        print("*"*80)
        print("\tUser Based CF : Customers who are similar to you also liked ...")
        print("*"*80)
        # Compute user similarity matrix of size, len(users) X len(users)
        print("Compute User-User Similarity Matrix using co-occurence of items...")
        start_time = default_timer()
        self.user_similarity_matrix_df = self.__compute_user_similarity()
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

    def __generate_top_recommendations(self, user_id, known_interacted_items):
        """Most similar users for a given user are filtered. 
           Items accessed by most similar users are recommended in decreasing order of averaged similarity scores weighted by user similarity.
        """
        # Calculate a weighted average of the scores in cooccurence matrix for
        # all user items.
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']

        similar_users_weights = self.__get_similar_users(user_id)
        similar_user_ids = similar_users_weights.index
        #print(similar_user_ids)
        #top_10_users = list(similar_users_weights.head(10).index)
        #print(top_10_users)
        #input()
        similar_users_uim_df = self.uim_df.loc[similar_user_ids]
        weighted_similar_users_uim_df = similar_users_uim_df.mul(similar_users_weights, axis='index')
        no_of_similar_users = weighted_similar_users_uim_df.shape[0]
        if no_of_similar_users != 0:
            item_scores = weighted_similar_users_uim_df.sum(axis=0) / float(no_of_similar_users)
            item_scores.sort_values(inplace=True, ascending=False)
            #print(item_scores)
            #item_scores = item_scores[item_scores > 0]

            rank = 1
            for item_id, score in item_scores.items():
                if not self.allow_recommending_known_items and item_id in known_interacted_items:#to avoid items which user has already aware
                    continue
                if rank > self.no_of_recs:#limit no of recommendations
                    break
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
        self.uim_df = self.load_uim()

        if os.path.exists(self.model_file):
            self.user_similarity_matrix_df = joblib.load(self.model_file)
            #print(self.user_similarity_matrix_df.shape)
            LOGGER.debug("Loaded Trained Model")
            # Use the cooccurence matrix to make recommendations
            start_time = default_timer()            
            known_interacted_items = self.items_for_evaluation[user_id]['known_interacted_items']            
            if len(known_interacted_items) == 0:
                print("User {} has not interacted with any known items, Unable to generate any recommendations...".format(user_id))
                user_recommendations = None
            else:
                user_recommendations = self.__generate_top_recommendations(user_id, known_interacted_items)
            # recommended_items = list(user_recommendations[self.item_id_col].values)
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
            known_interacted_items = self.items_for_evaluation[user_id]['known_interacted_items']            
            if len(known_interacted_items) == 0:
                #print("User {} has not interacted with any items in past, Unable to generate any recommendations...".format(user_id))
                recommended_items = []
            else:
                user_recommendations = self.__generate_top_recommendations(user_id, known_interacted_items)
                recommended_items = list(user_recommendations[self.item_id_col].values)
            self.items_for_evaluation[user_id]['items_recommended'] = recommended_items

            recommended_items_dict = dict()
            for _, recs in user_recommendations.iterrows():
                item_id = recs[self.item_id_col]
                score = round(recs['score'], 3)
                rank = recs['rank']
                recommended_items_dict[item_id] = {'score' : score, 'rank' : rank}
            self.items_for_evaluation[user_id]['items_recommended_score'] = recommended_items_dict

            items_to_be_interacted_set = set(self.items_for_evaluation[user_id]['items_to_be_interacted'])
            items_recommended_set = set(recommended_items)
            correct_recommendations = items_to_be_interacted_set & items_recommended_set
            no_of_correct_recommendations = len(correct_recommendations)
            self.items_for_evaluation[user_id]['no_of_correct_recommendations'] = no_of_correct_recommendations
            self.items_for_evaluation[user_id]['correct_recommendations'] = list(correct_recommendations)
        return self.items_for_evaluation

    def evaluate(self, no_of_recs_to_eval, eval_res_file='evaluation_results.json'):
        """Evaluate trained model for different no of ranked recommendations"""
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
                no_of_recs_to_eval, self.items_for_evaluation, self.items_all)
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
