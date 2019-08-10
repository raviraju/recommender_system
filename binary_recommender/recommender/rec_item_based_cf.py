"""Module for Item Based Colloborative Filtering Recommender"""
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

class ItemBasedCFRecommender(Recommender):
    """Item based colloborative filtering recommender system model"""

    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)
        self.model_file = os.path.join(self.model_dir,
                                       'item_based_cf_model.pkl')
        
        self.debug = False
        if 'debug' in kwargs:
            self.debug = kwargs['debug']        
        
        self.item_similarity_matrix_df = None
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
        print("\tComputing User Item Matrix of Users and Items in Train & Known Test Data...")        
        uim_df = pd.get_dummies(self.train_data[self.item_id_col])\
                   .groupby(self.train_data[self.user_id_col])\
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

    def __compute_item_similarity(self):
        """Compute matrix using cooccurence of users"""
        #Compute User Item Matrix
        # self.uim_df = self.__compute_uim()
        uim_df = self.__compute_uim()
        items_sorted = sorted(uim_df.columns)        #Sort Items
        self.uim_df = uim_df[items_sorted]
        self.uim_df.sort_index(axis=0, inplace=True) #Sort Users
        
        if self.debug:
            self.save_uim(self.uim_df)
        uim = self.uim_df.values

        #stats
        users = [str(idx) for idx in self.uim_df.index]
        no_of_users = len(users)
        items = [str(col) for col in self.uim_df.columns]
        # no_of_items = len(items)
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

        #No of Users who interact with both items(i and j)
        #Compute Item Similarity Matrix with intersection of users interacted
        print()
        print("\tFinding No of Users who interact with both   items (i and j)")
        # print("\tComputing Item Similarity Matrix with intersection of users interacted...")
        start_time = default_timer()        
        #intersection is like the `&` operator,
        #i.e., item A has user X and item B has user X -> intersection
        #multiplication of 1s and 0s is equivalent to the `&` operator
        intersection = np.dot(uim.T, uim)   #4*3 x 3*4 --> 4*4 Item-Item
        intersection_df = pd.DataFrame(intersection,
                                       columns=items,
                                       index=items)

        if self.debug:
            intersection_df_fname = os.path.join(self.model_dir, 'intersection.csv')
            intersection_df.to_csv(intersection_df_fname, index=False)
        end_time = default_timer()
        print("{:50}    {}".format("\tCompleted. ",
                                   utilities.convert_sec(end_time - start_time)))

        #No of Users who interact with either items(i or j)
        #Compute Item Similarity Matrix with union of users interacted
        print()
        print("\tFinding No of Users who interact with either items (i or j)")
        # print("\tComputing Item Similarity Matrix with union of users interacted...")
        start_time = default_timer()
        #union is like the `|` operator, i.e., item A has user X or item B has user X -> union
        #`0*0=0`, `0*1=0`, and `1*1=1`, so `*` is equivalent to `|` if we consider `0=T` and `1=F`
        #Hence we obtain flip_uim
        flip_uim = 1-uim    #3*4
        users_left_out_of_union = np.dot(flip_uim.T, flip_uim)   #4*3 x 3*4 --> 4*4 Item-Item
        union = no_of_users - users_left_out_of_union
        union_df = pd.DataFrame(union,
                                columns=items,
                                index=items)        
        if self.debug:
            union_df_fname = os.path.join(self.model_dir, 'union.csv')
            union_df.to_csv(union_df_fname, index=False)
        end_time = default_timer()
        print("{:50}    {}".format("\tCompleted. ",
                                   utilities.convert_sec(end_time - start_time)))

        #Compute Item Similarity Matrix with Jaccard Similarity of users
        print()
        print("\tComputing Item Similarity Matrix using Jaccard Similarity of User Interactions...")
        start_time = default_timer()
        jaccard_df = intersection_df.div(union_df)
        np.fill_diagonal(jaccard_df.values, 0.0) #set diagonal elements to 0
        jaccard_df.fillna(value=0, inplace=True) #handle Nan values filled due to 0/0
        if self.debug:
            jaccard_df_fname = os.path.join(self.model_dir, 'jaccard.csv')
            jaccard_df.to_csv(jaccard_df_fname, index=False)
        end_time = default_timer()
        print("\t\tItem Similarity Matrix Shape :", jaccard_df.shape)
        print("{:50}    {}".format("\tCompleted. ",
                                   utilities.convert_sec(end_time - start_time)))
        return jaccard_df

    def train(self):
        """Train the item similarity based recommender system model"""
        super().train()

        print()
        print("*"*80)
        print("\tItem Based CF : Customers who liked this item also liked ...")
        print("*"*80)
        # Compute item similarity matrix of size, len(items) X len(items)
        print("Compute Item-Item Similarity Matrix using co-ocuurence of users...")
        start_time = default_timer()
        self.item_similarity_matrix_df = self.__compute_item_similarity()
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))
        #print(self.item_similarity_matrix_df.shape)
        joblib.dump(self.item_similarity_matrix_df, self.model_file)
        LOGGER.debug("Saved Model : " + self.model_file)
    #######################################
    def __generate_top_recommendations(self, user_id, known_interacted_items):
        """Items which are similar to items accessed by a user are filtered and recommended in decreasing order of averaged similarity scores."""
        # Calculate a weighted average of the scores in cooccurence matrix for
        # all user items.
        items_to_recommend = []


        interacted_items_similarity_matrix_df = self.item_similarity_matrix_df.loc[known_interacted_items]
        no_of_known_interacted_items = interacted_items_similarity_matrix_df.shape[0]
        if no_of_known_interacted_items != 0:
            item_scores = interacted_items_similarity_matrix_df.sum(axis=0) / float(no_of_known_interacted_items)
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
        
        if os.path.exists(self.model_file):
            self.item_similarity_matrix_df = joblib.load(self.model_file)
            LOGGER.debug("Loaded Trained Model")
            start_time = default_timer()            
            known_interacted_items = self.items_for_evaluation[user_id]['known_interacted_items']
            items_to_recommend_df = self.__generate_top_recommendations(user_id, known_interacted_items)
            end_time = default_timer()
            print("{:50}    {}".format("Recommendations generated. ",
                                       utilities.convert_sec(end_time - start_time)))
            return items_to_recommend_df
        else:
            print("Trained Model not found !!!. Failed to generate recommendations")
            return None
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
        
        if os.path.exists(self.model_file):
            self.item_similarity_matrix_df = joblib.load(self.model_file)
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
