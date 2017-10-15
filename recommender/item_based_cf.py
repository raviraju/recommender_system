"""Module for Item Based Colloborative Filtering Recommender"""
import os
import sys
from timeit import default_timer

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.reco_interface import RecommenderIntf

class ItemBasedCFRecommender(RecommenderIntf):
    """Item based colloborative filtering recommender system model"""
    def __init__(self, results_dir):
        """constructor"""
        super().__init__(results_dir)
        self.cooccurence_matrix = None
        self.items_dict = None
        self.rev_items_dict = None
        self.recommendations = None

    def get_user_items(self, user_id):
        """Get unique items for a given user"""
        user_data = self.train_data[self.train_data[self.user_id] == user_id]
        user_items = list(user_data[self.item_id].unique())
        return user_items

    def get_item_users(self, item):
        """Get unique users for a given item"""
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = list(item_data[self.user_id].unique())
        return item_users

    def get_all_users_train_data(self):
        """Get unique users in the training data"""
        all_users = list(self.train_data[self.user_id].unique())
        return all_users

    def get_all_items_train_data(self):
        """Get unique items in the training data"""
        all_items = list(self.train_data[self.item_id].unique())
        return all_items

    def construct_cooccurence_matrix(self, user_items, all_items):
        """Construct cooccurence matrix"""
        ####################################
        # Get users for all items in user_items.
        ####################################
        user_items_users = []
        for i in range(0, len(user_items)):
            user_items_users.append(self.get_item_users(user_items[i]))

        ###############################################
        # Initialize the item cooccurence matrix of size
        # len(user_items) X len(items)
        ###############################################
        cooccurence_matrix = np.matrix(
            np.zeros(shape=(len(user_items), len(all_items))), float)

        #############################################################
        # Calculate similarity between user items and all unique items
        # in the training data
        #############################################################
        for i in range(0, len(all_items)):
            # Calculate unique users of item i of all_items
            items_i_data = self.train_data[self.train_data[self.item_id] == all_items[i]]
            users_i = set(items_i_data[self.user_id].unique())

            for j in range(0, len(user_items)):
                # Get unique users of item j of user_items
                users_j = user_items_users[j]

                # Calculate intersection of users of items i and j
                users_intersection = users_i.intersection(users_j)

                # Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    # Calculate union of users of items i and j
                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0

        non_zeros = np.count_nonzero(cooccurence_matrix)
        print("Non zero values in Co-Occurence_matrix : {}".format(non_zeros))
        density = float(non_zeros / cooccurence_matrix.size)
        print("Density : {}".format(density))
        return cooccurence_matrix

    def generate_top_recommendations(self, user, cooccurence_matrix, all_items, user_items):
        """Use the cooccurence matrix to make top recommendations"""
        # Calculate a weighted average of the scores in cooccurence matrix for
        # all user items.
        user_sim_scores = cooccurence_matrix.sum(axis=0) / float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        # Sort the indices of user_sim_scores based upon their value
        # Also maintain the corresponding score
        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)

        # Create a dataframe from the following
        columns = ['user_id', 'item', 'score', 'rank']
        # index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)

        # Fill the dataframe with top 10 item based recommendations
        rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_items[sort_index[i][1]] not in user_items and rank <= 10:
                df.loc[len(df)] = [user, all_items[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1

        # Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no items for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    def train(self, train_data, user_id, item_id):
        """Train the item similarity based recommender system model"""
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    def recommend(self, user_id):
        """Use the item similarity based recommender system model to make recommendations"""
        all_users = self.get_all_users_train_data()
        print("No. of users in the training set: {}".format(len(all_users)))

        ######################################################
        # A. Get all unique items in the training data
        ######################################################
        all_items = self.get_all_items_train_data()
        print("No. of items in the training set: {}".format(len(all_items)))

        ########################################
        # B. Get all unique items for this user
        ########################################
        user_items = self.get_user_items(user_id)
        print("No. of items for the user_id {} : {}".format(user_id, len(user_items)))

        ###############################################
        # C. Construct item cooccurence matrix of size
        # len(user_items) X len(items)
        ###############################################
        start_time = default_timer()
        cooccurence_matrix = self.construct_cooccurence_matrix(user_items, all_items)
        end_time = default_timer()
        print("{:50}    {}".format("Computing CoOccurence Matrix Completed in : ", utilities.convert_sec(end_time - start_time)))

        #######################################################
        # D. Use the cooccurence matrix to make recommendations
        #######################################################
        start_time = default_timer()
        user_recommendations = self.generate_top_recommendations(user_id, cooccurence_matrix, all_items, user_items)
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated in : ", utilities.convert_sec(end_time - start_time)))

        recommendations_file = os.path.join(self.results_dir, 'item_based_cf_recommendation.csv')
        user_recommendations.to_csv(recommendations_file)

        return user_recommendations

    def get_similar_items(self, item_list):
        """Get items similar to given items"""
        user_items = item_list
        ######################################################
        # B. Get all unique items (items) in the training data
        ######################################################
        all_items = self.get_all_items_train_data()
        print("No. of unique items in the training set: {}".format(all_items))

        ###############################################
        # C. Construct item cooccurence matrix of size
        # len(user_items) X len(items)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_items, all_items)

        #######################################################
        # D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        user_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_items, user_items)

        return user_recommendations
