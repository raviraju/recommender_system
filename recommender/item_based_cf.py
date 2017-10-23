"""Module for Item Based Colloborative Filtering Recommender"""
import os
import sys
import logging
from timeit import default_timer
from pprint import pprint
import joblib

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.reco_interface import RecommenderIntf
from recommender.evaluation import PrecisionRecall


class ItemBasedCFRecommender(RecommenderIntf):
    """Item based colloborative filtering recommender system model"""
    def __derive_stats(self):
        """private function, derive stats"""
        LOGGER.debug("Getting All Users and Items")
        self.all_users = list(self.train_data[self.user_id_col].unique())
        LOGGER.debug("No. of users in the training set: " + str(len(self.all_users)))
        self.all_items = list(self.train_data[self.item_id_col].unique())
        LOGGER.debug("No. of items in the training set: " + str(len(self.all_items)))

        LOGGER.debug("Getting Distinct Users for each Item")
        item_users_df = self.train_data.groupby([self.item_id_col]).agg(
            {self.user_id_col: (lambda x: list(x.unique()))})
        self.item_users_df = item_users_df.rename(
            columns={self.user_id_col: 'users'}).reset_index()
        #print(self.item_users_df.head())
        item_users_file = os.path.join(self.model_dir, 'item_users.csv')
        item_users_df.to_csv(item_users_file)

        LOGGER.debug("Getting Distinct Items for each User")
        user_items_df = self.train_data.groupby([self.user_id_col]).agg(
            {self.item_id_col: (lambda x: list(x.unique()))})
        self.user_items_df = user_items_df.rename(
            columns={self.item_id_col: 'items'}).reset_index()
        #print(self.user_items_df.head())
        user_items_file = os.path.join(self.model_dir, 'user_items.csv')
        user_items_df.to_csv(user_items_file)        

    def __init__(self, results_dir, model_dir, train_data, test_data, user_id_col, item_id_col, no_of_recs=10):
        """constructor"""
        super().__init__(results_dir, model_dir, train_data, test_data, user_id_col, item_id_col, no_of_recs)

        self.all_users = None
        self.all_items = None
        self.user_items_df = None
        self.item_users_df = None
        self.__derive_stats()

        self.cooccurence_matrix = None
        self.model_file = os.path.join(self.model_dir, 'item_based_model.pkl')

    def __get_items(self, user_id):
        """private function, Get unique items for a given user"""
        user_data = self.user_items_df[
            self.user_items_df[self.user_id_col] == user_id]
        #print(user_data)
        user_items = (user_data['items'].values)[0]
        return user_items

    def __get_users(self, item_id):
        """private function, Get unique users for a given item"""
        item_data = self.item_users_df[
            self.item_users_df[self.item_id_col] == item_id]
        item_users = (item_data['users'].values)[0]
        return item_users

    def __get_all_users(self):
        """private function, Get unique users in the training data"""
        return self.all_users

    def __get_all_items(self):
        """private function, Get unique items in the training data"""
        return self.all_items

    def __construct_cooccurence_matrix(self, items):
        """private function, Construct cooccurence matrix"""
        # Initialize the item cooccurence matrix
        len_items = len(items)
        cooccurence_matrix = np.matrix(
            np.zeros(shape=(len_items, len_items)), float)

        # Calculate similarity between item pairs for upper triangular elements
        for i, item_i in enumerate(items):
            # Get unique users of item_i
            users_i = set(self.__get_users(item_i))

            for j, item_j in enumerate(items):
                if i == j:
                    cooccurence_matrix[i, j] = 1.0
                    continue
                if i > j:
                    continue #same result as corresponding j, i

                # Get unique users of item_j
                users_j = set(self.__get_users(item_j))

                # Calculate intersection of users of items i and j
                users_intersection = users_i.intersection(users_j)
                no_of_common_users = len(users_intersection)
                # Calculate cooccurence_matrix[i,j] as Jaccard Index
                if no_of_common_users != 0:
                    # Calculate union of users of items i and j
                    users_union = users_i.union(users_j)
                    no_of_all_users = len(users_union)
                    if no_of_all_users != 0:
                        cooccurence_matrix[i, j] = float(
                            no_of_common_users) / float(no_of_all_users)
                        cooccurence_matrix[j, i] = cooccurence_matrix[i, j]

        non_zeros = np.count_nonzero(cooccurence_matrix)
        print("Non zero values in Co-Occurence_matrix : {}".format(non_zeros))
        density = float(non_zeros / cooccurence_matrix.size)
        print("Density : {}".format(density))
        return cooccurence_matrix

    def train(self):
        """Train the item similarity based recommender system model"""
        # Construct item cooccurence matrix of size, len(items) X len(items)
        start_time = default_timer()
        self.cooccurence_matrix = self.__construct_cooccurence_matrix(
            self.all_items)
        end_time = default_timer()
        print("{:50}    {}".format("Training Completed in : ", utilities.convert_sec(end_time - start_time)))
        #print(self.cooccurence_matrix.shape)
        joblib.dump(self.cooccurence_matrix, self.model_file)
        LOGGER.debug("Saved Model")

    def __generate_top_recommendations(self, user, all_items, user_items):
        """Use the cooccurence matrix to make top recommendations"""
        # Calculate a weighted average of the scores in cooccurence matrix for
        # all user items.
        user_sim_scores = self.cooccurence_matrix.sum(
            axis=0) / float(self.cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        # Sort the indices of user_sim_scores based upon their value
        # Also maintain the corresponding score
        sort_index = sorted(((e, i) for i, e in enumerate(
            list(user_sim_scores))), reverse=True)

        # Create a dataframe from the following
        columns = ['user_id', 'item_id', 'score', 'rank']
        # index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)

        # Fill the dataframe with top 10 item based recommendations
        rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_items[sort_index[i][1]] not in user_items and rank <= self.no_of_recs:
                df.loc[len(df)] = [user, all_items[
                    sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1

        # Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("""The current user has no items for training the item similarity 
                     based recommendation model.""")
            return -1
        else:
            return df

    def recommend(self, user_id):
        """Generate item recommendations for given user_id"""
        if not os.path.exists(self.model_file):
            print("Trained Model not found !!!. Failed to recommend")
            return None

        self.cooccurence_matrix = joblib.load(self.model_file)
        #print(self.cooccurence_matrix.shape)
        LOGGER.debug("Loaded Trained Model")
        all_items = self.__get_all_items()
        # Get all unique items for this user
        user_items = self.__get_items(user_id)
        print("No. of items for the user_id {} : {}".format(
            user_id, len(user_items)))

        # Use the cooccurence matrix to make recommendations
        start_time = default_timer()
        user_recommendations = self.__generate_top_recommendations(
            user_id, all_items, user_items)
        recommended_items = user_recommendations['item_id']
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated in : ",
                                   utilities.convert_sec(end_time - start_time)))       
        return recommended_items

    def __get_items_for_eval(self, users_test_sample):
        """Generate recommended and interacted items for users in the user test sample"""
        eval_items = dict()

        all_items = self.__get_all_items()
        for user_id in users_test_sample:
            eval_items[user_id] = dict()
            eval_items[user_id]['items_recommended'] = dict()
            eval_items[user_id]['items_interacted'] = dict()
            # Get all unique items for this user
            user_items = self.__get_items(user_id)

            user_recommendations = self.__generate_top_recommendations(
                user_id, all_items, user_items)
            recommended_items = user_recommendations['item_id']
            eval_items[user_id]['items_recommended'] = recommended_items

            #Get items for user_id from test_data
            test_data_user = self.test_data[self.test_data[self.user_id_col] == user_id]
            eval_items[user_id]['items_interacted'] = test_data_user[self.item_id_col].unique()
        return eval_items

    def eval(self, sample_test_users_percentage, no_of_recs_to_eval):
        """Evaluate trained model"""
        start_time = default_timer()        
        if os.path.exists(self.model_file):
            self.cooccurence_matrix = joblib.load(self.model_file)
            #print(self.cooccurence_matrix.shape)
            LOGGER.debug("Loaded Trained Model")

            #Get a sample of common users from test and training set
            users_test_sample = self.fetch_sample_test_users(sample_test_users_percentage)
            if len(users_test_sample) == 0:
                print("""None of users are common in training and test data.
                         Hence cannot evaluate model""")
                return {'status' : "Common Users not found, Failed to Evaluate"}

            #Generate recommendations for the test sample users
            eval_items = self.__get_items_for_eval(users_test_sample)

            precision_recall_intf = PrecisionRecall()
            results = precision_recall_intf.compute_precision_recall(
                no_of_recs_to_eval, eval_items)
            end_time = default_timer()
            print("{:50}    {}".format("Evaluation Completed in : ", utilities.convert_sec(end_time - start_time)))                
            return results
        else:
            print("Trained Model not found !!!. Failed to evaluate")
            results = {'status' : "Trained Model not found !!!. Failed to evaluate"}
            end_time = default_timer()
            print("{:50}    {}".format("Evaluation Completed in : ", utilities.convert_sec(end_time - start_time)))
            return results

    def get_similar_items(self, item_list):
        """Get items similar to given items"""
        if not os.path.exists(self.model_file):
            print("Trained Model not found !!!. Failed to get similar items")
            return None
        self.cooccurence_matrix = joblib.load(self.model_file)
        #print(self.cooccurence_matrix.shape)
        LOGGER.debug("Loaded Trained Model")
        user_items = item_list
        all_items = self.__get_all_items()

        # Use the cooccurence matrix to make recommendations
        user_id = ""
        recommendations = self.__generate_top_recommendations(
            user_id, all_items, user_items)
        similar_items = recommendations['item_id']
        return similar_items

def load_train_test(model_dir):
    """Load Train and Test Data"""
    train_file = os.path.join(model_dir, 'train_data.csv')
    train_data = pd.read_csv(train_file)
    test_file = os.path.join(model_dir, 'test_data.csv')
    test_data = pd.read_csv(test_file)
    print("{:30} : {}".format("No of records in train_data", len(train_data)))
    print("{:30} : {}".format("No of records in test_data", len(test_data)))
    return train_data, test_data

def train(train_data, test_data, user_id_col, item_id_col, results_dir, model_dir):
    """train recommender"""
    print("Training Recommender...")
    model = ItemBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col,
                                   item_id_col)
    model.train()
    print('*' * 80)

def evaluate(user_id_col, item_id_col,
             results_dir, model_dir,
             no_of_recs_to_eval, sample_test_users_percentage):
    """evaluate recommender"""
    print("Loading Training and Test Data")
    train_data, test_data = load_train_test(model_dir)
    # print(train_data.head(5))
    # print(test_data.head(5))
    print('*' * 80)

    print("Evaluating Recommender System")
    model = ItemBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col,
                                   item_id_col)
    results = model.eval(sample_test_users_percentage, no_of_recs_to_eval)
    pprint(results)
    print('*' * 80)

def recommend(user_id, user_id_col, item_id_col, results_dir, model_dir):
    """recommend items for user"""
    print("Loading Training and Test Data")
    train_data, test_data = load_train_test(model_dir)
    # print(train_data.head(5))
    # print(test_data.head(5))
    print('*' * 80)

    model = ItemBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col,
                                   item_id_col)
    items = list(test_data[test_data[user_id_col] == user_id][item_id_col].unique())
    print("Items recommended for a user with user_id : {}".format(user_id))
    recommended_items = model.get_similar_items(items)
    for item in recommended_items:
        print(item)
    print('*' * 80)

def train_eval_recommend(train_data, test_data,
                         user_id_col, item_id_col,
                         results_dir, model_dir,
                         no_of_recs_to_eval,
                         sample_test_users_percentage):
    """Train Evaluate and Recommend for Item Based Recommender"""

    print("Training Recommender...")
    model = ItemBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col,
                                   item_id_col)
    model.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    results = model.eval(sample_test_users_percentage, no_of_recs_to_eval)
    pprint(results)
    print('*' * 80)

    print("Testing Recommendation for an User")
    users = test_data[user_id_col].unique()
    user_id = users[0]
    items = list(test_data[test_data[user_id_col] == user_id][item_id_col].unique())
    print("Items recommended for a user with user_id : {}".format(user_id))
    recommended_items = model.get_similar_items(items)
    for item in recommended_items:
        print(item)
    print('*' * 80)
