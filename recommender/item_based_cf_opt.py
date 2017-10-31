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
from recommender.reco_interface import RecommenderIntf
from recommender.evaluation import PrecisionRecall

class ItemBasedCFRecommender(RecommenderIntf):
    """Item based colloborative filtering recommender system model"""

    def __derive_stats(self):
        """private function, derive stats"""
        LOGGER.debug("Train Data :: Deriving Stats...")
        self.users_train = list(self.train_data[self.user_id_col].unique())
        LOGGER.debug("Train Data :: No. of users : " + str(len(self.users_train)))
        self.items_train = list(self.train_data[self.item_id_col].unique())
        LOGGER.debug("Train Data :: No. of items : " + str(len(self.items_train)))

        users_items_train_dict = {
            'users_train' : self.users_train,
            'items_train' : self.items_train
        }
        #pprint(users_items_train_dict)
        users_items_train_file = os.path.join(self.model_dir, 'users_items_train.json')
        utilities.dump_json_file(users_items_train_dict, users_items_train_file)

        LOGGER.debug("Train Data :: Getting Distinct Users for each Item")
        item_users_train_df = self.train_data.groupby([self.item_id_col])\
                                             .agg({
                                                 self.user_id_col: (lambda x: list(x.unique()))
                                                 })
        item_users_train_df = item_users_train_df.rename(columns={self.user_id_col: 'users'})\
                                                      .reset_index()
        for _, item_users in item_users_train_df.iterrows():
            item = item_users[str(self.item_id_col)]
            users = item_users['users']
            self.item_users_train_dict[item] = users
        item_users_train_file = os.path.join(self.model_dir, 'item_users_train.json')
        utilities.dump_json_file(self.item_users_train_dict, item_users_train_file)

        LOGGER.debug("Train Data :: Getting Distinct Items for each User")
        user_items_train_df = self.train_data.groupby([self.user_id_col])\
                                             .agg({
                                                 self.item_id_col: (lambda x: list(x.unique()))
                                                 })
        user_items_train_df = user_items_train_df.rename(columns={self.item_id_col: 'items'})\
                                                 .reset_index()
        for _, user_items in user_items_train_df.iterrows():
            user = user_items[str(self.user_id_col)]
            items = user_items['items']
            self.user_items_train_dict[user] = items
        user_items_train_file = os.path.join(self.model_dir, 'user_items_train.json')
        utilities.dump_json_file(self.user_items_train_dict, user_items_train_file)
        ########################################################################
        LOGGER.debug("Test Data :: Deriving Stats...")
        self.users_test = list(self.test_data[self.user_id_col].unique())
        LOGGER.debug("Test Data :: No. of users : " + str(len(self.users_test)))
        self.items_test = list(self.test_data[self.item_id_col].unique())
        LOGGER.debug("Test Data :: No. of items : " + str(len(self.items_test)))

        users_items_test_dict = {
            'users_test' : self.users_test,
            'items_test' : self.items_test
        }
        users_items_test_file = os.path.join(self.model_dir, 'users_items_test.json')
        utilities.dump_json_file(users_items_test_dict, users_items_test_file)

        LOGGER.debug("Test Data :: Getting Distinct Users for each Item")
        item_users_test_df = self.test_data.groupby([self.item_id_col])\
                                             .agg({
                                                 self.user_id_col: (lambda x: list(x.unique()))
                                                 })
        item_users_test_df = item_users_test_df.rename(columns={self.user_id_col: 'users'})\
                                                      .reset_index()
        for _, item_users in item_users_test_df.iterrows():
            item = item_users[str(self.item_id_col)]
            users = item_users['users']
            self.item_users_test_dict[item] = users

        item_users_test_file = os.path.join(self.model_dir, 'item_users_test.json')
        utilities.dump_json_file(self.item_users_test_dict, item_users_test_file)

        LOGGER.debug("Test Data :: Getting Distinct Items for each User")
        user_items_test_df = self.test_data.groupby([self.user_id_col])\
                                             .agg({
                                                 self.item_id_col: (lambda x: list(x.unique()))
                                                 })
        user_items_test_df = user_items_test_df.rename(columns={self.item_id_col: 'items'})\
                                                      .reset_index()
        for _, user_items in user_items_test_df.iterrows():
            user = user_items[str(self.user_id_col)]
            items = user_items['items']
            self.user_items_test_dict[user] = items
        user_items_test_file = os.path.join(self.model_dir, 'user_items_test.json')
        utilities.dump_json_file(self.user_items_test_dict, user_items_test_file)

    def __load_stats(self):
        """private function, derive stats"""
        LOGGER.debug("Train Data :: Loading Stats...")
        users_items_train_file = os.path.join(self.model_dir, 'users_items_train.json')
        users_items_train_dict = utilities.load_json_file(users_items_train_file)
        self.users_train = users_items_train_dict['users_train']
        LOGGER.debug("Train Data :: No. of users : " + str(len(self.users_train)))
        self.items_train = users_items_train_dict['items_train']
        LOGGER.debug("Train Data :: No. of items : " + str(len(self.items_train)))

        LOGGER.debug("Train Data :: Loading Distinct Users for each Item")
        item_users_train_file = os.path.join(self.model_dir, 'item_users_train.json')
        self.item_users_train_dict = utilities.load_json_file(item_users_train_file)

        LOGGER.debug("Train Data :: Loading Distinct Items for each User")
        user_items_train_file = os.path.join(self.model_dir, 'user_items_train.json')
        self.user_items_train_dict = utilities.load_json_file(user_items_train_file)
        ############################################################################
        LOGGER.debug("Test Data :: Loading Stats...")
        users_items_test_file = os.path.join(self.model_dir, 'users_items_test.json')
        users_items_test_dict = utilities.load_json_file(users_items_test_file)
        self.users_test = users_items_test_dict['users_test']
        LOGGER.debug("Test Data :: No. of users : " + str(len(self.users_test)))
        self.items_test = users_items_test_dict['items_test']
        LOGGER.debug("Test Data :: No. of items : " + str(len(self.items_test)))

        LOGGER.debug("Test Data :: Loading Distinct Users for each Item")
        item_users_test_file = os.path.join(self.model_dir, 'item_users_test.json')
        self.item_users_test_dict = utilities.load_json_file(item_users_test_file)

        LOGGER.debug("Test Data :: Loading Distinct Items for each User")
        user_items_test_file = os.path.join(self.model_dir, 'user_items_test.json')
        self.user_items_test_dict = utilities.load_json_file(user_items_test_file)

    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, no_of_recs=50):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, no_of_recs)
        self.users_train = None
        self.items_train = None
        self.user_items_train_dict = dict()
        self.item_users_train_dict = dict()

        self.users_test = None
        self.items_test = None
        self.user_items_test_dict = dict()
        self.item_users_test_dict = dict()

        self.cooccurence_matrix_df = None
        self.model_file = os.path.join(self.model_dir, 'item_based_model.pkl')

    def __get_items(self, user_id, dataset='train'):
        """private function, Get unique items for a given user"""
        if dataset == "train":
            # condition = self.user_items_train_df[self.user_id_col] == user_id
            # user_data = self.user_items_train_df[condition]
            user_items = self.user_items_train_dict[user_id]
        else:#test
            # condition = self.user_items_test_df[self.user_id_col] == user_id
            # user_data = self.user_items_test_df[condition]
            user_items = self.user_items_test_dict[user_id]
        #print(user_data)
        #user_items = (user_data['items'].values)[0]
        return user_items

    def __get_users(self, item_id, dataset='train'):
        """private function, Get unique users for a given item"""
        if dataset == "train":
            # condition = self.item_users_train_df[self.item_id_col] == item_id
            # item_data = self.item_users_train_df[condition]
            item_users = self.item_users_train_dict[item_id]
        else:#test
            # condition = self.item_users_test_df[self.item_id_col] == item_id
            # item_data = self.item_users_test_df[condition]
            item_users = self.item_users_test_dict[item_id]
        #item_users = (item_data['users'].values)[0]
        return item_users

    def __get_all_users(self, dataset='train'):
        """private function, Get unique users in the data"""
        if dataset == "train":
            return self.users_train
        else:#test
            return self.users_test

    def __get_all_items(self, dataset='train'):
        """private function, Get unique items in the data"""
        if dataset == "train":
            return self.items_train
        else:#test
            return self.items_test

    def __construct_cooccurence_matrix(self):
        """private function, Construct cooccurence matrix"""
        #Construct User Item Matrix
        uim_df = pd.get_dummies(self.train_data[self.item_id_col])\
                   .groupby(self.train_data[self.user_id_col])\
                   .apply(max)
        uim = uim_df.as_matrix()

        #stats
        items = list(uim_df.columns)
        no_of_items = len(items)
        users = list(uim_df.index)
        no_of_users = len(users)
        print("No of Items : ", no_of_items)
        print("No of Users : ", no_of_users)
        non_zero_count = np.count_nonzero(uim)
        count = uim.size
        density = non_zero_count/count
        #print(non_zero_count, count, density)
        print("Density of User Item Matrix : ", density)

        #Compute Item-Item Matrix with intersection of users
        item_item_intersection = np.dot(uim.T, uim)
        item_item_intersection_df = pd.DataFrame(item_item_intersection,
                                                 columns=uim_df.columns,
                                                 index=uim_df.columns)
        item_item_intersection_df_fname = os.path.join(self.model_dir,
                                                       'item_item_intersection.csv')
        item_item_intersection_df.to_csv(item_item_intersection_df_fname, index=False)

        #Compute Item-Item Matrix with union of users
        flip_uim = 1-uim
        users_left_out_of_union = np.dot(flip_uim.T, flip_uim)
        item_item_union = no_of_users - users_left_out_of_union
        item_item_union_df = pd.DataFrame(item_item_union,
                                          columns=uim_df.columns,
                                          index=uim_df.columns)
        item_item_union_df_fname = os.path.join(self.model_dir,
                                                'item_item_union.csv')
        item_item_union_df.to_csv(item_item_union_df_fname, index=False)

        #Compute Item-Item Matrix with Jaccard Similarity of users
        item_item_jaccard = item_item_intersection/item_item_union
        item_item_jaccard_df = pd.DataFrame(item_item_jaccard,
                                            columns=uim_df.columns,
                                            index=uim_df.columns)
        item_item_jaccard_df_fname = os.path.join(self.model_dir,
                                                  'item_item_jaccard.csv')
        item_item_jaccard_df.to_csv(item_item_jaccard_df_fname, index=False)
        return item_item_jaccard_df

    def train(self):
        """Train the item similarity based recommender system model"""
        self.__derive_stats()
        print("Training...")
        # Construct item cooccurence matrix of size, len(items) X len(items)
        start_time = default_timer()
        self.cooccurence_matrix_df = self.__construct_cooccurence_matrix()
        end_time = default_timer()
        print("{:50}    {}".format("Training Completed in : ",
                                   utilities.convert_sec(end_time - start_time)))
        #print(self.cooccurence_matrix_df.shape)
        joblib.dump(self.cooccurence_matrix_df, self.model_file)
        LOGGER.debug("Saved Model")

    def __generate_top_recommendations(self, user_id, user_items):
        """Use the cooccurence matrix to make top recommendations"""
        # Calculate a weighted average of the scores in cooccurence matrix for
        # all user items.
        items_to_recommend = []
        columns = ['user_id', 'item_id', 'score', 'rank']

        sub_cooccurence_matrix_df = self.cooccurence_matrix_df.loc[user_items]
        no_of_user_items = sub_cooccurence_matrix_df.shape[0]
        if no_of_user_items != 0:
            item_scores = sub_cooccurence_matrix_df.sum(axis=0) / float(no_of_user_items)
            item_scores.sort_values(inplace=True, ascending=False)
            item_scores = item_scores[item_scores > 0]

            rank = 1
            for item_id, score in item_scores.items():
                if item_id in user_items:#to avoid items which user has already aware
                    continue
                if rank > self.no_of_recs:#limit no of recommendations
                    break
                item_dict = {
                    'user_id' : user_id,
                    'item_id' : item_id,
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

    def recommend_items(self, user_id, dataset='train'):
        """Generate item recommendations for given user_id from chosen dataset"""
        self.__load_stats()
        if not os.path.exists(self.model_file):
            print("Trained Model not found !!!. Failed to recommend")
            return None
        self.cooccurence_matrix_df = joblib.load(self.model_file)
        #print(self.cooccurence_matrix_df.shape)
        LOGGER.debug("Loaded Trained Model")
        # Get all unique items for this user
        user_items = self.__get_items(user_id, dataset)
        print("No. of items for the user_id {} : {}".format(user_id,
                                                            len(user_items)))

        # Use the cooccurence matrix to make recommendations
        start_time = default_timer()
        user_recommendations = self.__generate_top_recommendations(user_id,
                                                                   user_items)
        recommended_items = list(user_recommendations['item_id'].values)
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated in : ",
                                   utilities.convert_sec(end_time - start_time)))
        return recommended_items

    def __split_items(self, items_interacted, hold_out_ratio):
        """return assume_interacted_items, hold_out_items"""
        items_interacted_set = set(items_interacted)
        assume_interacted_items = set()
        hold_out_items = set()
        # print("Items Interacted : ")
        # print(items_interacted)
        hold_out_items = set(self.get_random_sample(items_interacted, hold_out_ratio))
        # print("Items Held Out : ")
        # print(hold_out_items)
        # print("No of items to hold out:", len(hold_out_items))
        assume_interacted_items = items_interacted_set - hold_out_items
        # print("Items Assume to be interacted : ")
        # print(assume_interacted_items)
        # print("No of interacted_items assumed:", len(assume_interacted_items))
        # input()
        return list(assume_interacted_items), list(hold_out_items)

    def __get_known_items(self, items_interacted):
        """return filtered items which are present in training set"""
        known_items_interacted = []
        items_training_set = self.__get_all_items(dataset='train')
        for item in items_interacted:
            if item in items_training_set:
                known_items_interacted.append(item)
        return known_items_interacted

    def __get_items_for_eval(self, dataset='train', hold_out_ratio=0.5):
        """Generate recommended and interacted items for users"""
        eval_items = dict()
        users = self.__get_all_users(dataset)
        no_of_users = len(users)
        no_of_users_considered = 0
        for user_id in users:

            # Get all items with which user has interacted
            items_interacted = self.__get_items(user_id, dataset)
            if dataset != 'train':
                items_interacted = self.__get_known_items(items_interacted)
            assume_interacted_items, hold_out_items = self.__split_items(items_interacted,
                                                                         hold_out_ratio)
            if len(assume_interacted_items) == 0 or len(hold_out_items) == 0:
                # print("WARNING !!!. User {} exempted from evaluation".format(user_id))
                # print("Items Interacted Assumed : ")
                # print(assume_interacted_items)
                # print("Hold Out Items")
                # print(hold_out_items)
                # input()
                continue

            eval_items[user_id] = dict()
            eval_items[user_id]['items_recommended'] = []
            eval_items[user_id]['items_interacted'] = []
            no_of_users_considered += 1
            user_recommendations = self.__generate_top_recommendations(user_id,
                                                                       assume_interacted_items)
            recommended_items = list(user_recommendations['item_id'].values)
            eval_items[user_id]['items_recommended'] = recommended_items

            eval_items[user_id]['items_interacted'] = hold_out_items
        print("Evaluation : No of users : ", no_of_users)
        print("Evaluation : No of users considered : ", no_of_users_considered)
        return eval_items

    def evaluate(self, no_of_recs_to_eval, dataset='train', hold_out_ratio=0.5):
        """Evaluate trained model"""
        print("Evaluating...")
        self.__load_stats()
        start_time = default_timer()
        if os.path.exists(self.model_file):
            self.cooccurence_matrix_df = joblib.load(self.model_file)
            #print(self.cooccurence_matrix_df.shape)
            LOGGER.debug("Loaded Trained Model")

            #Generate recommendations for the users
            eval_items = self.__get_items_for_eval(dataset, hold_out_ratio)
            precision_recall_eval_file = os.path.join(self.results_dir, 'eval_items.json')
            utilities.dump_json_file(eval_items, precision_recall_eval_file)
            #pprint(eval_items)
            #input()
            precision_recall_intf = PrecisionRecall()
            results = precision_recall_intf.compute_precision_recall(
                no_of_recs_to_eval, eval_items)
            end_time = default_timer()
            print("{:50}    {}".format("Evaluation Completed in : ",
                                       utilities.convert_sec(end_time - start_time)))
            return results
        else:
            print("Trained Model not found !!!. Failed to evaluate")
            results = {'status' : "Trained Model not found !!!. Failed to evaluate"}
            end_time = default_timer()
            print("{:50}    {}".format("Evaluation Completed in : ", utilities.convert_sec(end_time - start_time)))
            return results

    def get_similar_items(self, item_list, dataset='train'):
        """Get items similar to given items"""
        self.__load_stats()
        if not os.path.exists(self.model_file):
            print("Trained Model not found !!!. Failed to get similar items")
            return None
        self.cooccurence_matrix_df = joblib.load(self.model_file)
        #print(self.cooccurence_matrix_df.shape)
        LOGGER.debug("Loaded Trained Model")
        items_interacted = item_list
        if dataset != 'train':
            items_interacted = self.__get_known_items(items_interacted)
        if len(items_interacted) == 0:
            print("""The following items are not found in training data.
                  Hence no recommendations can be generated""")
            pprint(item_list)
            similar_items = []
        else:
            # Use the cooccurence matrix to make recommendations
            user_id = ""
            user_recommendations = self.__generate_top_recommendations(
                user_id, items_interacted)
            similar_items = list(user_recommendations['item_id'].values)
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

def train(train_data, test_data, user_id_col, item_id_col,
          results_dir, model_dir):
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
             no_of_recs_to_eval, dataset='train', hold_out_ratio=0.5):
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
    results = model.evaluate(no_of_recs_to_eval, dataset, hold_out_ratio)
    pprint(results)
    print('*' * 80)

def recommend(user_id, user_id_col, item_id_col,
              results_dir, model_dir, dataset='train'):
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
    #items = list(test_data[test_data[user_id_col] == user_id][item_id_col].unique())
    print("Items recommended for a user with user_id : {}".format(user_id))
    recommended_items = model.recommend_items(user_id, dataset)
    if recommended_items:
        for item in recommended_items:
            print(item)
    else:
        print("No items to recommend")
    print('*' * 80)

def train_eval_recommend(train_data, test_data,
                         user_id_col, item_id_col,
                         results_dir, model_dir,
                         no_of_recs_to_eval, dataset='train', hold_out_ratio=0.5):
    """Train Evaluate and Recommend for Item Based Recommender"""

    print("Training Recommender...")
    model = ItemBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col,
                                   item_id_col)
    model.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    results = model.evaluate(no_of_recs_to_eval, dataset, hold_out_ratio)
    pprint(results)
    print('*' * 80)

    print("Testing Recommendation for an User")
    users = test_data[user_id_col].unique()
    user_id = users[0]
    items = list(test_data[test_data[user_id_col] == user_id][item_id_col].unique())
    print("Items recommended for a user with user_id : {}".format(user_id))
    recommended_items = model.get_similar_items(items, dataset)
    for item in recommended_items:
        print(item)
    print('*' * 80)
