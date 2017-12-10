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
from recommender.reco_interface import RecommenderIntf
from recommender.reco_interface import load_train_test
from recommender.evaluation import PrecisionRecall

class UserBasedCFRecommender(RecommenderIntf):
    """User based colloborative filtering recommender system model"""

    def __derive_stats(self):
        """private function, derive stats"""
        LOGGER.debug("Train Data :: Deriving Stats...")
        self.users_train = [str(user_id) for user_id in self.train_data[self.user_id_col].unique()]
        LOGGER.debug("Train Data :: No. of users : " + str(len(self.users_train)))
        self.items_train = [str(item_id) for item_id in self.train_data[self.item_id_col].unique()]
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
            users = [str(user) for user in item_users['users']]
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
            items = [str(item) for item in user_items['items']]
            self.user_items_train_dict[user] = items
        user_items_train_file = os.path.join(self.model_dir, 'user_items_train.json')
        utilities.dump_json_file(self.user_items_train_dict, user_items_train_file)
        ########################################################################
        LOGGER.debug("Test Data  :: Deriving Stats...")
        self.users_test = [str(user_id) for user_id in self.test_data[self.user_id_col].unique()]
        LOGGER.debug("Test Data  :: No. of users : " + str(len(self.users_test)))
        self.items_test = [str(item_id) for item_id in self.test_data[self.item_id_col].unique()]
        LOGGER.debug("Test Data  :: No. of items : " + str(len(self.items_test)))

        users_items_test_dict = {
            'users_test' : self.users_test,
            'items_test' : self.items_test
        }
        users_items_test_file = os.path.join(self.model_dir, 'users_items_test.json')
        utilities.dump_json_file(users_items_test_dict, users_items_test_file)

        LOGGER.debug("Test Data  :: Getting Distinct Users for each Item")
        item_users_test_df = self.test_data.groupby([self.item_id_col])\
                                             .agg({
                                                 self.user_id_col: (lambda x: list(x.unique()))
                                                 })
        item_users_test_df = item_users_test_df.rename(columns={self.user_id_col: 'users'})\
                                                      .reset_index()
        for _, item_users in item_users_test_df.iterrows():
            item = item_users[str(self.item_id_col)]
            users = [str(user) for user in item_users['users']]
            self.item_users_test_dict[item] = users

        item_users_test_file = os.path.join(self.model_dir, 'item_users_test.json')
        utilities.dump_json_file(self.item_users_test_dict, item_users_test_file)

        LOGGER.debug("Test Data  :: Getting Distinct Items for each User")
        user_items_test_df = self.test_data.groupby([self.user_id_col])\
                                             .agg({
                                                 self.item_id_col: (lambda x: list(x.unique()))
                                                 })
        user_items_test_df = user_items_test_df.rename(columns={self.item_id_col: 'items'})\
                                                      .reset_index()
        for _, user_items in user_items_test_df.iterrows():
            user = user_items[str(self.user_id_col)]
            items = [str(item) for item in user_items['items']]
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
        LOGGER.debug("Test Data  :: Loading Stats...")
        users_items_test_file = os.path.join(self.model_dir, 'users_items_test.json')
        users_items_test_dict = utilities.load_json_file(users_items_test_file)
        self.users_test = users_items_test_dict['users_test']
        LOGGER.debug("Test Data  :: No. of users : " + str(len(self.users_test)))
        self.items_test = users_items_test_dict['items_test']
        LOGGER.debug("Test Data  :: No. of items : " + str(len(self.items_test)))

        LOGGER.debug("Test Data  :: Loading Distinct Users for each Item")
        item_users_test_file = os.path.join(self.model_dir, 'item_users_test.json')
        self.item_users_test_dict = utilities.load_json_file(item_users_test_file)

        LOGGER.debug("Test Data  :: Loading Distinct Items for each User")
        user_items_test_file = os.path.join(self.model_dir, 'user_items_test.json')
        self.user_items_test_dict = utilities.load_json_file(user_items_test_file)

    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, no_of_recs=10):
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
        self.similar_users = None
        self.uim_df = None
        self.model_file = os.path.join(self.model_dir, 'user_based_model.pkl')

    def __get_items(self, user_id, dataset='train'):
        """private function, Get unique items for a given user"""
        if dataset == "train":
            user_items = self.user_items_train_dict[user_id]
        else:#test
            user_items = self.user_items_test_dict[user_id]
        return user_items

    def __get_users(self, item_id, dataset='train'):
        """private function, Get unique users for a given item"""
        if dataset == "train":
            item_users = self.item_users_train_dict[item_id]
        else:#test
            item_users = self.item_users_test_dict[item_id]
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

    def __derive_similar_users(self, jaccard_df):
        """fetch ranked similar users for each user"""
        similar_users_dict = dict()
        for user_id, similar_users in jaccard_df.iterrows():
            #print(user_id)
            #print(similar_users)
            sorted_similar_users = similar_users.sort_values(ascending=False)
            most_similar_users = (sorted_similar_users.drop(user_id))

            ranked_similar_users = dict()
            rank = 1
            for similar_user_id, score in most_similar_users.iteritems():
                #print(user_id, rank, similar_user_id, score)
                ranked_similar_users[rank] = {similar_user_id : score}
                rank = rank + 1
            similar_users_dict[user_id] = ranked_similar_users

        self.similar_users = similar_users_dict
        similar_users_file = os.path.join(self.model_dir, 'similar_users.json')
        utilities.dump_json_file(self.similar_users, similar_users_file)

    def __load_similar_users(self):
        """load ranked similar users for each user"""
        similar_users_file = os.path.join(self.model_dir, 'similar_users.json')
        self.similar_users = utilities.load_json_file(similar_users_file)

    def __load_uim(self):
        """load user interaction matrix"""
        uim_df_fname = os.path.join(self.model_dir, 'uim.csv')
        self.uim_df = pd.read_csv(uim_df_fname)

    def __construct_cooccurence_matrix(self):
        """private function, Construct cooccurence matrix"""
        #Construct User Item Matrix
        self.uim_df = pd.get_dummies(self.train_data[self.item_id_col])\
                        .groupby(self.train_data[self.user_id_col])\
                        .apply(max)
        uim_df_fname = os.path.join(self.model_dir, 'uim.csv')
        self.uim_df.to_csv(uim_df_fname)

        uim = self.uim_df.as_matrix()
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

        #stats
        items = list(self.uim_df.columns)
        no_of_items = len(items)
        users = list(self.uim_df.index)
        no_of_users = len(users)
        print("No of Items : ", no_of_items)
        print("No of Users : ", no_of_users)
        non_zero_count = np.count_nonzero(uim)
        count = uim.size
        density = non_zero_count/count
        print("Density of User Item Matrix : ", density)

        #Compute User-User Similarity Matrix with intersection of items interacted
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

        #Compute User-User Similarity Matrix with union of items interacted
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

        #Compute User-User Similarity Matrix with Jaccard Similarity of items
        jaccard = intersection/union
        jaccard_df = pd.DataFrame(jaccard,
                                  columns=users,
                                  index=users)
        jaccard_df_fname = os.path.join(self.model_dir,
                                        'jaccard.csv')
        jaccard_df.to_csv(jaccard_df_fname, index=False)

        self.__derive_similar_users(jaccard_df)

        return jaccard_df

    def train(self):
        """Train the user similarity based recommender system model"""
        self.__derive_stats()
        print("Training...")
        # Construct user cooccurence matrix of size, len(users) X len(users)
        start_time = default_timer()
        self.cooccurence_matrix_df = self.__construct_cooccurence_matrix()
        end_time = default_timer()
        print("{:50}    {}".format("Training Completed in : ",
                                   utilities.convert_sec(end_time - start_time)))
        #print(self.cooccurence_matrix_df.shape)
        joblib.dump(self.cooccurence_matrix_df, self.model_file)
        LOGGER.debug("Saved Model")

    def __get_similar_users(self, user_id):
        similar_user_scores = self.similar_users[user_id].values()
        #print(similar_user_scores)
        similar_users = dict()
        for similar_user_score in similar_user_scores:
            for user_id, score in similar_user_score.items():
                similar_users[user_id] = score
        #pprint(similar_users)
        return similar_users

    def __generate_top_recommendations(self, user_id, user_items):
        """Use the cooccurence matrix to make top recommendations"""
        # Calculate a weighted average of the scores in cooccurence matrix for
        # all user items.

        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']

        similar_users = self.__get_similar_users(user_id)
        #pprint(similar_users)
        similar_users_weights = pd.Series(similar_users)
        #print(similar_users_weights)
        similar_user_ids = similar_users_weights.index
        #print(similar_user_ids)

        sub_uim_df = self.uim_df.loc[similar_user_ids]
        print(sub_uim_df.head())
        weighted_sub_uim_df = sub_uim_df.mul(similar_users_weights, axis='index')
        print(weighted_sub_uim_df.head())

        no_of_similar_users = sub_uim_df.shape[0]
        if no_of_similar_users != 0:
            item_scores = sub_uim_df.sum(axis=0) / float(no_of_similar_users)
            item_scores.sort_values(inplace=True, ascending=False)
            item_scores = item_scores[item_scores > 0]

            rank = 1
            for item_id, score in item_scores.items():
                if item_id in user_items:#to avoid items which user has already aware
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

    def recommend_items(self, user_id, dataset='test'):
        """Generate item recommendations for given user_id from chosen dataset"""
        self.__load_stats()
        self.__load_similar_users()
        self.__load_uim()

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
        recommended_items = list(user_recommendations[self.item_id_col].values)
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated in : ",
                                   utilities.convert_sec(end_time - start_time)))
        return recommended_items

    def __split_items(self, items_interacted, hold_out_ratio):
        """return assume_interacted_items, hold_out_items"""
        #items_interacted_set = set(items_interacted)
        items_interacted_set = set()
        for i in items_interacted:
            items_interacted_set.add(i)

        assume_interacted_items = set()
        hold_out_items = set()
        # print("Items Interacted : ")
        # print(items_interacted)
        
        no_of_items_interacted = len(items_interacted_set)
        no_of_items_to_be_held = int(no_of_items_interacted*hold_out_ratio)
        hold_out_items = set(list(items_interacted_set)[-no_of_items_to_be_held:])
        #hold_out_items = set(self.get_random_sample(items_interacted, hold_out_ratio))
        
        
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

    def __get_items_for_eval(self, dataset='test', hold_out_ratio=0.5):
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
            eval_items[user_id]['assume_interacted_items'] = []
            eval_items[user_id]['items_interacted'] = []
            no_of_users_considered += 1
            user_recommendations = self.__generate_top_recommendations(user_id,
                                                                       assume_interacted_items)
            recommended_items = list(user_recommendations[self.item_id_col].values)
            eval_items[user_id]['items_recommended'] = recommended_items
            eval_items[user_id]['assume_interacted_items'] = assume_interacted_items
            eval_items[user_id]['items_interacted'] = hold_out_items
        print("Evaluation : No of users : ", no_of_users)
        print("Evaluation : No of users considered : ", no_of_users_considered)
        return eval_items

    def evaluate(self, no_of_recs_to_eval, dataset='test', hold_out_ratio=0.5):
        """Evaluate trained model"""
        print("Evaluating...")
        self.__load_stats()
        self.__load_similar_users()
        self.__load_uim()

        start_time = default_timer()
        if os.path.exists(self.model_file):
            self.cooccurence_matrix_df = joblib.load(self.model_file)
            #print(self.cooccurence_matrix_df.shape)
            LOGGER.debug("Loaded Trained Model")

            #Generate recommendations for the users
            eval_items = self.__get_items_for_eval(dataset, hold_out_ratio)
            precision_recall_eval_file = os.path.join(self.model_dir, 'eval_items.json')
            utilities.dump_json_file(eval_items, precision_recall_eval_file)
            #pprint(eval_items)

            precision_recall_intf = PrecisionRecall()
            results = precision_recall_intf.compute_precision_recall(
                no_of_recs_to_eval, eval_items)
            end_time = default_timer()
            print("{:50}    {}".format("Evaluation Completed in : ",
                                       utilities.convert_sec(end_time - start_time)))
            
            results_file = os.path.join(self.model_dir, 'results.json')
            utilities.dump_json_file(results, results_file)
            
            return results
        else:
            print("Trained Model not found !!!. Failed to evaluate")
            results = {'status' : "Trained Model not found !!!. Failed to evaluate"}
            end_time = default_timer()
            print("{:50}    {}".format("Evaluation Completed in : ",
                                       utilities.convert_sec(end_time - start_time)))
            
            results_file = os.path.join(self.model_dir, 'results.json')
            utilities.dump_json_file(results, results_file)
            
            return results

def train(results_dir, model_dir, train_test_dir,
          user_id_col, item_id_col,
          no_of_recs=10):
    """train recommender"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    print("Training Recommender...")
    model = UserBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col, no_of_recs)
    model.train()
    print('*' * 80)

def evaluate(results_dir, model_dir, train_test_dir,
             user_id_col, item_id_col,
             no_of_recs_to_eval, dataset='test',
             no_of_recs=10, hold_out_ratio=0.5):
    """evaluate recommender"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    print("Evaluating Recommender System")
    model = UserBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col, no_of_recs)
    results = model.evaluate(no_of_recs_to_eval, dataset, hold_out_ratio)
    pprint(results)
    print('*' * 80)

def recommend(results_dir, model_dir, train_test_dir,
              user_id_col, item_id_col,
              user_id, no_of_recs=10, dataset='test', metadata_fields=None):
    """recommend items for user"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    model = UserBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col, no_of_recs)

    print("Items interactions for a user with user_id : {}".format(user_id))
    interacted_items = list(test_data[test_data[user_id_col] == user_id][item_id_col])
    for item in interacted_items:
        print(item)
        if (metadata_fields is not None):
            record = test_data[test_data[item_id_col] == item]
            if not record.empty:
                for field in metadata_fields:
                    print("\t{} : {}".format(field, record[field].values[0]))
            
    print()
    print("Items recommended for a user with user_id : {}".format(user_id))
    recommended_items = model.recommend_items(user_id, dataset)
    print()
    if recommended_items:
        for item in recommended_items:
            print(item)
            if (metadata_fields is not None):
                record = train_data[train_data[item_id_col] == item]
                if not record.empty:
                    for field in metadata_fields:
                        print("\t{} : {}".format(field, record[field].values[0]))
    else:
        print("No items to recommend")
    print('*' * 80)

def train_eval_recommend(results_dir, model_dir, train_test_dir,
                         user_id_col, item_id_col,
                         no_of_recs_to_eval, dataset='test',
                         no_of_recs=10, hold_out_ratio=0.5):
    """Train Evaluate and Recommend for Item Based Recommender"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    print("Training Recommender...")
    model = UserBasedCFRecommender(results_dir, model_dir,
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
    recommended_items = model.recommend_items(user_id, dataset)
    print()
    for item in recommended_items:
        print(item)
    print('*' * 80)
