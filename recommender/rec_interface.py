"""Module for Recommender"""
import os
import sys
import random
import logging
from timeit import default_timer

from pprint import pprint
from abc import ABCMeta, abstractmethod
from shutil import copyfile

import pandas as pd

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.aggregate import Aggregator
from recommender.evaluation import PrecisionRecall

class Recommender(metaclass=ABCMeta):
    """Abstract Base Class Interface"""
    def derive_stats(self):
        """derive stats"""
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

    def load_stats(self):
        """load stats"""
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
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        self.results_dir = results_dir
        self.model_dir = model_dir

        self.train_data = train_data
        self.test_data = test_data
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.no_of_recs = kwargs['no_of_recs']
        self.hold_out_ratio = kwargs['hold_out_ratio']

        self.users_train = None
        self.items_train = None
        self.user_items_train_dict = dict()
        self.item_users_train_dict = dict()

        self.users_test = None
        self.items_test = None
        self.user_items_test_dict = dict()
        self.item_users_test_dict = dict()

        self.items_for_evaluation = None
        self.test_data_for_evaluation = None

    def get_all_users(self, dataset='train'):
        """Get unique users in the dataset"""
        if dataset == "train":
            return self.users_train
        else:#test
            return self.users_test

    def get_all_items(self, dataset='train'):
        """Get unique items in the dataset"""
        if dataset == "train":
            return self.items_train
        else:#test
            return self.items_test

    def get_items(self, user_id, dataset='train'):
        """Get unique items for a given user_id in the dataset"""
        if dataset == "train":
            user_items = self.user_items_train_dict[user_id]
        else:#test
            user_items = self.user_items_test_dict[user_id]
        return user_items

    def get_users(self, item_id, dataset='train'):
        """Get unique users for a given item_id in the dataset"""
        if dataset == "train":
            item_users = self.item_users_train_dict[item_id]
        else:#test
            item_users = self.item_users_test_dict[item_id]
        return item_users

    def get_known_items(self, items_interacted):
        """return filtered items which are present in training set"""
        known_items_interacted = []
        items_training_set = self.get_all_items(dataset='train')
        for item in items_interacted:
            if item in items_training_set:
                known_items_interacted.append(item)
        return known_items_interacted

    def split_items(self, items_interacted, hold_out_ratio):
        """return assume_interacted_items, hold_out_items"""
        #print("items_interacted : ", items_interacted)

        assume_interacted_items = []
        hold_out_items = []

        no_of_items_interacted = len(items_interacted)
        no_of_items_to_be_held = int(no_of_items_interacted*hold_out_ratio)
        no_of_items_assumed_interacted = no_of_items_interacted - no_of_items_to_be_held
        #print("no_of_items_interacted : ", no_of_items_interacted)
        #print("no_of_items_to_be_held : ", no_of_items_to_be_held)
        #print("no_of_items_assumed_interacted : ", no_of_items_assumed_interacted)

        if no_of_items_assumed_interacted != 0:
            assume_interacted_items = items_interacted[:no_of_items_assumed_interacted]
        if no_of_items_to_be_held != 0:
            hold_out_items = items_interacted[-no_of_items_to_be_held:]

        #print("assume_interacted_items : ", assume_interacted_items)
        #print("hold_out_items : ", hold_out_items)
        #input()
        return assume_interacted_items, hold_out_items

    def save_items_for_evaluation(self):
        """save items to be considered for evaluation for each test user id"""
        items_for_evaluation_file = os.path.join(self.model_dir, 'items_for_evaluation.json')
        utilities.dump_json_file(self.items_for_evaluation, items_for_evaluation_file)

    def load_items_for_evaluation(self):
        """load items to be considered for evaluation for each test user id"""
        items_for_evaluation_file = os.path.join(self.model_dir, 'items_for_evaluation.json')
        self.items_for_evaluation = utilities.load_json_file(items_for_evaluation_file)

    def get_test_data_for_evaluation(self):
        """get test data for evaluation by deriving items to be considered
        for each test user id"""
        self.items_for_evaluation = dict()
        users = self.get_all_users(dataset='test')
        no_of_users = len(users)
        no_of_users_considered = 0
        self.test_data_for_evaluation = None
        for user_id in users:
            #print(user_id)
            # Get all items with which user has interacted
            items_interacted = self.get_items(user_id, dataset='test')
            # Filter items which are found in train data
            items_interacted = self.get_known_items(items_interacted)
            assume_interacted_items, hold_out_items = self.split_items(items_interacted,
                                                                       self.hold_out_ratio)
            if len(assume_interacted_items) == 0 or len(hold_out_items) == 0:
                # print("WARNING !!!. User {} exempted from evaluation".format(user_id))
                # print("Items Interacted Assumed : ")
                # print(assume_interacted_items)
                # print("Hold Out Items")
                # print(hold_out_items)
                # input()
                continue
            '''
            print(user_id)
            print("assume_interacted_items")
            print(assume_interacted_items)
            
            print("hold_out_items")
            print(hold_out_items)
            print()
            '''

            self.items_for_evaluation[user_id] = dict()
            self.items_for_evaluation[user_id]['items_recommended'] = []
            self.items_for_evaluation[user_id]['assume_interacted_items'] = assume_interacted_items
            self.items_for_evaluation[user_id]['items_interacted'] = hold_out_items
            no_of_users_considered += 1

            tmp = self.test_data[self.test_data[self.user_id_col] == user_id]
            filter_tmp = tmp.loc[tmp[self.item_id_col].isin(assume_interacted_items)]
            #print("filter_tmp")
            #print(filter_tmp)
            if self.test_data_for_evaluation is None:
                self.test_data_for_evaluation = filter_tmp.copy()
            else:
                self.test_data_for_evaluation = self.test_data_for_evaluation.append(filter_tmp, ignore_index=True)
            #print("self.test_data_for_evaluation")
            #print(self.test_data_for_evaluation[[self.user_id_col, self.item_id_col]])
        print("No of test users : ", no_of_users)
        print("No of test users considered for evaluation : ", len(self.items_for_evaluation))
        self.save_items_for_evaluation()
        return self.test_data_for_evaluation

    @abstractmethod
    def train(self):
        """train recommender"""
        self.derive_stats()
        self.get_test_data_for_evaluation()

    @abstractmethod
    def recommend_items(self, user_id):
        """recommend items for given user_id from test dataset"""
        self.load_stats()
        self.load_items_for_evaluation()

    @abstractmethod
    def evaluate(self, no_of_recs_to_eval, eval_res_file='evaluation_results.json'):
        """evaluate trained model for different no of ranked recommendations"""
        self.load_stats()
        self.load_items_for_evaluation()

    def get_random_sample(self, list_a, percentage):
        """return random percentage of values from a list"""
        k = int(len(list_a) * percentage)
        random.seed(0)
        indicies = random.sample(range(len(list_a)), k)
        new_list = [list_a[i] for i in indicies]
        return new_list

    def fetch_sample_test_users(self, percentage):
        """return test sample of users"""
        #Find users common between training and test set
        users_set_train_data = set(self.train_data[self.user_id_col].unique())
        users_set_test_data = set(self.test_data[self.user_id_col].unique())
        users_test_and_training = list(users_set_test_data.intersection(users_set_train_data))
        print("No of users common between training and test set : {}"\
             .format(len(users_test_and_training)))

        #Take only random user_sample of common users for evaluation
        users_test_sample = self.get_random_sample(users_test_and_training, percentage)
        print("Sample no of common users, used for evaluation : {}".format(len(users_test_sample)))
        return users_test_sample

class HybridRecommender():
    """Hybrid Recommender to combine recommendations"""
    def __init__(self, recommenders,
                 results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col,
                 **kwargs):
        """constructor"""
        self.results_dir = results_dir
        self.model_dir = model_dir

        self.train_data = train_data
        self.test_data = test_data
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col

        self.no_of_recs = kwargs['no_of_recs']
        self.hold_out_ratio = kwargs['hold_out_ratio']

        self.recommender_kwargs = dict(kwargs)
        self.recommender_kwargs['no_of_recs'] = 100

        self.items_for_evaluation = None

        self.recommenders = recommenders
        self.recommender_objs = []
        for recommender in self.recommenders:
            # print(recommender.__name__)
            recommender_model_dir = os.path.join(self.model_dir,
                                                 recommender.__name__)
            if not os.path.exists(recommender_model_dir):
                os.makedirs(recommender_model_dir)
            recommender_obj = recommender(self.results_dir,
                                          recommender_model_dir,
                                          self.train_data,
                                          self.test_data,
                                          self.user_id_col,
                                          self.item_id_col,
                                          **self.recommender_kwargs)
            self.recommender_objs.append(recommender_obj)

    def train(self):
        """train individual recommender"""
        for recommender_obj in self.recommender_objs:
            print("Training using : ", type(recommender_obj).__name__)
            recommender_obj.train()
            print('*' * 80)
        #copy eval file of first recommender(as it should be same for all recommenders)
        model_eval_items_file = os.path.join(self.model_dir,
                                             list(self.recommenders.keys())[0].__name__,
                                             'items_for_evaluation.json')
        eval_items_file = os.path.join(self.model_dir, 'items_for_evaluation.json')
        copyfile(model_eval_items_file, eval_items_file)

    def recommend_items(self, user_id, user_interacted_items):
        """combine items recommended for user from given set of recommenders"""
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']
        recommendations = dict()
        # get recommendations from each recommender
        for recommender_obj in self.recommender_objs:
            print("Recommending using : ", type(recommender_obj).__name__)
            user_recommendations = recommender_obj.recommend_items(user_id)
            for _, row in user_recommendations.iterrows():
                item_id = row[self.item_id_col]
                score = row['score']
                if item_id not in recommendations:
                    recommendations[item_id] = dict()
                    for rec_obj in self.recommender_objs:
                        recommendations[item_id][type(rec_obj).__name__] = 0.0
                recommendations[item_id][type(recommender_obj).__name__] = score

        # get weighted avg of recommendation scores for each item
        aggregation_items = []
        for item_id in recommendations:
            record = dict()
            record['item_id'] = item_id
            for rec_obj in self.recommender_objs:
                record[type(rec_obj).__name__] = recommendations[item_id][type(rec_obj).__name__]
            aggregation_items.append(record)
        aggregation_df = pd.DataFrame(aggregation_items)
        #print(aggregation_df.head())
        column_weights_dict = dict()
        for rec, weight in self.recommenders.items():
            column_weights_dict[rec.__name__] = weight
        aggregate_file = os.path.join(self.model_dir, 'scores_aggregation.csv')
        res_aggregator = Aggregator(aggregation_df, aggregate_file)
        aggregation_results = res_aggregator.weighted_avg(column_weights_dict)
        #print(aggregation_results.head())
        if aggregation_results is not None:
            rank = 1
            for _, res in aggregation_results.iterrows():
                item_id = res['item_id']
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

    def __save_items_for_evaluation(self):
        """save items to be considered for evaluation for each test user id"""
        items_for_evaluation_file = os.path.join(self.model_dir, 'items_for_evaluation.json')
        utilities.dump_json_file(self.items_for_evaluation, items_for_evaluation_file)

    def __load_items_for_evaluation(self):
        """load items to be considered for evaluation for each test user id"""
        items_for_evaluation_file = os.path.join(self.model_dir, 'items_for_evaluation.json')
        self.items_for_evaluation = utilities.load_json_file(items_for_evaluation_file)

    def __recommend_items_to_evaluate(self):
        """recommend items for all users from test dataset"""
        self.__load_items_for_evaluation()
        for user_id in self.items_for_evaluation:
            assume_interacted_items = self.items_for_evaluation[user_id]['assume_interacted_items']
            user_recommendations = self.recommend_items(user_id,
                                                        assume_interacted_items)

            recommended_items = list(user_recommendations[self.item_id_col].values)
            self.items_for_evaluation[user_id]['items_recommended'] = recommended_items
        return self.items_for_evaluation

    def evaluate(self, no_of_recs_to_eval, eval_res_file='evaluation_results.json'):
        """evaluate recommendations"""
        start_time = default_timer()
        #Generate recommendations for the users
        self.items_for_evaluation = self.__recommend_items_to_evaluate()
        self.__save_items_for_evaluation()

        precision_recall_intf = PrecisionRecall()
        evaluation_results = precision_recall_intf.compute_precision_recall(
            no_of_recs_to_eval, self.items_for_evaluation)
        end_time = default_timer()
        print("{:50}    {}".format("Evaluation Completed. ",
                                   utilities.convert_sec(end_time - start_time)))

        results_file = os.path.join(self.model_dir, eval_res_file)
        utilities.dump_json_file(evaluation_results, results_file)

        return evaluation_results

def load_train_test(train_data_file, test_data_file, user_id_col, item_id_col):
    """Loads data and returns training and test set"""
    print("Loading Training and Test Data")
    if os.path.exists(train_data_file):
        train_data = pd.read_csv(train_data_file, dtype=object)
    else:
        print("Unable to find train data in : ", train_data_file)
        exit(0)

    if os.path.exists(test_data_file):
        test_data = pd.read_csv(test_data_file, dtype=object)
    else:
        print("Unable to find test data in : ", train_data_file)
        exit(0)

    print("{:30} : {}".format("Train Data : No of records", len(train_data)))
    print("{:30} : {}".format("Train Data : No of users  ",
                              len(train_data[user_id_col].unique())))
    print("{:30} : {}".format("Train Data : No of items  ",
                              len(train_data[item_id_col].unique())))
    print()
    print("{:30} : {}".format("Test Data  : No of records", len(test_data)))
    print("{:30} : {}".format("Test Data  : No of users  ",
                              len(test_data[user_id_col].unique())))
    print("{:30} : {}".format("Test Data  : No of items  ",
                              len(test_data[item_id_col].unique())))
    print('#' * 40)
    return train_data, test_data

def train(recommender_obj,
          results_dir, model_dir,
          train_data_file, test_data_file,
          user_id_col, item_id_col,
          **kwargs):
    """train recommender"""
    train_data, test_data = load_train_test(train_data_file,
                                            test_data_file,
                                            user_id_col,
                                            item_id_col)
    recommender = recommender_obj(results_dir, model_dir,
                                  train_data, test_data,
                                  user_id_col, item_id_col,
                                  **kwargs)
    recommender.train()
    print('*' * 80)

def recommend(recommender_obj,
              results_dir, model_dir,
              train_data_file, test_data_file,
              user_id_col, item_id_col,
              user_id, **kwargs):
    """recommend items for user"""
    train_data, test_data = load_train_test(train_data_file,
                                            test_data_file,
                                            user_id_col,
                                            item_id_col)
    recommender = recommender_obj(results_dir, model_dir,
                                  train_data, test_data,
                                  user_id_col, item_id_col,
                                  **kwargs)
    eval_items_file = os.path.join(model_dir, 'items_for_evaluation.json')
    eval_items = utilities.load_json_file(eval_items_file)
    if user_id in eval_items:
        assume_interacted_items = eval_items[user_id]['assume_interacted_items']
        items_interacted = eval_items[user_id]['items_interacted']

        print("Assumed Item interactions for a user with user_id : {}".format(user_id))
        for item in assume_interacted_items:
            print(item)

        print()
        print("Items to be interacted for a user with user_id : {}".format(user_id))
        for item in items_interacted:
            print(item)

        print()
        print("Items recommended for a user with user_id : {}".format(user_id))
        user_recommendations = recommender.recommend_items(user_id)
        recommended_items = list(user_recommendations[item_id_col].values)
        print()
        if recommended_items:
            for recommended_item in recommended_items:
                print(recommended_item)
        else:
            print("No items to recommend")
        print('*' * 80)
    else:
        print("""Cannot generate recommendations as either
              items assumed to be interacted or items held out are None""")

def evaluate(recommender_obj,
             results_dir, model_dir,
             train_data_file, test_data_file,
             user_id_col, item_id_col,
             no_of_recs_to_eval,
             eval_res_file, **kwargs):
    """evaluate recommender"""
    train_data, test_data = load_train_test(train_data_file,
                                            test_data_file,
                                            user_id_col,
                                            item_id_col)
    recommender = recommender_obj(results_dir, model_dir,
                                  train_data, test_data,
                                  user_id_col, item_id_col,
                                  **kwargs)
    evaluation_results = recommender.evaluate(no_of_recs_to_eval,
                                              eval_res_file)
    pprint(evaluation_results)
    print('*' * 80)
    return evaluation_results

def train_eval_recommend(recommender_obj,
                         results_dir, model_dir,
                         train_data_file, test_data_file,
                         user_id_col, item_id_col,
                         no_of_recs_to_eval,
                         **kwargs):
    """train, evaluate and recommend"""
    train_data, test_data = load_train_test(train_data_file,
                                            test_data_file,
                                            user_id_col,
                                            item_id_col)
    recommender = recommender_obj(results_dir, model_dir,
                                  train_data, test_data,
                                  user_id_col, item_id_col,
                                  **kwargs)
    print("Training Recommender...")
    recommender.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    evaluation_results = recommender.evaluate(no_of_recs_to_eval)
    pprint(evaluation_results)
    print('*' * 80)

    print("Testing Recommendation for an User")
    items_for_evaluation_file = os.path.join(model_dir, 'items_for_evaluation.json')
    items_for_evaluation = utilities.load_json_file(items_for_evaluation_file)
    users = list(items_for_evaluation.keys())
    user_id = users[0]
    user_recommendations = recommender.recommend_items(user_id)
    recommended_items = list(user_recommendations[item_id_col].values)
    print("Items recommended for a user with user_id : {}".format(user_id))
    if recommended_items:
        for item in recommended_items:
            print(item)
    else:
        print("No items to recommend")
    print('*' * 80)

def kfold_evaluation(recommender_obj,
                     kfolds,
                     results_dir, model_dir,
                     train_data_dir, test_data_dir,
                     user_id_col, item_id_col,
                     no_of_recs_to_eval, **kwargs):
    """train and evaluation for kfolds of data"""
    kfold_experiments = dict()
    for kfold in range(kfolds):
        kfold_exp = kfold+1
        train_data_file = os.path.join(train_data_dir, str(kfold_exp) + '_train_data.csv')
        test_data_file = os.path.join(test_data_dir, str(kfold_exp) + '_test_data.csv')
        print("Loading...")
        print(train_data_file)
        print(test_data_file)
        kfold_model_dir = os.path.join(model_dir,
                                       'kfold_experiments',
                                       'kfold_exp_' + str(kfold_exp))
        if not os.path.exists(kfold_model_dir):
            os.makedirs(kfold_model_dir)

        train(recommender_obj,
              results_dir, kfold_model_dir,
              train_data_file, test_data_file,
              user_id_col, item_id_col,
              **kwargs)

        kfold_eval_file = 'kfold_exp_' + str(kfold_exp) + '_evaluation.json'
        evaluation_results = evaluate(recommender_obj,
                                      results_dir, kfold_model_dir,
                                      train_data_file, test_data_file,
                                      user_id_col, item_id_col,
                                      no_of_recs_to_eval,
                                      eval_res_file=kfold_eval_file, **kwargs)
        kfold_experiments[kfold_exp] = evaluation_results

    avg_kfold_exp_res = get_avg_kfold_exp_res(kfold_experiments)
    print('average of kfold evaluation results')
    pprint(avg_kfold_exp_res)
    results_file = os.path.join(model_dir, 'kfold_experiments', 'kfold_evaluation.json')
    utilities.dump_json_file(avg_kfold_exp_res, results_file)

def get_avg_kfold_exp_res(kfold_experiments):
    """compute avg scores for all kfold experiments"""
    #pprint(kfold_experiments)
    avg_kfold_exp_res = dict()
    avg_kfold_exp_res['no_of_items_to_recommend'] = dict()
    for _, kfold_exp_res in kfold_experiments.items():
        for no_of_items, score in kfold_exp_res['no_of_items_to_recommend'].items():
            exp_avg_f1_score = score['avg_f1_score']
            exp_avg_precision = score['avg_precision']
            exp_avg_recall = score['avg_recall']
            if no_of_items not in avg_kfold_exp_res['no_of_items_to_recommend']:
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items] = dict()
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_f1_score'] = exp_avg_f1_score
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_precision'] = exp_avg_precision
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_recall'] = exp_avg_recall
            else:
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_f1_score'] += exp_avg_f1_score
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_precision'] += exp_avg_precision
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_recall'] += exp_avg_recall

    #print('total_kfold_exp_res:')
    #pprint(avg_kfold_exp_res)
    no_of_kfold_exp = len(kfold_experiments)
    for no_of_items, score in avg_kfold_exp_res['no_of_items_to_recommend'].items():
        avg_kfold_avg_f1_score = round(score['avg_f1_score'] / no_of_kfold_exp, 4)
        avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_f1_score'] = avg_kfold_avg_f1_score

        avg_kfold_avg_precision = round(score['avg_precision'] / no_of_kfold_exp, 4)
        avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_precision'] = avg_kfold_avg_precision

        avg_kfold_avg_recall = round(score['avg_recall'] / no_of_kfold_exp, 4)
        avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_recall'] = avg_kfold_avg_recall

    #print('avg_kfold_exp_res:')
    #pprint(avg_kfold_exp_res)
    return avg_kfold_exp_res

def hybrid_train(recommenders,
                 results_dir, model_dir,
                 train_data_file, test_data_file,
                 user_id_col, item_id_col,
                 **kwargs):
    """train given set of recommenders"""
    train_data, test_data = load_train_test(train_data_file,
                                            test_data_file,
                                            user_id_col,
                                            item_id_col)
    hybrid_recommender = HybridRecommender(recommenders,
                                           results_dir, model_dir,
                                           train_data, test_data,
                                           user_id_col, item_id_col,
                                           **kwargs)
    hybrid_recommender.train()

def hybrid_recommend(recommenders,
                     results_dir, model_dir,
                     train_data_file, test_data_file,
                     user_id_col, item_id_col,
                     user_id, **kwargs):
    """recommmed items using given set of recommenders"""
    train_data, test_data = load_train_test(train_data_file,
                                            test_data_file,
                                            user_id_col,
                                            item_id_col)
    hybrid_recommender = HybridRecommender(recommenders,
                                           results_dir, model_dir,
                                           train_data, test_data,
                                           user_id_col, item_id_col,
                                           **kwargs)

    eval_items_file = os.path.join(model_dir, 'items_for_evaluation.json')
    eval_items = utilities.load_json_file(eval_items_file)
    if user_id in eval_items:
        assume_interacted_items = eval_items[user_id]['assume_interacted_items']
        items_interacted = eval_items[user_id]['items_interacted']

        print("Assumed Item interactions for a user with user_id : {}".format(user_id))
        for item_id in assume_interacted_items:
            print(item_id)

        print()
        print("Items to be interacted for a user with user_id : {}".format(user_id))
        for item_id in items_interacted:
            print(item_id)

        print()
        print("Items recommended for a user with user_id : {}".format(user_id))
        user_recommendations = hybrid_recommender.recommend_items(user_id,
                                                                  assume_interacted_items)
        #print(user_recommendations)
        recommended_items = list(user_recommendations[item_id_col].values)
        print()
        if recommended_items:
            for recommended_item in recommended_items:
                print(recommended_item)
        else:
            print("No items to recommend")
        print('*' * 80)
    else:
        print("""Cannot generate recommendations as either
              items assumed to be interacted or items held out are None""")

def hybrid_evaluate(recommenders,
                    results_dir, model_dir,
                    train_data_file, test_data_file,
                    user_id_col, item_id_col,
                    no_of_recs_to_eval,
                    eval_res_file, **kwargs):
    """evaluate recommended items using given set of recommenders"""
    train_data, test_data = load_train_test(train_data_file,
                                            test_data_file,
                                            user_id_col,
                                            item_id_col)
    hybrid_recommender = HybridRecommender(recommenders,
                                           results_dir, model_dir,
                                           train_data, test_data,
                                           user_id_col, item_id_col,
                                           **kwargs)
    evaluation_results = hybrid_recommender.evaluate(no_of_recs_to_eval,
                                                     eval_res_file)
    pprint(evaluation_results)
    print('*' * 80)
    return evaluation_results

def hybrid_kfold_evaluation(recommenders,
                            kfolds,
                            results_dir, model_dir,
                            train_data_dir, test_data_dir,
                            user_id_col, item_id_col,
                            no_of_recs_to_eval, **kwargs):
    """train and evaluation for kfolds of data"""
    kfold_experiments = dict()
    for kfold in range(kfolds):
        kfold_exp = kfold+1
        train_data_file = os.path.join(train_data_dir, str(kfold_exp) + '_train_data.csv')
        test_data_file = os.path.join(test_data_dir, str(kfold_exp) + '_test_data.csv')
        print("Loading...")
        print(train_data_file)
        print(test_data_file)
        kfold_model_dir = os.path.join(model_dir,
                                       'kfold_experiments',
                                       'kfold_exp_' + str(kfold_exp))
        if not os.path.exists(kfold_model_dir):
            os.makedirs(kfold_model_dir)

        hybrid_train(recommenders,
                     results_dir, kfold_model_dir,
                     train_data_file, test_data_file,
                     user_id_col, item_id_col,
                     **kwargs)

        kfold_eval_file = 'kfold_exp_' + str(kfold_exp) + '_evaluation.json'
        evaluation_results = hybrid_evaluate(recommenders,
                                             results_dir, kfold_model_dir,
                                             train_data_file, test_data_file,
                                             user_id_col, item_id_col,
                                             no_of_recs_to_eval,
                                             eval_res_file=kfold_eval_file, **kwargs)
        kfold_experiments[kfold_exp] = evaluation_results

    avg_kfold_exp_res = get_avg_kfold_exp_res(kfold_experiments)
    print('average of kfold evaluation results')
    pprint(avg_kfold_exp_res)
    results_file = os.path.join(model_dir, 'kfold_experiments', 'kfold_evaluation.json')
    utilities.dump_json_file(avg_kfold_exp_res, results_file)

def hybrid_train_eval_recommend(recommenders,
                                results_dir, model_dir,
                                train_data_file, test_data_file,
                                user_id_col, item_id_col,
                                no_of_recs_to_eval,
                                **kwargs):
    """train, evaluate and recommend"""
    train_data, test_data = load_train_test(train_data_file,
                                            test_data_file,
                                            user_id_col,
                                            item_id_col)
    hybrid_recommender = HybridRecommender(recommenders,
                                           results_dir, model_dir,
                                           train_data, test_data,
                                           user_id_col, item_id_col,
                                           **kwargs)
    print("Training Recommender...")
    hybrid_recommender.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    evaluation_results = hybrid_recommender.evaluate(no_of_recs_to_eval)
    pprint(evaluation_results)
    print('*' * 80)

    print("Testing Recommendation for an User")
    items_for_evaluation_file = os.path.join(model_dir, 'items_for_evaluation.json')
    items_for_evaluation = utilities.load_json_file(items_for_evaluation_file)
    users = list(items_for_evaluation.keys())
    user_id = users[0]
    assume_interacted_items = items_for_evaluation[user_id]['assume_interacted_items']
    user_recommendations = hybrid_recommender.recommend_items(user_id, assume_interacted_items)
    recommended_items = list(user_recommendations[item_id_col].values)
    print("Items recommended for a user with user_id : {}".format(user_id))
    if recommended_items:
        for item in recommended_items:
            print(item)
    else:
        print("No items to recommend")
    print('*' * 80)
