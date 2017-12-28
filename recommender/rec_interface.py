"""Module for Recommender"""
import os
import sys
import random
import logging

from abc import ABCMeta, abstractmethod

import pandas as pd

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities

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
    def evaluate(self, no_of_recs_to_eval):
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

def load_train_test(train_test_dir, user_id_col, item_id_col):
    """Loads data and returns training and test set"""
    print("Loading Training and Test Data")
    train_data_file = os.path.join(train_test_dir, 'train_data.csv')
    if os.path.exists(train_data_file):
        train_data = pd.read_csv(train_data_file, dtype=object)
    else:
        print("Unable to find train data in : ", train_data_file)
        exit(0)

    test_data_file = os.path.join(train_test_dir, 'test_data.csv')
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