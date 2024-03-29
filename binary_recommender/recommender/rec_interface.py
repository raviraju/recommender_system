"""Module for Recommender"""
import os
import sys
import random
import logging
from timeit import default_timer

from pprint import pprint
from abc import ABCMeta, abstractmethod
from fnmatch import fnmatch
from pathlib import Path
from shutil import copyfile

import pandas as pd
import numpy as np

import json
from pprint import pprint

from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.aggregate import Aggregator
from recommender.evaluation import PrecisionRecall

class Recommender(metaclass=ABCMeta):
    """Abstract Base Class Interface"""
    def get_users_no_of_items(self, dataset='train'):
        if dataset == "train":
            users_items = self.train_data.groupby(self.user_id_col)[self.item_id_col].unique()
        else:#test
            users_items = self.test_data.groupby(self.user_id_col)[self.item_id_col].unique()
        users_no_of_items = users_items.apply(lambda x: len(x))
        return users_no_of_items

    def get_user_groups(self, dataset='train'):
        users_no_of_items = self.get_users_no_of_items(dataset)
        users_no_of_items_dist = users_no_of_items.describe()
        # print(users_no_of_items_dist)
        quartiles = []
        if users_no_of_items_dist['min'] != users_no_of_items_dist['25%']:
            quartiles.append((users_no_of_items_dist['min'], users_no_of_items_dist['25%']))
        if users_no_of_items_dist['25%'] != users_no_of_items_dist['50%']:
            quartiles.append((users_no_of_items_dist['25%'], users_no_of_items_dist['50%']))
        if users_no_of_items_dist['50%'] != users_no_of_items_dist['75%']:
            quartiles.append((users_no_of_items_dist['50%'], users_no_of_items_dist['75%']))
        if users_no_of_items_dist['75%'] != users_no_of_items_dist['max']:
            quartiles.append((users_no_of_items_dist['75%'], users_no_of_items_dist['max']+1))
        # print(quartiles)
        
        # all_users_set = set()
        user_groups_dict = dict()
        i = 0 
        for quartile in quartiles:
            min_items, max_items = quartile
            user_group = dict()
            grp_id = 'user_grp_' + str(i+1)
            i += 1
            user_group['usage'] = '{:3d}  <= no_of_items <  {}'.format(int(min_items), int(max_items))        
            user_group['users'] = list(set(users_no_of_items[(users_no_of_items>=min_items) & \
                                                        (users_no_of_items<max_items)].index))
            user_group['no_of_users'] = len(user_group['users'])
            # all_users_set = all_users_set.union(user_group['users'])
    
            # print("\t", grp_id, ' ', user_group['no_of_users'], ' users interacted with ', user_group['usage'])
            user_groups_dict[grp_id] = user_group
                                                
        # print("Total No of Users : ", len(all_users_set))
        return user_groups_dict

    def derive_stats(self):
        """derive stats"""
        LOGGER.debug("Train Data :: Deriving Stats...")
        self.users_train = [str(user_id) for user_id in self.train_data[self.user_id_col].unique()]
        train_users_set = set(self.users_train)
        LOGGER.debug("Train Data :: No. of users : " + str(len(self.users_train)))
        self.items_train = [str(item_id) for item_id in self.train_data[self.item_id_col].unique()]
        train_items_set = set(self.items_train)
        LOGGER.debug("Train Data :: No. of items : " + str(len(self.items_train)))

        users_items_train_dict = {
            'users_train' : self.users_train,
            'items_train' : self.items_train
        }
        users_items_train_file = os.path.join(self.model_dir, 'users_items_train.json')
        utilities.dump_json_file(users_items_train_dict, users_items_train_file)

        LOGGER.debug("Test Data  :: Deriving Stats...")
        self.users_test = [str(user_id) for user_id in self.test_data[self.user_id_col].unique()]
        test_users_set = set(self.users_test)
        LOGGER.debug("Test Data  :: No. of users : " + str(len(self.users_test)))
        self.items_test = [str(item_id) for item_id in self.test_data[self.item_id_col].unique()]
        test_items_set = set(self.items_test)
        LOGGER.debug("Test Data  :: No. of items : " + str(len(self.items_test)))

        common_users_set = train_users_set.intersection(test_users_set)
        common_items_set = train_items_set.intersection(test_items_set)
    
        new_users_in_test_set = test_users_set - train_users_set
        new_items_in_test_set = test_items_set - train_items_set
    
        print("{:30} : {}".format("Train Data   : No of records", len(self.train_data)))    
        print("{:30} : {}".format("Train Data   : No of users  ", len(train_users_set)))    
        print("{:30} : {}".format("Train Data   : No of items  ", len(train_items_set)))
        print()
        print("{:30} : {}".format("Test Data    : No of records", len(self.test_data)))
        print("{:30} : {}".format("Test Data    : No of users  ", len(test_users_set)))
        print("{:30} : {}".format("Test Data    : No of items  ", len(test_items_set)))
        print()
        print("{:30} : {}".format("Common Data  : No of users  ", len(common_users_set)))
        print("{:30} : {}".format("Common Data  : No of items  ", len(common_items_set)))
        print()
        print("{:30} : {}".format("New in Test  : No of users  ", len(new_users_in_test_set)))
        print("{:30} : {}".format("New in Test  : No of items  ", len(new_items_in_test_set)))
        print('#' * 40)
        # train_items = set(train_items_set)
        # test_items = set(test_data[item_id_col].unique())
        # print("train_items-test_items : ", len(train_items - test_items))
        # print("test_items-train_items : ", len(test_items - train_items))

        users_items_test_dict = {
            'users_test' : self.users_test,
            'items_test' : self.items_test
        }
        users_items_test_file = os.path.join(self.model_dir, 'users_items_test.json')
        utilities.dump_json_file(users_items_test_dict, users_items_test_file)

        LOGGER.debug("All Data   :: Deriving Stats...")
        self.users_all = list(set(self.users_train) | set(self.users_test))
        LOGGER.debug("All Data   :: No. of users : " + str(len(self.users_all)))

        self.items_all = list(set(self.items_train) | set(self.items_test))
        LOGGER.debug("All Data   :: No. of items : " + str(len(self.items_all)))
        users_items_all_dict = {
            'users_all' : self.users_all,
            'items_all' : self.items_all
        }
        users_items_all_file = os.path.join(self.model_dir, 'users_items_all.json')
        utilities.dump_json_file(users_items_all_dict, users_items_all_file)
        #############################################################################################
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
        ########################################################################
        # LOGGER.debug("Train Data :: Getting User Groups")
        # self.train_user_groups_dict = self.get_user_groups(dataset='train')
        # user_groups_train_file = os.path.join(self.model_dir, 'user_groups_train.json')
        # utilities.dump_json_file(self.train_user_groups_dict, user_groups_train_file)

        LOGGER.debug("Test Data  :: Getting User Groups")
        self.test_user_groups_dict = self.get_user_groups(dataset='test')
        user_groups_test_file = os.path.join(self.model_dir, 'user_groups_test.json')
        utilities.dump_json_file(self.test_user_groups_dict, user_groups_test_file)

        #Identify User Group of Test User
        for test_user_id in self.items_for_evaluation:
            for user_grp in self.test_user_groups_dict:
                users = self.test_user_groups_dict[user_grp]['users']
                if test_user_id in users:
                    self.items_for_evaluation[test_user_id][user_grp] = dict()
                    self.items_for_evaluation[test_user_id][user_grp]['usage'] = self.test_user_groups_dict[user_grp]['usage']
                    self.items_for_evaluation[test_user_id][user_grp]['no_of_users'] = self.test_user_groups_dict[user_grp]['no_of_users']
                    break
        
        self.save_items_for_evaluation()

    def load_stats(self):
        """load stats"""
        LOGGER.debug("All Data   :: Loading Stats...")
        users_items_all_file = os.path.join(self.model_dir, 'users_items_all.json')
        users_items_all_dict = utilities.load_json_file(users_items_all_file)
        self.users_all = users_items_all_dict['users_all']
        LOGGER.debug("All Data   :: No. of users : " + str(len(self.users_all)))
        self.items_all = users_items_all_dict['items_all']
        LOGGER.debug("All Data   :: No. of items : " + str(len(self.items_all)))

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

        self.hold_out_strategy = "assume_ratio"
        if 'hold_out_strategy' in kwargs:
            self.hold_out_strategy = kwargs['hold_out_strategy']

        self.assume_first_n = None
        self.assume_ratio = None
        self.hold_last_n = None
        if 'assume_first_n' in kwargs:
            self.assume_first_n = kwargs['assume_first_n']
        if 'assume_ratio' in kwargs:
            self.assume_ratio = kwargs['assume_ratio']
        if 'hold_last_n' in kwargs:
            self.hold_last_n = kwargs['hold_last_n']

        self.users_all = None
        self.items_all = None

        self.users_train = None
        self.items_train = None
        self.user_items_train_dict = dict()
        self.item_users_train_dict = dict()

        self.users_test = None
        self.items_test = None
        self.user_items_test_dict = dict()
        self.item_users_test_dict = dict()

        self.items_for_evaluation = None
        self.known_interactions_from_test_df = None
        self.allow_recommending_known_items = False

    def get_all_users(self, dataset='train'):
        """Get unique users in the dataset"""
        if dataset == "train":
            return self.users_train
        elif dataset == "test":
            return self.users_test
        else:
            return self.users_all

    def get_all_items(self, dataset='train'):
        """Get unique items in the dataset"""
        if dataset == "train":
            return self.items_train
        elif dataset == "test":
            return self.items_test
        else:
            return self.items_all

    def get_items(self, user_id, dataset='train'):
        """Get unique items for a given user_id in the dataset"""
        user_items = []
        if dataset == "train":
            if user_id in self.user_items_train_dict: 
                user_items = self.user_items_train_dict[user_id]
        else:#test
            if user_id in self.user_items_test_dict: 
                user_items = self.user_items_test_dict[user_id]
        return user_items

    def get_users(self, item_id, dataset='train'):
        """Get unique users for a given item_id in the dataset"""
        item_users = []
        if dataset == "train":
            if item_id in self.item_users_train_dict: 
                item_users = self.item_users_train_dict[item_id]
        else:#test
            if item_id in self.item_users_test_dict:
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

    def split_items_assume_ratio(self, items_interacted, assume_ratio):
        """return assume_interacted_items, hold_out_items by assuming assume_ratio of items to be interacted"""        
        assume_interacted_items = []
        hold_out_items = []

        no_of_items_interacted = len(items_interacted)
        no_of_items_assumed_interacted = int(no_of_items_interacted*assume_ratio)
        # no_of_items_to_be_held = no_of_items_interacted - no_of_items_assumed_interacted
        # print("no_of_items_interacted : ", no_of_items_interacted)
        # print("no_of_items_assumed_interacted : ", no_of_items_assumed_interacted)
        # print("no_of_items_to_be_held : ", no_of_items_to_be_held)        

        assume_interacted_items = items_interacted[:no_of_items_assumed_interacted]
        hold_out_items = items_interacted[no_of_items_assumed_interacted:]

        # print("len(items_interacted) : ", no_of_items_interacted)
        # print("len(assume_interacted_items) : ", len(assume_interacted_items))
        # print("len(hold_out_items) : ", len(hold_out_items))
        # input()
        return assume_interacted_items, hold_out_items
    
    def split_items_assume_first_n(self, items_interacted, assume_first_n = 10):
        """return assume_interacted_items, hold_out_items by assuming assume_first_n items to be interacted"""
        assume_interacted_items = []
        hold_out_items = []

        # no_of_items_interacted = len(items_interacted)
        no_of_items_assumed_interacted = assume_first_n
        # no_of_items_to_be_held = no_of_items_interacted - no_of_items_assumed_interacted
        # print("no_of_items_interacted : ", no_of_items_interacted)
        # print("no_of_items_assumed_interacted : ", no_of_items_assumed_interacted)
        # print("no_of_items_to_be_held : ", no_of_items_to_be_held)

        assume_interacted_items = items_interacted[:no_of_items_assumed_interacted]
        hold_out_items = items_interacted[no_of_items_assumed_interacted:]

        # print("len(items_interacted) : ", no_of_items_interacted)
        # print("len(assume_interacted_items) : ", len(assume_interacted_items))
        # print("len(hold_out_items) : ", len(hold_out_items))
        # input()
        return assume_interacted_items, hold_out_items

    def split_items_hold_last_n(self, items_interacted, hold_last_n=10):
        """return assume_interacted_items, hold_out_items by holding out hold_last_n and rest as items interacted"""
        assume_interacted_items = []
        hold_out_items = []

        no_of_items_interacted = len(items_interacted)
        no_of_items_to_be_held = hold_last_n
        no_of_items_assumed_interacted = no_of_items_interacted - no_of_items_to_be_held
        #print("no_of_items_interacted : ", no_of_items_interacted)
        #print("no_of_items_assumed_interacted : ", no_of_items_assumed_interacted)
        #print("no_of_items_to_be_held : ", no_of_items_to_be_held)        

        assume_interacted_items = items_interacted[:no_of_items_assumed_interacted]
        hold_out_items = items_interacted[no_of_items_assumed_interacted:]

        # print("len(items_interacted) : ", no_of_items_interacted)
        # print("len(assume_interacted_items) : ", len(assume_interacted_items))
        # print("len(hold_out_items) : ", len(hold_out_items))
        # input()
        return assume_interacted_items, hold_out_items

    def save_items_for_evaluation(self):
        """save items to be considered for evaluation for each test user id"""
        items_for_evaluation_file = os.path.join(self.model_dir, 'items_for_evaluation.json')
        utilities.dump_json_file(self.items_for_evaluation, items_for_evaluation_file)

    def load_items_for_evaluation(self):
        """load items to be considered for evaluation for each test user id"""
        items_for_evaluation_file = os.path.join(self.model_dir, 'items_for_evaluation.json')
        self.items_for_evaluation = utilities.load_json_file(items_for_evaluation_file)
        for test_user_id in self.items_for_evaluation:
            if self.items_for_evaluation[test_user_id]['is_items_to_be_interacted_known']:
                self.allow_recommending_known_items = True
                break

    def prepare_test_data_for_eval(self):
        """get test user interactions for evaluation by deriving items to be considered for each user according to hold_out_stratergy"""        
        user_items_train_df = self.train_data.groupby([self.user_id_col])\
                                             .agg({
                                                 self.item_id_col: (lambda x: list(x.unique()))
                                                 })
        user_items_test_df = self.test_data.groupby([self.user_id_col])\
                                             .agg({
                                                 self.item_id_col: (lambda x: list(x.unique()))
                                                 })
        self.items_for_evaluation = dict()
        self.known_interactions_from_test_df = None
        self.held_out_interactions_from_test_df = None
        for test_user_id in user_items_test_df.index:
            #print(test_user_id)
            # Get all items with which user has interacted in test
            items_interacted_in_test = user_items_test_df.loc[test_user_id].values[0]

            if self.hold_out_strategy == "assume_ratio":
                assume_interacted_items, hold_out_items = self.split_items_assume_ratio(items_interacted_in_test,
                                                                           self.assume_ratio)
            elif self.hold_out_strategy == "assume_first_n":
                assume_interacted_items, hold_out_items = self.split_items_assume_first_n(items_interacted_in_test,
                                                                                          self.assume_first_n)
            elif self.hold_out_strategy == "hold_last_n":
                assume_interacted_items, hold_out_items = self.split_items_hold_last_n(items_interacted_in_test,
                                                                                       self.hold_last_n)
            elif self.hold_out_strategy == "hold_all":
                assume_interacted_items = []
                hold_out_items = items_interacted_in_test
            else:
                print("Invalid hold_out strategy!!!")
                exit(-1)

            if len(hold_out_items) == 0:
            # if len(assume_interacted_items) == 0 or len(hold_out_items) == 0:
                # print("WARNING !!!. User {} exempted from evaluation".format(user_id))
                # print("Items Interacted Assumed : ")
                # print(assume_interacted_items)
                # print("Hold Out Items")
                # print(hold_out_items)
                # input()
                continue
            '''
            print(test_user_id)
            print("assume_interacted_items")
            print(assume_interacted_items)
            
            print("hold_out_items")
            print(hold_out_items)
            print()
            '''            
            if test_user_id in user_items_train_df.index:
                items_interacted_in_train = user_items_train_df.loc[test_user_id].values[0]
            else:
                items_interacted_in_train = []            
            self.items_for_evaluation[test_user_id] = dict()
            self.items_for_evaluation[test_user_id]['items_interacted_in_train'] = items_interacted_in_train            
            self.items_for_evaluation[test_user_id]['assume_interacted_items'] = assume_interacted_items
            self.items_for_evaluation[test_user_id]['known_interacted_items'] = list(set(items_interacted_in_train).union(set(assume_interacted_items)))
            self.items_for_evaluation[test_user_id]['items_to_be_interacted'] = hold_out_items            
            self.items_for_evaluation[test_user_id]['items_recommended'] = []

            items_to_be_interacted_known = set(self.items_for_evaluation[test_user_id]['items_to_be_interacted']).intersection(set(self.items_for_evaluation[test_user_id]['known_interacted_items']))
            self.items_for_evaluation[test_user_id]['items_to_be_interacted_known'] = list(items_to_be_interacted_known)
            if len(self.items_for_evaluation[test_user_id]['items_to_be_interacted_known']) > 0:
                self.items_for_evaluation[test_user_id]['is_items_to_be_interacted_known'] = True
                self.allow_recommending_known_items = True
            else:
                self.items_for_evaluation[test_user_id]['is_items_to_be_interacted_known'] = False

            #Identify New Users in Test Data
            if test_user_id in user_items_train_df.index:
                self.items_for_evaluation[test_user_id]['is_new_user_in_test'] = False
            else:
                self.items_for_evaluation[test_user_id]['is_new_user_in_test'] = True

            if self.hold_out_strategy == "hold_all":
                self.known_interactions_from_test_df = pd.DataFrame()
                self.held_out_interactions_from_test_df = self.test_data
            else:
                test_user_df = self.test_data[self.test_data[self.user_id_col] == test_user_id]

                test_user_known_df = pd.DataFrame(test_user_df.loc[test_user_df[self.item_id_col].isin(assume_interacted_items)])
                if self.known_interactions_from_test_df is None:
                    self.known_interactions_from_test_df = test_user_known_df
                else:
                    self.known_interactions_from_test_df = pd.concat([self.known_interactions_from_test_df,
                                                                      test_user_known_df])
                
                test_user_held_out_df = pd.DataFrame(test_user_df.loc[test_user_df[self.item_id_col].isin(hold_out_items)])
                if self.held_out_interactions_from_test_df is None:
                    self.held_out_interactions_from_test_df = test_user_held_out_df
                else:
                    self.held_out_interactions_from_test_df = pd.concat([self.held_out_interactions_from_test_df,
                                                                         test_user_held_out_df])

        LOGGER.debug("Train Data Stats...")        
        LOGGER.debug("\tShape : " + str(self.train_data.shape))
        LOGGER.debug("\tNo of Users : " + str(len(self.train_data[self.user_id_col].unique())))
        LOGGER.debug("\tNo of Items : " + str(len(self.train_data[self.item_id_col].unique())))
        LOGGER.debug("Test Data Stats...")
        LOGGER.debug("\tShape : " + str(self.test_data.shape))
        LOGGER.debug("\tNo of Users : " + str(len(self.test_data[self.user_id_col].unique())))
        LOGGER.debug("\tNo of Items : " + str(len(self.test_data[self.item_id_col].unique())))
        
        LOGGER.debug("Combining train_data with interactions known from test...")
        #self.train_data = self.train_data.append(self.known_interactions_from_test_df, ignore_index=True)
        self.train_data = pd.concat([self.train_data, self.known_interactions_from_test_df])
        self.test_data = self.held_out_interactions_from_test_df

        LOGGER.debug("Train Data Stats...")        
        LOGGER.debug("\tShape : " + str(self.train_data.shape))
        LOGGER.debug("\tNo of Users : " + str(len(self.train_data[self.user_id_col].unique())))
        LOGGER.debug("\tNo of Items : " + str(len(self.train_data[self.item_id_col].unique())))
        LOGGER.debug("Test Data Stats...")
        LOGGER.debug("\tShape : " + str(self.test_data.shape))
        LOGGER.debug("\tNo of Users : " + str(len(self.test_data[self.user_id_col].unique())))
        LOGGER.debug("\tNo of Items : " + str(len(self.test_data[self.item_id_col].unique())))


    @abstractmethod
    def train(self):
        """train recommender"""        
        self.prepare_test_data_for_eval()
        self.derive_stats()

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

        self.hold_out_strategy = "assume_ratio"
        if 'hold_out_strategy' in kwargs:
            self.hold_out_strategy = kwargs['hold_out_strategy']

        self.assume_ratio = None
        self.assume_first_n = None
        self.next_n = None
        self.hold_last_n = None
        if 'assume_ratio' in kwargs:
            self.assume_ratio = kwargs['assume_ratio']
        if 'assume_first_n' in kwargs:
            self.assume_first_n = kwargs['assume_first_n']
        if 'next_n' in kwargs:
            self.next_n = kwargs['next_n']
        if 'hold_last_n' in kwargs:
            self.hold_last_n = kwargs['hold_last_n']


        self.recommender_kwargs = dict(kwargs)
        #self.recommender_kwargs['no_of_recs'] = 100
        self.allow_recommending_known_items = False

        self.items_for_evaluation = None

        self.aggregation_all_df = pd.DataFrame()

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

        model_users_items_all_file = os.path.join(self.model_dir,
                                                    list(self.recommenders.keys())[0].__name__,
                                                    'users_items_all.json')
        users_items_all_file = os.path.join(self.model_dir,
                                              'users_items_all.json')
        copyfile(model_users_items_all_file, users_items_all_file)

    def recommend_items(self, user_id, known_interacted_items):
        """combine items recommended for user from given set of recommenders"""
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']
        recommendations = dict()
        # get recommendations from each recommender
        for recommender_obj in self.recommender_objs:
            recommender_type = type(recommender_obj).__name__
            #print("Recommending using : ", recommender_type)
            user_recommendations = recommender_obj.recommend_items(user_id)
            for _, row in user_recommendations.iterrows():
                item_id = row[self.item_id_col]
                score = row['score']
                if item_id not in recommendations:
                    recommendations[item_id] = dict()
                    for rec_obj in self.recommender_objs:
                        recommendations[item_id][type(rec_obj).__name__] = 0.0
                recommendations[item_id][recommender_type] = score

        # get weighted avg of recommendation scores for each item
        aggregation_items = []
        for item_id in recommendations:
            record = dict()
            record['user_id'] = user_id
            record['item_id'] = item_id
            # scores = []
            for rec_obj in self.recommender_objs:
                recommender_type = type(rec_obj).__name__
                score = recommendations[item_id][recommender_type]
                record[recommender_type] = score
                # scores.append(score)
            # if sum(scores) == 0:#skip recommendations where score for each recommender is 0
            #     #print("skipping record")
            #     #pprint(record)
            #     #input()
            #     continue
            aggregation_items.append(record)
        aggregation_df = pd.DataFrame(aggregation_items)
        #print(aggregation_df.head())

        column_weights_dict = dict()
        for rec, weight in self.recommenders.items():
            column_weights_dict[rec.__name__] = weight

        res_aggregator = Aggregator(aggregation_df)
        aggregation_results = res_aggregator.weighted_avg(column_weights_dict)
        #print(aggregation_results.head())
        self.aggregation_all_df = self.aggregation_all_df.append(aggregation_results)

        if aggregation_results is not None:
            rank = 1
            for _, res in aggregation_results.iterrows():
                item_id = res['item_id']
                user_id = res['user_id']
                score = res['weighted_avg']
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

    def __get_item_viewed_status(self):
        """item viewed or not status for user"""
        users = self.aggregation_all_df['user_id'].unique()
        #print(len(users))
        self.aggregation_all_df['watched'] = 0
        #print(self.aggregation_all_df.head())
        for user_id in users:
            #print(user_id)
            items = self.aggregation_all_df[self.aggregation_all_df['user_id'] == user_id]['item_id'].unique()
            #print(len(items))
            items_interacted_in_test = self.recommender_objs[0].get_items(user_id, dataset='test')
            #print(len(items_interacted_in_test))

            filter_condition = (self.aggregation_all_df['user_id'] == user_id) & \
                               (self.aggregation_all_df['item_id'].isin(items_interacted_in_test))
            self.aggregation_all_df.loc[filter_condition, 'watched'] = 1
            #print(self.aggregation_all_df[(self.aggregation_all_df['user_id'] == user_id) & \
            #                              (self.aggregation_all_df['watched'] == 1)])
            #input()
        #print(self.aggregation_all_df.head())
        aggregate_file = os.path.join(self.model_dir, 'scores_aggregation.csv')
        #print(aggregate_file)
        #input()
        self.aggregation_all_df.to_csv(aggregate_file, index=False)

    def __recommend_items_to_evaluate(self):
        """recommend items for all users from test dataset"""
        self.__load_items_for_evaluation()
        for user_id in self.items_for_evaluation:
            known_interacted_items = self.items_for_evaluation[user_id]['known_interacted_items']
            user_recommendations = self.recommend_items(user_id, known_interacted_items)

            recommended_items = list(user_recommendations[self.item_id_col].values)
            self.items_for_evaluation[user_id]['items_recommended'] = recommended_items

            items_to_be_interacted_set = set(self.items_for_evaluation[user_id]['items_to_be_interacted'])
            items_recommended_set = set(recommended_items)
            correct_recommendations = items_to_be_interacted_set.intersection(items_recommended_set)
            no_of_correct_recommendations = len(correct_recommendations)
            self.items_for_evaluation[user_id]['no_of_correct_recommendations'] = no_of_correct_recommendations
            self.items_for_evaluation[user_id]['correct_recommendations'] = list(correct_recommendations)

            recommended_items_dict = dict()
            for _, recs in user_recommendations.iterrows():
                item_id = recs[self.item_id_col]
                score = recs['score']
                rank = recs['rank']
                recommended_items_dict[item_id] = {'score' : score, 'rank' : rank}
            self.items_for_evaluation[user_id]['items_recommended_score'] = recommended_items_dict
        self.__get_item_viewed_status()
        return self.items_for_evaluation

    def evaluate(self, no_of_recs_to_eval, eval_res_file='evaluation_results.json'):
        """evaluate recommendations"""
        start_time = default_timer()
        #Generate recommendations for the users
        self.items_for_evaluation = self.__recommend_items_to_evaluate()
        self.__save_items_for_evaluation()

        precision_recall_intf = PrecisionRecall()
        '''
        users_items_train_file = os.path.join(self.model_dir, 'users_items_train.json')
        users_items_train_dict = utilities.load_json_file(users_items_train_file)
        items_train = users_items_train_dict['items_train']
        evaluation_results = precision_recall_intf.compute_precision_recall(no_of_recs_to_eval,
                                                                            self.items_for_evaluation,
                                                                            items_train)
        '''
        users_items_all_file = os.path.join(self.model_dir, 'users_items_all.json')
        users_items_all_dict = utilities.load_json_file(users_items_all_file)
        items_all = users_items_all_dict['items_all']
        evaluation_results = precision_recall_intf.compute_precision_recall(no_of_recs_to_eval,
                                                                            self.items_for_evaluation,
                                                                            items_all)
        end_time = default_timer()
        print("{:50}    {}".format("Evaluation Completed. ",
                                   utilities.convert_sec(end_time - start_time)))

        results_file = os.path.join(self.model_dir, eval_res_file)
        utilities.dump_json_file(evaluation_results, results_file)

        return evaluation_results

def load_data(data_file):
    """Loads data and returns dataframe"""
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, dtype=object)
    else:
        print("Unable to find data_file : ", data_file)
        df = None
    return df

def train(recommender_obj,
          results_dir, model_dir,
          train_data_file, test_data_file,
          user_id_col, item_id_col,
          **kwargs):
    """train recommender"""
    print("Loading Train Data...")
    train_data = load_data(train_data_file)
    if train_data is None:
        exit(-1)
    print("Loading Test Data...")
    test_data = load_data(test_data_file)
    if test_data is None:
        exit(-1)

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
    print("Loading Train Data...")
    train_data = load_data(train_data_file)
    if train_data is None:
        exit(-1)
    print("Loading Test Data...")
    test_data = load_data(test_data_file)
    if test_data is None:
        exit(-1)    
    meta_data = None
    if 'meta_data_file' in kwargs:
        print("Loading Meta Data...")
        meta_data = load_data(kwargs['meta_data_file'])
        if meta_data is None:
            exit(-1)
    
    recommender = recommender_obj(results_dir, model_dir,
                                  train_data, test_data,
                                  user_id_col, item_id_col,
                                  **kwargs)
    eval_items_file = os.path.join(model_dir, 'items_for_evaluation.json')
    eval_items = utilities.load_json_file(eval_items_file)
    if user_id in eval_items:
        items_interacted_in_train = eval_items[user_id]['items_interacted_in_train']
        assume_interacted_items = eval_items[user_id]['assume_interacted_items']
        items_to_be_interacted = eval_items[user_id]['items_to_be_interacted']

        print("\nTrain Item interactions for a user with user_id   : {}".format(user_id))        
        if meta_data is not None:
            cols = [item_id_col]
            items_meta_data = meta_data[meta_data[item_id_col].isin(items_interacted_in_train)]
            if 'meta_data_fields' in kwargs:
                meta_data_fields = kwargs['meta_data_fields']
                cols.extend(meta_data_fields)
            print(items_meta_data[cols])
        else:
            for item in items_interacted_in_train:
                print(item)

        print("\nAssumed Item interactions for a user with user_id : {}".format(user_id))
        if meta_data is not None:
            cols = [item_id_col]
            items_meta_data = meta_data[meta_data[item_id_col].isin(assume_interacted_items)]
            if 'meta_data_fields' in kwargs:
                meta_data_fields = kwargs['meta_data_fields']
                cols.extend(meta_data_fields)
            print(items_meta_data[cols])
        else:
            for item in assume_interacted_items:
                print(item)

        print()
        print("\nItems to be interacted for a user with user_id    : {}".format(user_id))
        if meta_data is not None:
            cols = [item_id_col]
            items_meta_data = meta_data[meta_data[item_id_col].isin(items_to_be_interacted)]
            if 'meta_data_fields' in kwargs:
                meta_data_fields = kwargs['meta_data_fields']
                cols.extend(meta_data_fields)
            print(items_meta_data[cols])
        else:
            for item in items_to_be_interacted:
                print(item)

        print()
        print("\nTop {} Items recommended for a user with user_id  : {}".format(recommender.no_of_recs, user_id))
        items_to_recommend_df = recommender.recommend_items(user_id)
        if items_to_recommend_df is not None:
            recommended_items = list(items_to_recommend_df[item_id_col].values)
          
            if meta_data is not None and 'meta_data_fields' in kwargs:
                cols = [item_id_col]
                cols.extend(kwargs['meta_data_fields'])
                items_to_recommend_df = items_to_recommend_df.merge(meta_data[cols], how='inner')
                pprint(items_to_recommend_df)#.to_dict(orient='index'))
            else:                
                for item in recommended_items:
                    print(item)

            print()
            print("\nItems correctly recommended for a user with user_id  : {}".format(user_id))
            correct_recommendations = set(items_to_be_interacted).intersection(set(recommended_items))
            if meta_data is not None and 'meta_data_fields' in kwargs:
                correct_items_to_recommend_df = items_to_recommend_df[items_to_recommend_df[item_id_col].isin(correct_recommendations)]
                print(correct_items_to_recommend_df)
            else:
                for item in correct_recommendations:
                    print(item)
        else:
            print("No items to recommend")
        print('*' * 80)
    else:
        print("""Cannot generate recommendations as either items assumed to be interacted or items held out are None""")

def evaluate(recommender_obj,
             results_dir, model_dir,
             train_data_file, test_data_file,
             user_id_col, item_id_col,
             no_of_recs_to_eval,
             eval_res_file, **kwargs):
    """evaluate recommender"""
    print("Loading Train Data...")
    train_data = load_data(train_data_file)
    if train_data is None:
        exit(-1)
    print("Loading Test Data...")
    test_data = load_data(test_data_file)
    if test_data is None:
        exit(-1)
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
    print("Loading Train Data...")
    train_data = load_data(train_data_file)
    if train_data is None:
        exit(-1)
    print("Loading Test Data...")
    test_data = load_data(test_data_file)
    if test_data is None:
        exit(-1)
    meta_data = None
    if 'meta_data_file' in kwargs:
        print("Loading Meta Data...")
        meta_data = load_data(kwargs['meta_data_file'])
        if meta_data is None:
            exit(-1)

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

    print("One of the Best Recommendations")
    items_for_evaluation_file = os.path.join(model_dir, 'items_for_evaluation.json')
    items_for_evaluation = utilities.load_json_file(items_for_evaluation_file)
    users = list(items_for_evaluation.keys())

    best_user_id = users[0]
    max_no_of_correct_recommendations = 0
    for user_id in items_for_evaluation:
        no_of_correct_recommendations = items_for_evaluation[user_id]['no_of_correct_recommendations']
        if no_of_correct_recommendations > max_no_of_correct_recommendations:
            max_no_of_correct_recommendations = no_of_correct_recommendations
            best_user_id = user_id
    print("Top {} Items recommended for a user with user_id : {}".format(recommender.no_of_recs, best_user_id))
    items_to_recommend_df = recommender.recommend_items(best_user_id)        
    if items_to_recommend_df is not None:
        recommended_items = list(items_to_recommend_df[item_id_col].values)
      
        if meta_data is not None and 'meta_data_fields' in kwargs:
            cols = [item_id_col]
            cols.extend(kwargs['meta_data_fields'])
            items_to_recommend_df = items_to_recommend_df.merge(meta_data[cols], how='left')
            pprint(items_to_recommend_df)#.to_dict(orient='index'))
        else:                
            for item in recommended_items:
                print(item)

        items_to_be_interacted = items_for_evaluation[best_user_id]['items_to_be_interacted']
        print()
        print("\nItems correctly recommended for a user with user_id  : {}".format(best_user_id))
        correct_recommendations = set(items_to_be_interacted).intersection(set(recommended_items))
        if meta_data is not None and 'meta_data_fields' in kwargs:
            correct_items_to_recommend_df = items_to_recommend_df[items_to_recommend_df[item_id_col].isin(correct_recommendations)]
            print(correct_items_to_recommend_df)
        else:
            for item in correct_recommendations:
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
    avg_kfold_exp_res['total_no_of_test_users_considered_for_evaluation'] = 0
    for _, kfold_exp_res in kfold_experiments.items():
        no_of_test_users_considered_for_evaluation = kfold_exp_res['no_of_test_users_considered_for_evaluation']
        avg_kfold_exp_res['total_no_of_test_users_considered_for_evaluation'] += no_of_test_users_considered_for_evaluation
        for no_of_items, score in kfold_exp_res['no_of_items_to_recommend'].items():
            exp_avg_f1_score = score['avg_f1_score']
            exp_avg_mcc_score = score['avg_mcc_score']
            
            exp_avg_tpr = score['avg_tpr']
            exp_avg_fpr = score['avg_fpr']
            
            exp_avg_precision = score['avg_precision']
            exp_avg_recall = score['avg_recall']
            if no_of_items not in avg_kfold_exp_res['no_of_items_to_recommend']:
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items] = dict()
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_f1_score'] = exp_avg_f1_score
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_mcc_score'] = exp_avg_mcc_score
                
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_tpr'] = exp_avg_tpr
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_fpr'] = exp_avg_fpr
                
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_precision'] = exp_avg_precision
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_recall'] = exp_avg_recall
            else:
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_f1_score'] += exp_avg_f1_score
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_mcc_score'] += exp_avg_mcc_score
                
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_tpr'] += exp_avg_tpr
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_fpr'] += exp_avg_fpr
                
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_precision'] += exp_avg_precision
                avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_recall'] += exp_avg_recall

    #print('total_kfold_exp_res:')
    #pprint(avg_kfold_exp_res)
    no_of_kfold_exp = len(kfold_experiments)
    for no_of_items, score in avg_kfold_exp_res['no_of_items_to_recommend'].items():
        avg_kfold_avg_f1_score = round(score['avg_f1_score'] / no_of_kfold_exp, 4)
        avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_f1_score'] = avg_kfold_avg_f1_score

        avg_kfold_avg_mcc_score = round(score['avg_mcc_score'] / no_of_kfold_exp, 4)
        avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_mcc_score'] = avg_kfold_avg_mcc_score
        
        avg_kfold_avg_tpr = round(score['avg_tpr'] / no_of_kfold_exp, 4)
        avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_tpr'] = avg_kfold_avg_tpr

        avg_kfold_avg_fpr = round(score['avg_fpr'] / no_of_kfold_exp, 4)
        avg_kfold_exp_res['no_of_items_to_recommend'][no_of_items]['avg_fpr'] = avg_kfold_avg_fpr

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
    print("Loading Train Data...")
    train_data = load_data(train_data_file)
    if train_data is None:
        exit(-1)
    print("Loading Test Data...")
    test_data = load_data(test_data_file)
    if test_data is None:
        exit(-1)
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
    print("Loading Train Data...")
    train_data = load_data(train_data_file)
    if train_data is None:
        exit(-1)
    print("Loading Test Data...")
    test_data = load_data(test_data_file)
    if test_data is None:
        exit(-1)
    meta_data = None
    if 'meta_data_file' in kwargs:
        print("Loading Meta Data...")
        meta_data = load_data(kwargs['meta_data_file'])
        if meta_data is None:
            exit(-1)
    
    hybrid_recommender = HybridRecommender(recommenders,
                                           results_dir, model_dir,
                                           train_data, test_data,
                                           user_id_col, item_id_col,
                                           **kwargs)
    eval_items_file = os.path.join(model_dir, 'items_for_evaluation.json')
    eval_items = utilities.load_json_file(eval_items_file)
    if user_id in eval_items:
        items_interacted_in_train = eval_items[user_id]['items_interacted_in_train']
        assume_interacted_items = eval_items[user_id]['assume_interacted_items']
        known_interacted_items = eval_items[user_id]['known_interacted_items']
        items_to_be_interacted = eval_items[user_id]['items_to_be_interacted']

        print("\nTrain Item interactions for a user with user_id   : {}".format(user_id))        
        if meta_data is not None:
            cols = [item_id_col]
            items_meta_data = meta_data[meta_data[item_id_col].isin(items_interacted_in_train)]
            if 'meta_data_fields' in kwargs:
                meta_data_fields = kwargs['meta_data_fields']
                cols.extend(meta_data_fields)
            print(items_meta_data[cols])
        else:
            for item in items_interacted_in_train:
                print(item)

        print("\nAssumed Item interactions for a user with user_id : {}".format(user_id))
        if meta_data is not None:
            cols = [item_id_col]
            items_meta_data = meta_data[meta_data[item_id_col].isin(assume_interacted_items)]
            if 'meta_data_fields' in kwargs:
                meta_data_fields = kwargs['meta_data_fields']
                cols.extend(meta_data_fields)
            print(items_meta_data[cols])
        else:
            for item in assume_interacted_items:
                print(item)

        print()
        print("\nItems to be interacted for a user with user_id    : {}".format(user_id))
        if meta_data is not None:
            cols = [item_id_col]
            items_meta_data = meta_data[meta_data[item_id_col].isin(items_to_be_interacted)]
            if 'meta_data_fields' in kwargs:
                meta_data_fields = kwargs['meta_data_fields']
                cols.extend(meta_data_fields)
            print(items_meta_data[cols])
        else:
            for item in items_to_be_interacted:
                print(item)

        print()
        print("\nTop {} Items recommended for a user with user_id  : {}".format(hybrid_recommender.no_of_recs, user_id))
        items_to_recommend_df = hybrid_recommender.recommend_items(user_id, known_interacted_items)        
        if items_to_recommend_df is not None:
            recommended_items = list(items_to_recommend_df[item_id_col].values)
          
            if meta_data is not None and 'meta_data_fields' in kwargs:
                cols = [item_id_col]
                cols.extend(kwargs['meta_data_fields'])
                items_to_recommend_df = items_to_recommend_df.merge(meta_data[cols], how='inner')
                pprint(items_to_recommend_df)#.to_dict(orient='index'))
            else:                
                for item in recommended_items:
                    print(item)

            print()
            print("\nItems correctly recommended for a user with user_id  : {}".format(user_id))
            correct_recommendations = set(items_to_be_interacted).intersection(set(recommended_items))
            if meta_data is not None and 'meta_data_fields' in kwargs:
                correct_items_to_recommend_df = items_to_recommend_df[items_to_recommend_df[item_id_col].isin(correct_recommendations)]
                print(correct_items_to_recommend_df)
            else:
                for item in correct_recommendations:
                    print(item)
        else:
            print("No items to recommend")
        print('*' * 80)
    else:
        print("""Cannot generate recommendations as either items assumed to be interacted or items held out are None""")

def hybrid_evaluate(recommenders,
                    results_dir, model_dir,
                    train_data_file, test_data_file,
                    user_id_col, item_id_col,
                    no_of_recs_to_eval,
                    eval_res_file, **kwargs):
    """evaluate recommended items using given set of recommenders"""
    print("Loading Train Data...")
    train_data = load_data(train_data_file)
    if train_data is None:
        exit(-1)
    print("Loading Test Data...")
    test_data = load_data(test_data_file)
    if test_data is None:
        exit(-1)
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

def hybrid_train_eval_recommend(recommenders,
                                results_dir, model_dir,
                                train_data_file, test_data_file,
                                user_id_col, item_id_col,
                                no_of_recs_to_eval,
                                **kwargs):
    """train, evaluate and recommend"""
    print("Loading Train Data...")
    train_data = load_data(train_data_file)
    if train_data is None:
        exit(-1)
    print("Loading Test Data...")
    test_data = load_data(test_data_file)
    if test_data is None:
        exit(-1)
    meta_data = None
    if 'meta_data_file' in kwargs:
        print("Loading Meta Data...")
        meta_data = load_data(kwargs['meta_data_file'])
        if meta_data is None:
            exit(-1)

    hybrid_recommender = HybridRecommender(recommenders,
                                           results_dir, model_dir,
                                           train_data, test_data,
                                           user_id_col, item_id_col,
                                           **kwargs)
    print(model_dir)
    print("Training Recommender...")
    hybrid_recommender.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    evaluation_results = hybrid_recommender.evaluate(no_of_recs_to_eval)
    pprint(evaluation_results)
    print('*' * 80)

    print("One of the Best Recommendations")
    items_for_evaluation_file = os.path.join(model_dir, 'items_for_evaluation.json')
    items_for_evaluation = utilities.load_json_file(items_for_evaluation_file)
    users = list(items_for_evaluation.keys())

    best_user_id = users[0]
    max_no_of_correct_recommendations = 0
    for user_id in items_for_evaluation:
        no_of_correct_recommendations = items_for_evaluation[user_id]['no_of_correct_recommendations']
        if no_of_correct_recommendations > max_no_of_correct_recommendations:
            max_no_of_correct_recommendations = no_of_correct_recommendations
            best_user_id = user_id
    known_interacted_items = items_for_evaluation[best_user_id]['known_interacted_items']
    print("Top {} Items recommended for a user with user_id : {}".format(hybrid_recommender.no_of_recs, best_user_id))
    items_to_recommend_df = hybrid_recommender.recommend_items(user_id, known_interacted_items)
    if items_to_recommend_df is not None:
        recommended_items = list(items_to_recommend_df[item_id_col].values)
      
        if meta_data is not None and 'meta_data_fields' in kwargs:
            cols = [item_id_col]
            cols.extend(kwargs['meta_data_fields'])
            items_to_recommend_df = items_to_recommend_df.merge(meta_data[cols], how='left')
            pprint(items_to_recommend_df)#.to_dict(orient='index'))
        else:                
            for item in recommended_items:
                print(item)

        items_to_be_interacted = items_for_evaluation[best_user_id]['items_to_be_interacted']
        print()
        print("\nItems correctly recommended for a user with user_id  : {}".format(best_user_id))
        correct_recommendations = set(items_to_be_interacted).intersection(set(recommended_items))
        if meta_data is not None and 'meta_data_fields' in kwargs:
            correct_items_to_recommend_df = items_to_recommend_df[items_to_recommend_df[item_id_col].isin(correct_recommendations)]
            print(correct_items_to_recommend_df)
        else:
            for item in correct_recommendations:
                print(item)
    else:
        print("No items to recommend")
    print('*' * 80)

def hybrid_kfold_evaluation(recommenders,
                            kfolds,
                            results_dir, model_dir,
                            train_data_dir, test_data_dir,
                            user_id_col, item_id_col,
                            no_of_recs_to_eval, **kwargs):
    """train and evaluation for kfolds of data"""
    kfold_experiments = dict()
    all_items_for_evaluation = dict()
    scores_aggregation_df_list = []
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

        scores_aggregation_df = pd.read_csv(os.path.join(kfold_model_dir,
                                                         'scores_aggregation.csv'))
        #print(len(scores_aggregation_df['user_id'].unique()))
        scores_aggregation_df_list.append(scores_aggregation_df)

        items_for_evaluation = utilities.load_json_file(os.path.join(kfold_model_dir,
                                                                     'items_for_evaluation.json'))
        all_items_for_evaluation.update(items_for_evaluation)

    all_scores_aggregation_df = pd.concat(scores_aggregation_df_list, axis=0)
    #print(len(all_scores_aggregation_df['user_id'].unique()))
    all_scores_aggregation_file = os.path.join(model_dir, 'kfold_experiments', 'all_scores_aggregation.csv')
    all_scores_aggregation_df.to_csv(all_scores_aggregation_file)

    all_items_for_evaluation_file = os.path.join(model_dir, 'kfold_experiments', 'all_items_for_evaluation.json')
    #print(len(all_items_for_evaluation.keys()))
    utilities.dump_json_file(all_items_for_evaluation, all_items_for_evaluation_file)

    avg_kfold_exp_res = get_avg_kfold_exp_res(kfold_experiments)
    print('average of kfold evaluation results')
    pprint(avg_kfold_exp_res)
    results_file = os.path.join(model_dir, 'kfold_experiments', 'kfold_evaluation.json')
    utilities.dump_json_file(avg_kfold_exp_res, results_file)

def check_sanity(weights):
    """perform sanity checks on aggregation weights"""
    max_weight = weights[0]
    for weight in weights:
        if weight > max_weight:
            max_weight = weight
    """Ensure each weight is non-zero"""
    if 0.0 in weights:
        print("One of the weights is zero")
        print(weights)
        new_weights = []
        for weight in weights:
            if weight == 0:
                new_weight = weight + 0.01
            elif weight == max_weight:
                new_weight = weight - 0.01
            else:
                new_weight = weight
            new_weights.append(round(new_weight, 2))
        weights = new_weights
        print("corrected weights : ")
        print(weights)
        #input()

    max_weight = weights[0]
    for weight in weights:
        if weight > max_weight:
            max_weight = weight
    """Sum of Weights to be 1.0"""
    sum_of_weights = round(np.sum(weights), 2)
    #print("sum_of_weights : ", sum_of_weights)
    if sum_of_weights != 1.0:
        print("aggregation weights {}, do not sum to 1.0".format(sum_of_weights))
        #remove the delta from highest weight
        difference = sum_of_weights - 1.0

        new_weights = []
        for weight in weights:
            if weight == max_weight:
                new_weight = weight - difference
            else:
                new_weight = weight
            new_weights.append(round(new_weight, 2))
        weights = new_weights
        print(weights)
        print("corrected weights sum to : ", round(np.sum(weights), 2))

    return weights

def get_aggregation_weights(scores_aggregation_file_path):
    """train logistic regression and return normalized model coefficients as weights"""
    print("Train logistic regression and fetch normalized model coefficients as weights...")
    df = pd.read_csv(scores_aggregation_file_path)
    df.rename(index=str, columns={'Unnamed: 0':'id'}, inplace=True)
    #print(df.head())

    recommender_features = []
    for col in df.columns:
        if 'Recommender' in col:
            recommender_features.append(col)

    #print(recommender_features)
    target = 'watched'
    X = df[recommender_features]
    y = df[target]

    logreg = LogisticRegression(class_weight='balanced')
    logreg.fit(X, y)
    coefficients = logreg.coef_[0]
    #print("coefficients : ", coefficients)
    weights = np.round(coefficients/np.sum(coefficients), 2)
    #print("weights : ",  weights)
    weights = check_sanity(weights)

    aggregation_weights = dict()
    for rec, weight in zip(recommender_features, weights):
        #print(rec, weight)
        aggregation_weights[rec] = weight
    #pprint(aggregation_weights)
    return aggregation_weights

def generate_weighted_recommendations(dest_all_items_for_evaluation_file, aggregation_results, no_of_recs, allow_recommending_known_items=False):
    """generate recommendations based on weighted avg scores"""
    with open(dest_all_items_for_evaluation_file, 'r') as json_file:
        all_items_for_evaluation = json.load(json_file)
        new_all_items_for_evaluation = dict()
        for user_id in all_items_for_evaluation:
            known_interacted_items = all_items_for_evaluation[user_id]['known_interacted_items']
            items_to_be_interacted = all_items_for_evaluation[user_id]['items_to_be_interacted']
            new_all_items_for_evaluation[user_id] = dict()
            new_all_items_for_evaluation[user_id]['known_interacted_items'] = known_interacted_items
            new_all_items_for_evaluation[user_id]['items_to_be_interacted'] = items_to_be_interacted
        #pprint(new_all_items_for_evaluation)
        aggregation_results['user_id'] = aggregation_results['user_id'].astype(str)
        aggregation_results['item_id'] = aggregation_results['item_id'].astype(str)

        for user_id in new_all_items_for_evaluation:
            items_to_recommend = []
            known_interacted_items = new_all_items_for_evaluation[user_id]['known_interacted_items']
            user_agg_results = aggregation_results[aggregation_results['user_id'] == user_id]

            recommended_items_dict = dict()
            if user_agg_results is not None:
                rank = 1
                for _, res in user_agg_results.iterrows():
                    item_id = res['item_id']
                    user_id = res['user_id']
                    score = res['weighted_avg']                    
                    if not allow_recommending_known_items and item_id in known_interacted_items:#to avoid items which user has already aware
                        #print("Skipping : ", item_id)
                        continue
                    if rank > no_of_recs:#limit no of recommendations
                        break
                    item_dict = {
                        'user_id' : user_id,
                        'item_id' : item_id,
                        'score' : score,
                        'rank' : rank
                    }
                    #print(user_id, item_id, score, rank)
                    items_to_recommend.append(item_dict)
                    recommended_items_dict[item_id] = {'score' : score, 'rank' : rank}
                    rank += 1
            res_df = pd.DataFrame(items_to_recommend)
            #print(res_df)
            recommended_items = list(res_df['item_id'].values)
            new_all_items_for_evaluation[user_id]['items_recommended'] = recommended_items

            items_to_be_interacted_set = set(new_all_items_for_evaluation[user_id]['items_to_be_interacted'])
            items_recommended_set = set(recommended_items)
            correct_recommendations = items_to_be_interacted_set & items_recommended_set
            no_of_correct_recommendations = len(correct_recommendations)
            new_all_items_for_evaluation[user_id]['no_of_correct_recommendations'] = no_of_correct_recommendations
            new_all_items_for_evaluation[user_id]['correct_recommendations'] = list(correct_recommendations)

            new_all_items_for_evaluation[user_id]['items_recommended_score'] = recommended_items_dict
            #pprint(new_all_items_for_evaluation[user_id])
            #input()
    return new_all_items_for_evaluation

def hybrid_evaluation_using_auto_weights(models_dir, all_items, item_id_col, no_of_recs, no_of_recs_to_eval):
    """parse hybrid recommendation files to aggregate based on weights generated by ML model"""
    #print(len(all_items))
    for root, _, files in os.walk(models_dir, topdown=False):
        for name in files:
            if fnmatch(name, 'scores_aggregation.csv'):
                src_scores_aggregation_file_path = (os.path.join(root, name))
                model_dir = Path(src_scores_aggregation_file_path).parent
                model_parent_dir = Path(model_dir).parent
                if ('hybrid' in model_dir.name):
                #if ('all_recommenders' in model_dir.name):
                    print("\nComputing Aggregation Weights for Model")
                    model_name = model_dir.name.replace('equal', 'logit')
                    #print(model_name)
                    logit_dir = str(model_dir).replace('equal', 'logit')
                    #print(logit_dir)
                    if not os.path.exists(logit_dir):
                        os.makedirs(logit_dir)
                    src_items_for_evaluation_file = os.path.join(model_dir,
                                                                     'items_for_evaluation.json')
                    dest_items_for_evaluation_file = os.path.join(logit_dir,
                                                                     'items_for_evaluation.json')
                    #print(src_items_for_evaluation_file, dest_items_for_evaluation_file)
                    copyfile(src_items_for_evaluation_file, dest_items_for_evaluation_file)
                    #print(dest_items_for_evaluation_file)

                    dest_scores_aggregation_file_path = os.path.join(logit_dir,
                                                                     'scores_aggregation.csv')
                    #print(src_scores_aggregation_file_path, dest_scores_aggregation_file_path)
                    copyfile(src_scores_aggregation_file_path, dest_scores_aggregation_file_path)
                    #print(dest_scores_aggregation_file_path)

                    config = dict()
                    config['model_dir_name'] = model_name
                    print("Loading {} ...".format(dest_scores_aggregation_file_path))
                    aggregation_weights = get_aggregation_weights(dest_scores_aggregation_file_path)
                    aggregation_df = pd.read_csv(dest_scores_aggregation_file_path)
                    #print(aggregation_df.head())
                    res_aggregator = Aggregator(aggregation_df)
                    pprint(aggregation_weights)
                    aggregation_results = res_aggregator.weighted_avg(aggregation_weights)
                    #print(aggregation_results.head())
                    print("Storing computed weighted avg in {}...".format(dest_scores_aggregation_file_path))
                    aggregation_results.to_csv(dest_scores_aggregation_file_path, index=False)

                    print("Generating Recommendations...")
                    items_for_evaluation = generate_weighted_recommendations(dest_items_for_evaluation_file,
                                                                             aggregation_results, no_of_recs)
                    with open(dest_items_for_evaluation_file, 'w') as json_file:
                        json.dump(items_for_evaluation, fp=json_file, indent=4)

                    precision_recall_intf = PrecisionRecall()
                    evaluation_results = precision_recall_intf.compute_precision_recall(no_of_recs_to_eval,
                                                                                        items_for_evaluation,
                                                                                        all_items)
                    #pprint(evaluation_results)
                    print("Evaluating...")
                    evaluation_file_path = os.path.join(logit_dir,
                                                        'evaluation_results.json')
                    with open(evaluation_file_path, 'w') as json_file:
                        json.dump(evaluation_results, fp=json_file, indent=4)

                    config['recommenders'] = aggregation_weights

                    weights_config_file = os.path.join(logit_dir, 'weights_config.json')
                    with open(weights_config_file, 'w') as json_file:
                        json.dump(config, fp=json_file, indent=4)
                    print("Aggregation weights config are present in  : ", weights_config_file)

def hybrid_kfold_evaluation_using_auto_weights(models_dir, all_items, item_id_col, no_of_recs, no_of_recs_to_eval):
    """parse hybrid recommendation files to aggregate based on weights generated by ML model"""
    #print(len(all_items))
    for root, _, files in os.walk(models_dir, topdown=False):
        for name in files:
            if fnmatch(name, 'all_scores_aggregation.csv'):
                src_scores_aggregation_file_path = (os.path.join(root, name))
                kfold_experiments_dir = Path(src_scores_aggregation_file_path).parent
                model_dir = Path(kfold_experiments_dir).parent
                model_parent_dir = Path(model_dir).parent
                if ('hybrid' in model_dir.name):
                #if ('all_recommenders' in model_dir.name):
                    print("\nComputing Aggregation Weights for Model")
                    model_name = model_dir.name.replace('equal', 'logit')
                    #print(model_name)
                    #print(kfold_experiments_dir)
                    logit_kfold_dir = str(kfold_experiments_dir).replace('equal', 'logit')
                    #print(logit_kfold_dir)
                    #input()
                    if not os.path.exists(logit_kfold_dir):
                        os.makedirs(logit_kfold_dir)
                    src_all_items_for_evaluation_file = os.path.join(kfold_experiments_dir,
                                                                     'all_items_for_evaluation.json')
                    dest_all_items_for_evaluation_file = os.path.join(logit_kfold_dir,
                                                                     'all_items_for_evaluation.json')
                    #print(src_all_items_for_evaluation_file, dest_all_items_for_evaluation_file)
                    copyfile(src_all_items_for_evaluation_file, dest_all_items_for_evaluation_file)
                    #print(dest_all_items_for_evaluation_file)

                    dest_scores_aggregation_file_path = os.path.join(logit_kfold_dir,
                                                                     'all_scores_aggregation.csv')
                    #print(src_scores_aggregation_file_path, dest_scores_aggregation_file_path)
                    copyfile(src_scores_aggregation_file_path, dest_scores_aggregation_file_path)
                    #print(dest_scores_aggregation_file_path)

                    config = dict()
                    config['model_dir_name'] = model_name
                    print("Loading {} ...".format(dest_scores_aggregation_file_path))
                    aggregation_weights = get_aggregation_weights(dest_scores_aggregation_file_path)
                    aggregation_df = pd.read_csv(dest_scores_aggregation_file_path)
                    #print(aggregation_df.head())
                    res_aggregator = Aggregator(aggregation_df)
                    pprint(aggregation_weights)
                    aggregation_results = res_aggregator.weighted_avg(aggregation_weights)
                    #print(aggregation_results.head())
                    print("Storing computed weighted avg in {}...".format(dest_scores_aggregation_file_path))
                    aggregation_results.to_csv(dest_scores_aggregation_file_path, index=False)

                    print("Generating Recommendations...")
                    all_items_for_evaluation = generate_weighted_recommendations(dest_all_items_for_evaluation_file,
                                                                                 aggregation_results, no_of_recs)
                    with open(dest_all_items_for_evaluation_file, 'w') as json_file:
                        json.dump(all_items_for_evaluation, fp=json_file, indent=4)

                    precision_recall_intf = PrecisionRecall()
                    evaluation_results = precision_recall_intf.compute_precision_recall(no_of_recs_to_eval,
                                                                                        all_items_for_evaluation,
                                                                                        all_items)
                    #pprint(evaluation_results)
                    print("Evaluating...")
                    evaluation_file_path = os.path.join(logit_kfold_dir,
                                                        'kfold_evaluation.json')
                    with open(evaluation_file_path, 'w') as json_file:
                        json.dump(evaluation_results, fp=json_file, indent=4)

                    config['recommenders'] = aggregation_weights

                    weights_config_file = os.path.join(logit_kfold_dir, 'weights_config.json')
                    with open(weights_config_file, 'w') as json_file:
                        json.dump(config, fp=json_file, indent=4)
                    print("Aggregation weights config are present in  : ", weights_config_file)
