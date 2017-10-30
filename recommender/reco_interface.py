"""Module for Recommender Abstract Base Class"""
import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABCMeta, abstractmethod

class RecommenderIntf(metaclass=ABCMeta):
    """Abstract Base Class Interface"""
    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, no_of_recs=10):
        """constructor"""
        self.results_dir = results_dir
        self.model_dir = model_dir

        self.train_data = train_data
        self.test_data = test_data
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col

        self.no_of_recs = no_of_recs

    @abstractmethod
    def train(self):
        """train recommender"""
        raise NotImplementedError()

    def eval(self, sample_test_users_percentage, no_of_recs_to_eval):
        """evaluate recommender"""
        raise NotImplementedError()

    def evaluate(self, no_of_recs_to_eval, dataset='train', hold_out_ratio=0.5):
        """evaluate recommender"""
        raise NotImplementedError()

    def recommend(self, user_id):
        """recommend items for given user_id"""
        raise NotImplementedError()

    def recommend_items(self, user_id, dataset='train'):
        """recommend items for given user_id"""
        raise NotImplementedError()

    def __get_random_sample(self, list_a, percentage):
        """private function, return random percentage of values from a list"""
        k = int(len(list_a) * percentage)
        #random.seed(0)
        indicies = random.sample(range(len(list_a)), k)
        new_list = [list_a[i] for i in indicies]
        return new_list

    def get_random_sample(self, list_a, percentage):
        """return random percentage of values from a list"""
        k = int(len(list_a) * percentage)
        #random.seed(0)
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
        users_test_sample = self.__get_random_sample(users_test_and_training, percentage)
        print("Sample no of common users, used for evaluation : {}".format(len(users_test_sample)))
        return users_test_sample
