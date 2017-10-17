"""Module for Recommender Abstract Base Class"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABCMeta, abstractmethod

class RecommenderIntf(metaclass=ABCMeta):
    """Abstract Base Class Interface"""
    def __init__(self, results_dir):
        """constructor"""
        self.results_dir = results_dir

        self.train_data = None
        self.user_id_col = None
        self.item_id_col = None

    @abstractmethod
    def train(self, train_data, user_id_col, item_id_col):
        """train recommender"""
        raise NotImplementedError()

    @abstractmethod
    def recommend(self, user_id, no_of_recommendations=10):
        """recommend items for given user_id"""
        raise NotImplementedError()
