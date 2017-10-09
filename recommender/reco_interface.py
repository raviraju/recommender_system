"""Module for Recommender Abstract Base Class"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABCMeta, abstractmethod

class RecommenderIntf(metaclass=ABCMeta):
    """Abstract Base Class Interface"""
    def __init__(self):
        """constructor"""
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.current_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.train_data = None
        self.user_id = None
        self.item_id = None

    @abstractmethod
    def train(self, train_data, user_id, item_id):
        """train recommender"""
        raise NotImplementedError()

    @abstractmethod
    def recommend(self, user_id):
        """recommend items for given user_id"""
        raise NotImplementedError()
