"""Module for Popularity Based Recommender"""
import os
import sys
import logging
from timeit import default_timer
from pprint import pprint
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.reco_interface import RecommenderIntf
from recommender.reco_interface import load_train_test
from recommender.evaluation import PrecisionRecall

class PopularityBasedRecommender(RecommenderIntf):
    """Popularity based recommender system model"""

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
        self.users_test = None
        self.items_test = None
        self.user_items_test_dict = dict()
        self.recommendations = None
        self.model_file = os.path.join(self.model_dir, 'popularity_based_model.pkl')

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
        LOGGER.debug("Test Data :: Deriving Stats...")
        self.users_test = [str(user_id) for user_id in self.test_data[self.user_id_col].unique()]
        LOGGER.debug("Test Data :: No. of users : " + str(len(self.users_test)))
        self.items_test = [str(item_id) for item_id in self.test_data[self.item_id_col].unique()]
        LOGGER.debug("Test Data :: No. of items : " + str(len(self.items_test)))

        users_items_test_dict = {
            'users_test' : self.users_test,
            'items_test' : self.items_test
        }
        users_items_test_file = os.path.join(self.model_dir, 'users_items_test.json')
        utilities.dump_json_file(users_items_test_dict, users_items_test_file)

        LOGGER.debug("Test Data :: Getting Distinct Items for each User")
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

        LOGGER.debug("Test Data :: Loading Distinct Items for each User")
        user_items_test_file = os.path.join(self.model_dir, 'user_items_test.json')
        self.user_items_test_dict = utilities.load_json_file(user_items_test_file)        

    def train(self):
        """train the popularity based recommender system model"""
        self.__derive_stats()
        print("Training...")
        start_time = default_timer()
        # Get a count of user_ids for each unique item as popularity score
        train_data_grouped = self.train_data.groupby([self.item_id_col])\
                                            .agg({self.user_id_col: 'count'})\
                                            .reset_index()
        train_data_grouped.rename(columns={self.user_id_col:'no_of_users',
                                           self.item_id_col:self.item_id_col},
                                  inplace=True)

        #Sort the items based upon popularity score : no_of_users
        train_data_sort = train_data_grouped.sort_values(['no_of_users', self.item_id_col],
                                                         ascending=[0, 1])

        #Generate a recommendation rank based upon score : no_of_users
        train_data_sort['rank'] = train_data_sort['no_of_users']\
                                  .rank(ascending=0, method='first')
        train_data_sort.reset_index(drop=True, inplace=True)

        self.recommendations = train_data_sort.head(self.no_of_recs)
        end_time = default_timer()
        print("{:50}    {}".format("Training Completed in : ",
                                   utilities.convert_sec(end_time - start_time)))
        joblib.dump(self.recommendations, self.model_file)
        LOGGER.debug("Saved Model")

    def recommend_items(self):
        """Generate item recommendations for given user_id"""
        if not os.path.exists(self.model_file):
            print("Trained Model not found !!!. Failed to recommend")
            return None
        self.recommendations = joblib.load(self.model_file)
        LOGGER.debug("Loaded Trained Model")

        start_time = default_timer()
        recommended_items = self.recommendations[self.item_id_col].tolist()
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated in : ",
                                   utilities.convert_sec(end_time - start_time)))
        return recommended_items

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

    def __get_items(self, user_id, dataset='train'):
        """private function, Get unique items for a given user"""
        if dataset == "train":
            user_items = self.user_items_train_dict[user_id]
        else:#test
            user_items = self.user_items_test_dict[user_id]
        return user_items

    def __generate_top_recommendations(self):
        """Generate top popularity recommendations"""
        recommended_items = self.recommendations[self.item_id_col].tolist()
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
        """
        for user_id in users:
            # Get all items with which user has interacted
            items_interacted = self.__get_items(user_id, dataset)
            eval_items[user_id] = dict()
            eval_items[user_id]['items_recommended'] = []
            eval_items[user_id]['items_interacted'] = []
            recommended_items = self.__generate_top_recommendations()
            eval_items[user_id]['items_recommended'] = recommended_items
            eval_items[user_id]['items_interacted'] = items_interacted
        print("Evaluation : No of users : ", no_of_users)
        """
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
            recommended_items = self.__generate_top_recommendations()
            eval_items[user_id]['items_recommended'] = recommended_items
            eval_items[user_id]['assume_interacted_items'] = assume_interacted_items
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
            self.recommendations = joblib.load(self.model_file)
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
    model = PopularityBasedRecommender(results_dir, model_dir,
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
    model = PopularityBasedRecommender(results_dir, model_dir,
                                       train_data, test_data,
                                       user_id_col, item_id_col, no_of_recs)
    results = model.evaluate(no_of_recs_to_eval, dataset, hold_out_ratio)
    pprint(results)
    print('*' * 80)

def recommend(results_dir, model_dir, train_test_dir,
              user_id_col, item_id_col,
              user_id, no_of_recs=10):
    """recommend items for user"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    model = PopularityBasedRecommender(results_dir, model_dir,
                                       train_data, test_data,
                                       user_id_col, item_id_col, no_of_recs)
       
    print("Items interactions for a user with user_id : {}".format(user_id))
    interacted_items = list(test_data[test_data[user_id_col] == user_id][item_id_col])
    for item in interacted_items:
        print(item)
            
    print()
    print("Items recommended for a user with user_id : {}".format(user_id))
    recommended_items = model.recommend_items()    
    if recommended_items:
        for item in recommended_items:
            print(item)
    else:
        print("No items to recommend")
    print('*' * 80)

def train_eval_recommend(results_dir, model_dir, train_test_dir,
                         user_id_col, item_id_col,
                         no_of_recs_to_eval, dataset='test',
                         no_of_recs=10, hold_out_ratio=0.5):
    """Train Evaluate and Recommend for Popularity Based Recommender"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    print("Training Recommender...")
    model = PopularityBasedRecommender(results_dir, model_dir,
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
    recommended_items = model.recommend_items()
    print()
    if recommended_items:
        for item in recommended_items:
            print(item)
    else:
        print("No items to recommend")
    print('*' * 80)
