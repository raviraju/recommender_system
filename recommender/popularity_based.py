"""Module for Popularity Based Recommender"""
import os
import sys
import logging
import random
from timeit import default_timer
import joblib

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.reco_interface import RecommenderIntf
from recommender.evaluation import PrecisionRecall

class PopularityBasedRecommender(RecommenderIntf):
    """Popularity based recommender system model"""
    def __derive_stats(self):
        """private function, derive stats"""
        LOGGER.debug("Getting All Users and Items")
        self.all_users = list(self.train_data[self.user_id_col].unique())
        LOGGER.debug("No. of users in the training set: " + str(len(self.all_users)))
        self.all_items = list(self.train_data[self.item_id_col].unique())
        LOGGER.debug("No. of items in the training set: " + str(len(self.all_items)))

    def __init__(self, results_dir, model_dir, train_data, test_data, user_id_col, item_id_col, no_of_recs=10):
        """constructor"""
        super().__init__(results_dir, model_dir, train_data, test_data, user_id_col, item_id_col, no_of_recs)

        self.all_users = None
        self.all_items = None
        self.__derive_stats()

        self.recommendations = None
        self.model_file = os.path.join(self.model_dir, 'popularity_based_model.pkl')

    def train(self):
        """train the popularity based recommender system model"""
        start_time = default_timer()
        # Get a count of user_ids for each unique item as popularity score
        train_data_grouped = self.train_data.groupby([self.item_id_col]).agg({self.user_id_col: 'count'}).reset_index()
        train_data_grouped.rename(columns={self.user_id_col: 'score', self.item_id_col:'item_id'}, inplace=True)

        #Sort the items based upon popularity score
        train_data_sort = train_data_grouped.sort_values(['score', 'item_id'], ascending=[0, 1])

        #Generate a recommendation rank based upon score
        train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        self.recommendations = train_data_sort.head(self.no_of_recs)
        end_time = default_timer()
        print("{:50}    {}".format("Training Completed in : ", utilities.convert_sec(end_time - start_time)))
        #print(self.cooccurence_matrix.shape)
        joblib.dump(self.recommendations, self.model_file)
        LOGGER.debug("Saved Model")

    def recommend(self, user_id):
        """Generate item recommendations for given user_id"""
        if not os.path.exists(self.model_file):
            print("Trained Model not found !!!. Failed to recommend")
            return None

        self.recommendations = joblib.load(self.model_file)
        LOGGER.debug("Loaded Trained Model")
        start_time = default_timer()
        recommended_items = self.recommendations['item_id'].tolist()
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated in : ",
                                   utilities.convert_sec(end_time - start_time)))

        return recommended_items

    def __generate_top_recommendations(self):
        """Use the recommendations dataframe to make top recommendations"""
        return self.recommendations

    def __get_items_for_eval(self, users_test_sample):
        """Generate recommended and interacted items for users in the user test sample"""
        eval_items = dict()

        for user_id in users_test_sample:
            eval_items[user_id] = dict()
            eval_items[user_id]['items_recommended'] = dict()
            eval_items[user_id]['items_interacted'] = dict()

            user_recommendations = self.__generate_top_recommendations()
            recommended_items = user_recommendations['item_id']
            eval_items[user_id]['items_recommended'] = recommended_items

            #Get items for user_id from test_data
            test_data_user = self.test_data[self.test_data[self.user_id_col] == user_id]
            eval_items[user_id]['items_interacted'] = test_data_user[self.item_id_col].unique()
        return eval_items

    def eval(self, sample_test_users_percentage, no_of_recs_to_eval):
        """Evaluate trained model"""
        if os.path.exists(self.model_file):
            self.recommendations = joblib.load(self.model_file)
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
            return results
        else:
            print("Trained Model not found !!!. Failed to evaluate")
            results = {'status' : "Trained Model not found !!!. Failed to evaluate"}
            return results
