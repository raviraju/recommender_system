"""Module for Popularity Based Recommender"""
import os
import sys
from timeit import default_timer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.reco_interface import RecommenderIntf

class PopularityBasedRecommender(RecommenderIntf):
    """Popularity based recommender system model"""
    def __init__(self, results_dir):
        """constructor"""
        super().__init__(results_dir)
        self.recommendations = None

    def train(self, train_data, user_id, item_id):
        """train the popularity based recommender system model"""
        start_time = default_timer()
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        all_users = list(self.train_data[self.user_id].unique())
        print("No. of users in the training set: {}".format(len(all_users)))
        all_items = list(self.train_data[self.item_id].unique())
        print("No. of items in the training set: {}".format(len(all_items)))
        #Get a count of user_ids for each unique item as popularity score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns={self.user_id: 'score'}, inplace=True)

        #Sort the items based upon popularity score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending=[0, 1])

        #Generate a recommendation rank based upon score
        train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        #Get the top 10 recommendations
        self.recommendations = train_data_sort.head(10)
        end_time = default_timer()
        print("{:50}    {}".format("Training Completed in : ", utilities.convert_sec(end_time - start_time)))

    def recommend(self, user_id):
        """Generate item recommendations for given user_id"""
        start_time = default_timer()
        user_recommendations = self.recommendations

        #Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id

        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated in : ", utilities.convert_sec(end_time - start_time)))

        recommendations_file = os.path.join(self.results_dir, 'popular_recommendation.csv')
        user_recommendations.to_csv(recommendations_file)

        return user_recommendations
