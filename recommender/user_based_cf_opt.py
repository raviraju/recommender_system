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
                 user_id_col, item_id_col,
                 no_of_recs=10, hold_out_ratio=0.5):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col,
                         no_of_recs)
        self.hold_out_ratio = hold_out_ratio
        
        self.users_train = None
        self.items_train = None
        self.user_items_train_dict = dict()
        self.item_users_train_dict = dict()

        self.users_test = None
        self.items_test = None
        self.user_items_test_dict = dict()
        self.item_users_test_dict = dict()

        self.user_similarity_matrix_df = None
        self.similar_users = None
        self.uim_df = None
        self.eval_items = None
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

    def __get_known_items(self, items_interacted):
        """return filtered items which are present in training set"""
        known_items_interacted = []
        items_training_set = self.__get_all_items(dataset='train')
        for item in items_interacted:
            if item in items_training_set:
                known_items_interacted.append(item)
        return known_items_interacted

    def __split_items(self, items_interacted):
        """return assume_interacted_items, hold_out_items"""
        #print(items_interacted)
        items_interacted = list(set(items_interacted))
        items_interacted.sort()                
        #print(items_interacted)
        #input()
        
        assume_interacted_items = set()
        hold_out_items = set()               
        no_of_items_interacted = len(items_interacted)
        no_of_items_to_be_held = int(no_of_items_interacted*self.hold_out_ratio)
        hold_out_items = set(items_interacted[-no_of_items_to_be_held:])
        #hold_out_items = set(self.get_random_sample(items_interacted, self.hold_out_ratio))
              
        assume_interacted_items = set(items_interacted) - hold_out_items

        assume_interacted_items = list(assume_interacted_items)
        assume_interacted_items.sort()
        
        hold_out_items = list(hold_out_items)
        hold_out_items.sort()
        
        return assume_interacted_items, hold_out_items
    
    def __save_eval_items(self):
        """save eval items"""
        eval_items_file = os.path.join(self.model_dir, 'eval_items.json')
        utilities.dump_json_file(self.eval_items, eval_items_file)
    
    def __load_eval_items(self):
        """load eval items"""
        eval_items_file = os.path.join(self.model_dir, 'eval_items.json')
        self.eval_items = utilities.load_json_file(eval_items_file)

    def __get_custom_test_data(self, dataset='test'):
        self.eval_items = dict()
        users = self.__get_all_users(dataset='test')
        no_of_users = len(users)
        no_of_users_considered = 0
        custom_test_data = None
        for user_id in users:                                    
            # Get all items with which user has interacted
            items_interacted = self.__get_items(user_id, dataset='test')
            if dataset != 'train':
                items_interacted = self.__get_known_items(items_interacted)
            assume_interacted_items, hold_out_items = self.__split_items(items_interacted)
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
            
            self.eval_items[user_id] = dict()
            self.eval_items[user_id]['items_recommended'] = []
            self.eval_items[user_id]['assume_interacted_items'] = assume_interacted_items
            self.eval_items[user_id]['items_interacted'] = hold_out_items
            no_of_users_considered += 1
            
                    
            tmp = self.test_data[self.test_data[self.user_id_col] == user_id]
            filter_tmp = tmp.loc[tmp[self.item_id_col].isin(assume_interacted_items)]
            #print("filter_tmp")
            #print(filter_tmp)
            if custom_test_data is None:
                custom_test_data = filter_tmp.copy()
            else:
                custom_test_data = custom_test_data.append(filter_tmp, ignore_index=True)
            #print("custom_test_data")
            #print(custom_test_data[[self.user_id_col, self.item_id_col]])
        
        print("No of test users considered for evaluation : ", len(self.eval_items))
        self.__save_eval_items()        
        return custom_test_data
    
    def __save_uim(self):
        """save user item interaction matrix"""        
        uim_df_fname = os.path.join(self.model_dir, 'uim.csv')
        uim_df = self.uim_df.reset_index()#so that user_id col is added as first col
        uim_df.to_csv(uim_df_fname, index=False)#do not write the default index, 
                                                #so that on read first col is picked as index col

    def __load_uim(self):
        """load user item interaction matrix"""
        uim_df_fname = os.path.join(self.model_dir, 'uim.csv')
        self.uim_df = pd.read_csv(uim_df_fname, index_col=[self.user_id_col])
        self.uim_df.index = self.uim_df.index.map(str)

    def __compute_user_similarity_matrix(self, custom_test_data):
        """private function, Construct cooccurence matrix"""
        #Construct User Item Matrix
        
        start_time = default_timer()
        print()
        print("Combining users from train and test...")
        users_data = self.train_data.append(custom_test_data, ignore_index=True)
        print("Constructing User Item Matrix...")
        self.uim_df = pd.get_dummies(users_data[self.item_id_col])\
                        .groupby(users_data[self.user_id_col])\
                        .apply(max)
        self.__save_uim()
        
        uim = self.uim_df.as_matrix()
        
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))
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
        items = [str(col) for col in self.uim_df.columns]
        no_of_items = len(items)
        users = [str(idx) for idx in self.uim_df.index]
        no_of_users = len(users)
        print("No of Items : ", no_of_items)
        print("No of Users : ", no_of_users)
        non_zero_count = np.count_nonzero(uim)
        count = uim.size
        density = non_zero_count/count
        print("Density of User Item Matrix : ", density)

        start_time = default_timer()
        print()
        print("Computing User-User Similarity Matrix with intersection of items interacted...")
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
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))
        
        start_time = default_timer()
        print()
        print("Computing User-User Similarity Matrix with union of items interacted...")
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
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))
        
        start_time = default_timer()
        print()
        print("Computing User-User Similarity Matrix with Jaccard Similarity of items interacted...")
        #Compute User-User Similarity Matrix with Jaccard Similarity of items
        jaccard = intersection/union
        jaccard_df = pd.DataFrame(jaccard,
                                  columns=users,
                                  index=users)
        jaccard_df_fname = os.path.join(self.model_dir,
                                        'jaccard.csv')
        jaccard_df.to_csv(jaccard_df_fname, index=False)
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))        
        return jaccard_df

    def train(self):
        """Train the user similarity based recommender system model"""
        self.__derive_stats()
        custom_test_data = self.__get_custom_test_data(dataset='test')
        
        # Construct user similarity matrix of size, len(users) X len(users)
        print("Construct user similarity matrix...")
        start_time = default_timer()
        self.user_similarity_matrix_df = self.__compute_user_similarity_matrix(custom_test_data)
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))
        #print(self.user_similarity_matrix_df.shape)
        joblib.dump(self.user_similarity_matrix_df, self.model_file)
        LOGGER.debug("Saved Model")        
           
    def __get_similar_users(self, user_id):
        #print(user_id)
        similar_users = self.user_similarity_matrix_df[user_id]
        sorted_similar_users = similar_users.sort_values(ascending=False)
        #print(len(sorted_similar_users))
        most_similar_users = (sorted_similar_users.drop(user_id))
        most_similar_users = most_similar_users[most_similar_users > 0]#score > 0
        #print(len(most_similar_users))
        #print(most_similar_users)
        return most_similar_users
        
    def __generate_top_recommendations(self, user_id, user_interacted_items):
        """Use the cooccurence matrix to make top recommendations"""
        # Calculate a weighted average of the scores in cooccurence matrix for
        # all user items.              
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']

        similar_users_weights = self.__get_similar_users(user_id)
        similar_user_ids = similar_users_weights.index
                
        sub_uim_df = self.uim_df.loc[similar_user_ids]
        weighted_sub_uim_df = sub_uim_df.mul(similar_users_weights, axis='index')
        no_of_similar_users = sub_uim_df.shape[0]
        if no_of_similar_users != 0:
            item_scores = sub_uim_df.sum(axis=0) / float(no_of_similar_users)
            item_scores.sort_values(inplace=True, ascending=False)
            #print(item_scores)
            #item_scores = item_scores[item_scores > 0]

            rank = 1
            for item_id, score in item_scores.items():
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

    def recommend_items(self, user_id):
        """Generate item recommendations for given user_id from chosen dataset"""
        self.__load_stats()
        self.__load_uim()
        self.__load_eval_items()
        #pprint(self.eval_items[user_id])
        
        if os.path.exists(self.model_file):
            self.user_similarity_matrix_df = joblib.load(self.model_file)
            #print(self.user_similarity_matrix_df.shape)
            LOGGER.debug("Loaded Trained Model")
            # Use the cooccurence matrix to make recommendations
            start_time = default_timer()
            user_recommendations = self.__generate_top_recommendations(user_id,
                                                                       self.eval_items[user_id]['assume_interacted_items'])
            recommended_items = list(user_recommendations[self.item_id_col].values)
            end_time = default_timer()
            print("{:50}    {}".format("Recommendations generated in : ",
                                       utilities.convert_sec(end_time - start_time)))
            return recommended_items
        else:
            print("Trained Model not found !!!. Failed to generate recommendations")
            return None

    def __get_reco_for_eval_items(self):
        """Generate recommended and interacted items for users"""       
        for user_id in self.eval_items:
            user_recommendations = self.__generate_top_recommendations(user_id,
                                                                       self.eval_items[user_id]['assume_interacted_items'])
            
            recommended_items = list(user_recommendations[self.item_id_col].values)
            self.eval_items[user_id]['items_recommended'] = recommended_items

        return self.eval_items

    def evaluate(self, no_of_recs_to_eval):
        """Evaluate trained model"""
        self.__load_stats()
        self.__load_uim()
        self.__load_eval_items()
                
        if os.path.exists(self.model_file):
            self.user_similarity_matrix_df = joblib.load(self.model_file)
            #print(self.user_similarity_matrix_df.shape)
            LOGGER.debug("Loaded Trained Model")
            
            start_time = default_timer()
            #Generate recommendations for the users
            self.eval_items = self.__get_reco_for_eval_items()
            self.__save_eval_items()

            precision_recall_intf = PrecisionRecall()
            results = precision_recall_intf.compute_precision_recall(
                no_of_recs_to_eval, self.eval_items)
            end_time = default_timer()
            print("{:50}    {}".format("Evaluation Completed in : ",
                                       utilities.convert_sec(end_time - start_time)))
            
            results_file = os.path.join(self.model_dir, 'results.json')
            utilities.dump_json_file(results, results_file)
            
            return results
        else:
            print("Trained Model not found !!!. Failed to evaluate")
            results = {'status' : "Trained Model not found !!!. Failed to evaluate"}            
            
            results_file = os.path.join(self.model_dir, 'results.json')
            utilities.dump_json_file(results, results_file)
            
            return results

def train(results_dir, model_dir, train_test_dir,
          user_id_col, item_id_col,
          no_of_recs=10, hold_out_ratio=0.5):
    """train recommender"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    print("Training Recommender...")
    model = UserBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col,
                                   no_of_recs, hold_out_ratio)
    model.train()
    print('*' * 80)

def evaluate(results_dir, model_dir, train_test_dir,
             user_id_col, item_id_col,
             no_of_recs_to_eval,
             no_of_recs=10, hold_out_ratio=0.5):
    """evaluate recommender"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    print("Evaluating Recommender System...")
    model = UserBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col,
                                   no_of_recs, hold_out_ratio)
    results = model.evaluate(no_of_recs_to_eval)
    pprint(results)
    print('*' * 80)

def recommend(results_dir, model_dir, train_test_dir,
              user_id_col, item_id_col,
              user_id,
              no_of_recs=10, hold_out_ratio=0.5):
    """recommend items for user"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    model = UserBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col,
                                   no_of_recs, hold_out_ratio)

    recommended_items = model.recommend_items(user_id)
    print("Items recommended for a user with user_id : {}".format(user_id))    
    if recommended_items:
        for item in recommended_items:
            print(item)            
    else:
        print("No items to recommend")
    print('*' * 80)

        
def train_eval_recommend(results_dir, model_dir, train_test_dir,
                         user_id_col, item_id_col,
                         no_of_recs_to_eval,
                         no_of_recs=10, hold_out_ratio=0.5):
    """Train Evaluate and Recommend for Item Based Recommender"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    print("Training Recommender...")
    model = UserBasedCFRecommender(results_dir, model_dir,
                                   train_data, test_data,
                                   user_id_col, item_id_col,
                                   no_of_recs, hold_out_ratio)
    model.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    results = model.evaluate(no_of_recs_to_eval)
    pprint(results)
    print('*' * 80)

    print("Testing Recommendation for an User")
    #users = test_data[user_id_col].unique()
    eval_items_file = os.path.join(model_dir, 'eval_items.json')
    eval_items = utilities.load_json_file(eval_items_file)
    users = list(eval_items.keys())
    user_id = users[0]    
    recommended_items = model.recommend_items(user_id)
    print("Items recommended for a user with user_id : {}".format(user_id))
    if recommended_items:
        for item in recommended_items:
            print(item)            
    else:
        print("No items to recommend")
    print('*' * 80)
