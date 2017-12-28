"""Module for Content Based Books Recommender"""
import os
import sys
import argparse
import logging
from timeit import default_timer
from pprint import pprint
import joblib

from collections import defaultdict
from math import log10
from sklearn.preprocessing import minmax_scale

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.rec_interface import load_train_test
from recommender.rec_interface import Recommender
import rec_interface as books_rec_interface
from recommender.evaluation import PrecisionRecall

class ContentBasedRecommender(books_rec_interface.BooksRecommender):
    """Content based recommender system model"""

    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)
        self.model_file = os.path.join(self.model_dir,
                                       'content_based_model.pkl')
        
        self.item_profile_train_dict = dict()
        self.item_profile_test_dict = dict()
        self.recommendations = None
    #######################################
    def derive_stats(self):
        """derive stats"""
        super().derive_stats()
          
        LOGGER.debug("Train Data       :: Getting Item Profiles")
        for item_id in self.items_train:
            item_profile = books_rec_interface.get_item_profile(self.train_data, item_id)
            name_tokens_set, author_set, keywords = item_profile
            profile = {'name_tokens' : list(name_tokens_set),
                       'author' : list(author_set), 
                       'keywords' : keywords}
            self.item_profile_train_dict[item_id] = profile

        item_profile_train_file = os.path.join(self.model_dir, 'item_profile_train.json')
        utilities.dump_json_file(self.item_profile_train_dict, item_profile_train_file)                       
        ########################################################################       
        LOGGER.debug("Test Data       :: Getting Item Profiles")
        for item_id in self.items_test:
            item_profile = books_rec_interface.get_item_profile(self.test_data, item_id)
            name_tokens_set, author_set, keywords = item_profile
            profile = {'name_tokens' : list(name_tokens_set),
                       'author' : list(author_set), 
                       'keywords' : keywords}           
            self.item_profile_test_dict[item_id] = profile

        item_profile_test_file = os.path.join(self.model_dir, 'item_profile_test.json')
        utilities.dump_json_file(self.item_profile_test_dict, item_profile_test_file)
        
    def load_stats(self):
        """load stats"""
        super().load_stats()

        LOGGER.debug("Train Data       :: Loading Item Profiles")
        item_profile_train_file = os.path.join(self.model_dir, 'item_profile_train.json')
        self.item_profile_train_dict = utilities.load_json_file(item_profile_train_file)
        ############################################################################      
        LOGGER.debug("Test Data :: Loading Item Profiles")
        item_profile_test_file = os.path.join(self.model_dir, 'item_profile_test.json')
        self.item_profile_test_dict = utilities.load_json_file(item_profile_test_file)
    #######################################
    def train(self):
        """train the content based recommender system model"""
        self.derive_stats()
        self.get_test_data_for_evaluation()
    #######################################
    def __get_item_profile(self, item_id, dataset='train'):
        """private function, return item profile saved for given item_id"""
        if dataset == "train":
            if item_id in self.item_profile_train_dict:
                return self.item_profile_train_dict[item_id]
            else:
                return {}
        else:#test
            if item_id in self.item_profile_test_dict:
                return self.item_profile_test_dict[item_id]
            else:
                return {}
        
    def __get_user_profile(self, user_items, dataset='train'):
        """private function, return user profile by merging item profiles for user interacted items"""
        user_profile_name_tokens_set = set()
        user_profile_authors_set = set()
        user_profile_keywords = []

        for item_id in user_items:
            item_profile = self.__get_item_profile(item_id, dataset)
            '''
            print(item_id)
            print(item_profile)
            print('%'*5)
            '''            
            name_tokens_set = set(item_profile['name_tokens'])
            user_profile_name_tokens_set |= name_tokens_set
            
            author_set = set(item_profile['author'])
            user_profile_authors_set |= author_set
                  
            keywords = item_profile['keywords']
            user_profile_keywords.extend(keywords)

        profile = {'name_tokens' : list(user_profile_name_tokens_set),
                   'author' : list(user_profile_authors_set),
                   'keywords' : user_profile_keywords
                  }       
        return profile
                   
    def __get_similarity_score(self, user_profile, item_profile):
        """similarity scores bw user and item profile"""
        name_tokens_similarity = books_rec_interface.get_jaccard_similarity(set(user_profile['name_tokens']),
                                                                            set(item_profile['name_tokens']))
        authors_similarity = books_rec_interface.get_jaccard_similarity(set(user_profile['author']),
                                                                        set(item_profile['author']))
        keywords_similarity = books_rec_interface.get_term_freq_similarity(user_profile['keywords'],
                                                                           item_profile['keywords'])
        '''
        print("\tname : {}, author : {}, keywords : {}, score : {} ".format(name_tokens_similarity,
                                                                          authors_similarity,
                                                                          keywords_similarity,
                                                                         score))  
        '''
        return (name_tokens_similarity, authors_similarity, keywords_similarity)
 
    def __weighted_avg(self, data_frame, columns_weights_dict):
        """compute weighted average defined by columns_weights_dict"""
        data_frame['sim_score'] = 0.0
        total = 0.0
        for col_name in columns_weights_dict:
            weighted_col = data_frame[col_name]*columns_weights_dict[col_name]
            data_frame['sim_score'] = data_frame['sim_score'] + weighted_col
            #print(data_frame['sim_score'])            
        return data_frame    
                
    def __generate_top_recommendations(self, user_id, user_interacted_items):        
        """Get all items from train data and recommend them which are most similar to user_profile"""
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']
        
        user_profile = self.__get_user_profile(user_interacted_items, dataset='test')        
        #print("User Profile")
        #print(user_profile)
        #print()
        
        items_train = self.get_all_items(dataset='train')
        item_scores = []
        for item_id in items_train:            
            item_profile = self.__get_item_profile(item_id, dataset='train')                        
            #print("\n\t" + item_id)
            #print(item_profile)
            similarity_scores = self.__get_similarity_score(user_profile, item_profile)
            (name_tokens_similarity, authors_similarity, keywords_similarity) = similarity_scores
            item_scores.append({self.item_id_col : item_id,
                                'name_tokens_similarity': name_tokens_similarity,
                                'authors_similarity': authors_similarity,
                                'keywords_similarity': keywords_similarity})
        item_scores_df = pd.DataFrame(item_scores)

        #print("scaling...")        
        values = item_scores_df[['name_tokens_similarity']].as_matrix()
        scaled_values = minmax_scale(values)
        item_scores_df['name_tokens_similarity_scaled'] = scaled_values
        
        values = item_scores_df[['authors_similarity']].as_matrix()
        scaled_values = minmax_scale(values)
        item_scores_df['authors_similarity_scaled'] = scaled_values
        
        values = item_scores_df[['keywords_similarity']].as_matrix()
        scaled_values = minmax_scale(values)
        item_scores_df['keywords_similarity_scaled'] = scaled_values

        columns_weights_dict = {'name_tokens_similarity_scaled' : 0.25,
                                'authors_similarity_scaled' : 0.25,
                                'keywords_similarity_scaled' : 0.5
                               }        
        #print("weighted_avg...")
        item_scores_df = self.__weighted_avg(item_scores_df, columns_weights_dict)
        
        #print("sorting...")
        #Sort the items based upon similarity scores
        item_scores_df = item_scores_df.sort_values(['sim_score', self.item_id_col],
                                                         ascending=[0, 1])
        #Generate a recommendation rank based upon score
        item_scores_df['rank'] = item_scores_df['sim_score'].rank(ascending=0, method='first')
        item_scores_df.reset_index(drop=True, inplace=True)
        #print(item_scores_df[item_scores_df['sim_score'] > 0])

        #print(item_scores_df.head())
        for _, item_score in item_scores_df.iterrows():
            if item_id in user_interacted_items:#to avoid items which user has already aware
                continue
            rank = item_score['rank']
            if rank > self.no_of_recs:#limit no of recommendations
                break
            item_dict = {
                self.user_id_col : user_id,
                self.item_id_col : item_score[self.item_id_col],
                'score' : item_score['sim_score'],
                'rank' : rank
            }
            items_to_recommend.append(item_dict)
        res_df = pd.DataFrame(items_to_recommend, columns=columns)
        # Handle the case where there are no recommendations
        # if res_df.shape[0] == 0:
        #     return None
        # else:
        #     return res_df
        return res_df
       
    def recommend_items(self, user_id):
        """recommend items for given user_id from test dataset"""
        self.load_stats()
        self.load_items_for_evaluation()
        
        start_time = default_timer()
        assume_interacted_items = self.items_for_evaluation[user_id]['assume_interacted_items']               
        user_recommendations = self.__generate_top_recommendations(user_id, assume_interacted_items)
        
        recommended_items = list(user_recommendations[self.item_id_col].values)
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated. ",
                                   utilities.convert_sec(end_time - start_time)))
        return recommended_items
    #######################################
    def __recommend_items_to_evaluate(self):
        """recommend items for all users from test dataset"""
        for user_id in self.items_for_evaluation:
            assume_interacted_items = self.items_for_evaluation[user_id]['assume_interacted_items']
            user_recommendations = self.__generate_top_recommendations(user_id,
                                                                       assume_interacted_items)

            recommended_items = list(user_recommendations[self.item_id_col].values)
            self.items_for_evaluation[user_id]['items_recommended'] = recommended_items
        return self.items_for_evaluation

    def evaluate(self, no_of_recs_to_eval):
        """evaluate trained model for different no of ranked recommendations"""        
        self.load_stats()
        self.load_items_for_evaluation()
        
        start_time = default_timer()
        #Generate recommendations for the users
        self.items_for_evaluation = self.__recommend_items_to_evaluate()
        self.save_items_for_evaluation()

        precision_recall_intf = PrecisionRecall()
        evaluation_results = precision_recall_intf.compute_precision_recall(
            no_of_recs_to_eval, self.items_for_evaluation)
        end_time = default_timer()
        print("{:50}    {}".format("Evaluation Completed. ",
                                   utilities.convert_sec(end_time - start_time)))

        results_file = os.path.join(self.model_dir, 'evaluation_results.json')
        utilities.dump_json_file(evaluation_results, results_file)

        return evaluation_results
    #######################################

def train(results_dir, model_dir, train_test_dir,
          user_id_col, item_id_col, **kwargs):
    """train recommender"""
    train_data, test_data = load_train_test(train_test_dir,
                                            user_id_col,
                                            item_id_col)

    print("Training Recommender...")
    model = ContentBasedRecommender(results_dir, model_dir,
                                    train_data, test_data,
                                    user_id_col, item_id_col, **kwargs)
    model.train()
    print('*' * 80)

def evaluate(results_dir, model_dir, train_test_dir,
             user_id_col, item_id_col, **kwargs):
    """evaluate recommender"""
    train_data, test_data = load_train_test(train_test_dir,
                                            user_id_col,
                                            item_id_col)

    print("Evaluating Recommender System")
    model = ContentBasedRecommender(results_dir, model_dir,
                                    train_data, test_data,
                                    user_id_col, item_id_col, **kwargs)
    evaluation_results = model.evaluate(kwargs['no_of_recs_to_eval'])
    pprint(evaluation_results)
    print('*' * 80)

def train_eval_recommend(results_dir, model_dir, train_test_dir,
                         user_id_col, item_id_col, **kwargs):
    """Train Evaluate and Recommend for Popularity Based Recommender"""
    train_data, test_data = load_train_test(train_test_dir,
                                            user_id_col,
                                            item_id_col)

    print("Training Recommender...")
    model = ContentBasedRecommender(results_dir, model_dir,
                                    train_data, test_data,
                                    user_id_col, item_id_col, **kwargs)
    model.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    results = model.evaluate(kwargs['no_of_recs_to_eval'])
    pprint(results)
    print('*' * 80)
    
    print("Testing Recommendation for an User")
    eval_items_file = os.path.join(model_dir, 'items_for_evaluation.json')
    eval_items = utilities.load_json_file(eval_items_file)
    users = list(eval_items.keys())
    user_id = users[0]    
    recommended_items = model.recommend_items(user_id)
    print("Items recommended for a user with user_id : {}".format(user_id))
    print()
    if recommended_items:
        for item in recommended_items:
            print(item)
    else:
        print("No items to recommend")
    print('*' * 80)

def recommend(results_dir, model_dir, train_test_dir,
              user_id_col, item_id_col, user_id, **kwargs):
    """recommend items for user"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    model = ContentBasedRecommender(results_dir, model_dir,
                                    train_data, test_data,
                                    user_id_col, item_id_col, **kwargs)

    metadata_fields = kwargs['metadata_fields']

    eval_items_file = os.path.join(model_dir, 'items_for_evaluation.json')
    eval_items = utilities.load_json_file(eval_items_file)
    if user_id in eval_items:
        assume_interacted_items = eval_items[user_id]['assume_interacted_items']
        items_interacted = eval_items[user_id]['items_interacted']

        print("Assumed Item interactions for a user with user_id : {}".format(user_id))
        for item in assume_interacted_items:
            print(item)
            if metadata_fields is not None:
                item_profile = books_rec_interface.get_item_profile(test_data, item)
                item_name_tokens, item_author, item_keywords = item_profile
                print("\t item_name_tokens : {}".format(item_name_tokens))
                print("\t item_author : {}".format(item_author))
                print("\t item_keywords : {}".format(item_keywords))
                print()

                record = test_data[test_data[item_id_col] == item]
                if not record.empty:
                    for field in metadata_fields:
                        print("\t {} : {}".format(field, record[field].values[0]))
                print('\t '+ '#'*30)

        print()
        print("Items to be interacted for a user with user_id : {}".format(user_id))
        for item in items_interacted:
            print(item)
            if metadata_fields is not None:
                item_profile = books_rec_interface.get_item_profile(test_data, item)
                item_name_tokens, item_author, item_keywords = item_profile
                print("\t item_name_tokens : {}".format(item_name_tokens))
                print("\t item_author : {}".format(item_author))
                print("\t item_keywords : {}".format(item_keywords))
                print()

                record = test_data[test_data[item_id_col] == item]
                if not record.empty:
                    for field in metadata_fields:
                        print("\t {} : {}".format(field, record[field].values[0]))
                print('\t '+ '#'*30)

        print()
        print("Items recommended for a user with user_id : {}".format(user_id))
        recommended_items = model.recommend_items(user_id)
        print()
        if recommended_items:
            for recommended_item in recommended_items:
                print(recommended_item)
                if metadata_fields is not None:
                    item_profile = books_rec_interface.get_item_profile(train_data,
                                                                        recommended_item)
                    item_name_tokens, item_author, item_keywords = item_profile
                    print("\t item_name_tokens : {}".format(item_name_tokens))
                    print("\t item_author : {}".format(item_author))
                    print("\t item_keywords : {}".format(item_keywords))
                    print()

                    record = train_data[train_data[item_id_col] == recommended_item]
                    if not record.empty:
                        for field in metadata_fields:
                            print("\t {} : {}".format(field, record[field].values[0]))
                    for interacted_item in items_interacted:
                        score = books_rec_interface.get_similarity_score(train_data,
                                                                         test_data,
                                                                         recommended_item,
                                                                         interacted_item)
                        print("\t {:20s} | {:20s} | {}".format('recommended_item',
                                                               'interacted_item',
                                                               'score'))
                        print("\t {:20s} | {:20s} | {}".format(recommended_item,
                                                               interacted_item,
                                                               score))
                        print()
                    print('\t '+ '#'*30)
        else:
            print("No items to recommend")
        print('*' * 80)
    else:
        print("""Cannot generate recommendations as either
              items assumed to be interacted or items held out are None""")

def main():
    """Content based recommender interface"""
    parser = argparse.ArgumentParser(description="Content Based Recommender")
    parser.add_argument("--train",
                        help="Train Model",
                        action="store_true")
    parser.add_argument("--eval",
                        help="Evaluate Trained Model",
                        action="store_true")
    parser.add_argument("--recommend",
                        help="Recommend Items for a User",
                        action="store_true")
    parser.add_argument("--user_id",
                        help="User Id to recommend items")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_dir = os.path.join(current_dir, 'model/content_based')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    user_id_col = 'learner_id'
    item_id_col = 'book_code'
    train_test_dir = os.path.join(current_dir, 'train_test_data')

    no_of_recs = 10
    hold_out_ratio = 0.5
    kwargs = {'no_of_recs':no_of_recs,
              'hold_out_ratio':hold_out_ratio              
             }
    #metadata_fields = None
    metadata_fields = ['T_BOOK_NAME', 'T_KEYWORD', 'T_AUTHOR']
    
    if args.train:
        train(results_dir, model_dir, train_test_dir,
              user_id_col, item_id_col, **kwargs)
    elif args.eval:
        kwargs['no_of_recs_to_eval'] = [1, 2, 5, 10]
        evaluate(results_dir, model_dir, train_test_dir,
                 user_id_col, item_id_col, **kwargs)
    elif args.recommend and args.user_id:
        kwargs['metadata_fields'] = metadata_fields
        recommend(results_dir, model_dir, train_test_dir,
                  user_id_col, item_id_col, args.user_id, **kwargs)
    else:
        kwargs['no_of_recs_to_eval'] = [1, 2, 5, 10]
        train_eval_recommend(results_dir, model_dir, train_test_dir,
                             user_id_col, item_id_col, **kwargs)

if __name__ == '__main__':
    main()
