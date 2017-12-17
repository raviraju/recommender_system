"""Module for Content Based Books Recommender"""
import os
import sys
import logging
from timeit import default_timer
from pprint import pprint
import joblib

import argparse
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.reco_interface import load_train_test
from recommender.reco_interface import RecommenderIntf
from recommender.evaluation import PrecisionRecall

import re
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
try:
    ENGLISH_STOPWORDS = set(stopwords.words("english"))
except LookupError as err:
    print("Download NLTK Stop words corpus")
    nltk.download('stopwords')
    ENGLISH_STOPWORDS = set(stopwords.words("english"))
#print(ENGLISH_STOPWORDS)

class ContentBasedRecommender(RecommenderIntf):
    """Content based recommender system model"""

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
        self.item_profile_train_dict = dict()
        self.item_profile_test_dict = dict()
        self.train_user_age_dict = dict()
        self.test_user_age_dict = dict()
        self.recommendations = None
        self.model_file = os.path.join(self.model_dir, 'content_based_model.pkl')

    def __get_items(self, user_id, dataset='train'):
        """private function, Get unique items for a given user"""
        if dataset == "train":
            user_items = self.user_items_train_dict[user_id]
        else:#test
            user_items = self.user_items_test_dict[user_id]
        return user_items
    
    def __get_user_age(self, user_id, dataset='train'):
        """private function, Get age for a given user"""
        if dataset == "train":
            age = self.train_user_age_dict[user_id]
        else:#test
            age = self.test_user_age_dict[user_id]
        return age
        
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
        
    def __preprocess_token(self, token):
        preprocessed_token = re.sub('[^a-zA-Z]', '', token)
        preprocessed_token = preprocessed_token.lower()
        #print(token, preprocessed_token)
        if preprocessed_token not in ENGLISH_STOPWORDS:
            return preprocessed_token
        else:
            return None
    
    def get_book_name(self, item_id, dataset='train'):
        if dataset == "train":
            df = self.train_data
        else:#test
            df = self.test_data
            
        records = df[df[self.item_id_col] == item_id]
        for _, record in records.iterrows():
            first_record = record
            break
        #print(first_record)
        book_name = first_record['T_BOOK_NAME']
        #print(book_name)
        book_name_tokens = set()
        if isinstance(book_name, str):        
            tokens = book_name.split(' ')
            for token in tokens:
                preprocessed_token = self.__preprocess_token(token)
                if preprocessed_token:
                    book_name_tokens.add(preprocessed_token)
        #print(book_name, book_name_tokens)
        return book_name, book_name_tokens

    def get_author(self, item_id, dataset='train'):
        if dataset == "train":
            df = self.train_data
        else:#test
            df = self.test_data
        records = df[df[self.item_id_col] == item_id]
        for _, record in records.iterrows():
            first_record = record
            break
        #print(first_record)
        author = first_record['T_AUTHOR']
        return author

    def get_book_keywords(self, item_id, dataset='train'):
        if dataset == "train":
            df = self.train_data
        else:#test
            df = self.test_data
        records = df[df[self.item_id_col] == item_id]
        for _, record in records.iterrows():
            first_record = record
            break
        #print(first_record)
        
        book_keywords = set()
        keywords = first_record['T_KEYWORD']
        if isinstance(keywords, str):
            keywords = keywords.split('|')
            #print(keywords)
            for keyword in keywords:
                tokens = keyword.split(' ')
                #print(tokens)
                for token in tokens:
                    preprocessed_token = self.__preprocess_token(token)
                    if preprocessed_token:
                        book_keywords.add(preprocessed_token)
        #print(book_keywords)
        #print('*'*30)
        return book_keywords

    def get_target_age_range(self, item_id, dataset='train'):
        if dataset == "train":
            df = self.train_data
        else:#test
            df = self.test_data
        records = df[df[self.item_id_col] == item_id]
        for _, record in records.iterrows():
            first_record = record
            break
        #print(first_record)
        begin_target_age = first_record['BEGIN_TARGET_AGE']
        end_target_age = first_record['END_TARGET_AGE']
        return begin_target_age, end_target_age

    def get_item_profile(self, item_id, dataset='train'):        
        name, name_tokens = self.get_book_name(item_id, dataset)
        
        author = self.get_author(item_id, dataset)
        author_info = set()
        if isinstance(author, str):
            author_info.add(author)
            
        keywords = self.get_book_keywords(item_id, dataset)
        
        begin_target_age, end_target_age = self.get_target_age_range(item_id, dataset)
        
        profile = {'name_tokens' : name_tokens, 'author' : author_info, 'keywords' : keywords,
                   'begin_target_age' : begin_target_age, 'end_target_age' : end_target_age
                  }
        return profile

    def __derive_stats(self):
        """private function, derive stats"""
        LOGGER.debug("Train Data :: Deriving Stats...")
        self.users_train = [user_id for user_id in self.train_data[self.user_id_col].unique()]
        LOGGER.debug("Train Data :: No. of users : " + str(len(self.users_train)))
        self.items_train = [item_id for item_id in self.train_data[self.item_id_col].unique()]
        LOGGER.debug("Train Data :: No. of items : " + str(len(self.items_train)))

        users_items_train_dict = {
            'users_train' : self.users_train,
            'items_train' : self.items_train
        }
        #pprint(users_items_train_dict)
        users_items_train_file = os.path.join(self.model_dir, 'users_items_train.json')
        utilities.dump_json_file(users_items_train_dict, users_items_train_file)
                 
        for user_id in self.users_train:
            records = self.train_data[self.train_data[self.user_id_col] == user_id]
            for _, record in records.iterrows():
                first_record = record
                break
            age = first_record['age']
            #print(user_id, age)
            
            self.train_user_age_dict[user_id] = age
        train_user_age_file = os.path.join(self.model_dir, 'train_user_age.json')
        utilities.dump_json_file(self.train_user_age_dict, train_user_age_file)
    
        LOGGER.debug("Train Data :: Getting Item Profiles")
        json_item_profile_train_dict = dict()#json serializable dict
        for item_id in self.items_train:
            profile = self.get_item_profile(item_id, dataset='train')
            self.item_profile_train_dict[item_id] = profile
            
            json_profile = dict()
            for key, val in profile.items():
                if isinstance(val, set):
                    json_profile[key] = list(val)
                else:
                    json_profile[key] = val
            json_item_profile_train_dict[item_id] = json_profile
        item_profile_train_file = os.path.join(self.model_dir, 'item_profile_train.json')
        utilities.dump_json_file(json_item_profile_train_dict, item_profile_train_file)
                        
        LOGGER.debug("Train Data :: Getting Distinct Items for each User")
        user_items_train_df = self.train_data.groupby([self.user_id_col])\
                                             .agg({
                                                 self.item_id_col: (lambda x: list(x.unique()))
                                                 })
        user_items_train_df = user_items_train_df.rename(columns={self.item_id_col: 'items'})\
                                                 .reset_index()
        for _, user_items in user_items_train_df.iterrows():
            user = user_items[self.user_id_col]
            items = [item for item in user_items['items']]
            self.user_items_train_dict[user] = items
        user_items_train_file = os.path.join(self.model_dir, 'user_items_train.json')
        utilities.dump_json_file(self.user_items_train_dict, user_items_train_file)
        ########################################################################
        LOGGER.debug("Test Data :: Deriving Stats...")
        self.users_test = [user_id for user_id in self.test_data[self.user_id_col].unique()]
        LOGGER.debug("Test Data :: No. of users : " + str(len(self.users_test)))
        self.items_test = [item_id for item_id in self.test_data[self.item_id_col].unique()]
        LOGGER.debug("Test Data :: No. of items : " + str(len(self.items_test)))

        users_items_test_dict = {
            'users_test' : self.users_test,
            'items_test' : self.items_test
        }
        users_items_test_file = os.path.join(self.model_dir, 'users_items_test.json')
        utilities.dump_json_file(users_items_test_dict, users_items_test_file)

        for user_id in self.users_test:
            records = self.test_data[self.test_data[self.user_id_col] == user_id]
            for _, record in records.iterrows():
                first_record = record
                break
            age = first_record['age']
            #print(user_id, age)
            self.test_user_age_dict[user_id] = age
        test_user_age_file = os.path.join(self.model_dir, 'test_user_age.json')
        utilities.dump_json_file(self.test_user_age_dict, test_user_age_file)
        
        LOGGER.debug("Test Data :: Getting Item Profiles")
        json_item_profile_test_dict = dict()#json serializable dict
        for item_id in self.items_test:
            profile = self.get_item_profile(item_id, dataset='test')
            self.item_profile_test_dict[item_id] = profile
            
            json_profile = dict()
            for key, val in profile.items():
                if isinstance(val, set):
                    json_profile[key] = list(val)
                else:
                    json_profile[key] = val
            json_item_profile_test_dict[item_id] = json_profile
        item_profile_test_file = os.path.join(self.model_dir, 'item_profile_test.json')
        utilities.dump_json_file(json_item_profile_test_dict, item_profile_test_file)
        
        
        LOGGER.debug("Test Data :: Getting Distinct Items for each User")
        user_items_test_df = self.test_data.groupby([self.user_id_col])\
                                             .agg({
                                                 self.item_id_col: (lambda x: list(x.unique()))
                                                 })
        user_items_test_df = user_items_test_df.rename(columns={self.item_id_col: 'items'})\
                                                      .reset_index()
        for _, user_items in user_items_test_df.iterrows():
            user = user_items[self.user_id_col]
            items = [item for item in user_items['items']]
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

        LOGGER.debug("Train Data :: Loading Item Profiles")
        item_profile_train_file = os.path.join(self.model_dir, 'item_profile_train.json')
        json_item_profile_train_dict = utilities.load_json_file(item_profile_train_file)
        for item_id, json_profile in json_item_profile_train_dict.items():
            profile = dict()
            for key, val in json_profile.items():
                if isinstance(val, list):
                    profile[key] = set(val)
                else:
                    profile[key] = val
            self.item_profile_train_dict[item_id] = profile                    
            
        LOGGER.debug("Train Data :: Loading Distinct Items for each User")
        user_items_train_file = os.path.join(self.model_dir, 'user_items_train.json')
        self.user_items_train_dict = utilities.load_json_file(user_items_train_file)
        
        LOGGER.debug("Train Data :: Loading Age for each User")
        train_user_age_file = os.path.join(self.model_dir, 'train_user_age.json')
        self.train_user_age_dict = utilities.load_json_file(train_user_age_file)
        ############################################################################
        LOGGER.debug("Test Data :: Loading Stats...")
        users_items_test_file = os.path.join(self.model_dir, 'users_items_test.json')
        users_items_test_dict = utilities.load_json_file(users_items_test_file)
        self.users_test = users_items_test_dict['users_test']
        LOGGER.debug("Test Data :: No. of users : " + str(len(self.users_test)))
        self.items_test = users_items_test_dict['items_test']
        LOGGER.debug("Test Data :: No. of items : " + str(len(self.items_test)))
        
        LOGGER.debug("Test Data :: Loading Item Profiles")
        item_profile_test_file = os.path.join(self.model_dir, 'item_profile_test.json')
        json_item_profile_test_dict = utilities.load_json_file(item_profile_test_file)
        for item_id, json_profile in json_item_profile_test_dict.items():
            profile = dict()
            for key, val in json_profile.items():
                if isinstance(val, list):
                    profile[key] = set(val)
                else:
                    profile[key] = val
            self.item_profile_test_dict[item_id] = profile
            
        LOGGER.debug("Test Data :: Loading Distinct Items for each User")
        user_items_test_file = os.path.join(self.model_dir, 'user_items_test.json')
        self.user_items_test_dict = utilities.load_json_file(user_items_test_file)        

        LOGGER.debug("Test Data :: Loading Age for each User")
        test_user_age_file = os.path.join(self.model_dir, 'test_user_age.json')
        self.test_user_age_dict = utilities.load_json_file(test_user_age_file)
        
    def train(self):
        """train the content based recommender system model"""
        start_time = default_timer()
        self.__derive_stats()
        print("Training...")
        end_time = default_timer()
        print("{:50}    {}".format("Training Completed in : ",
                                   utilities.convert_sec(end_time - start_time)))

    def __load_item_profile(self, item_id, dataset='train'):
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
        
    def get_user_profile(self, user_items, user_age, dataset='train'):        
        user_profile_name_tokens = set()
        user_profile_authors = set()
        user_profile_keywords = set()
        user_profile_begin_target_ages = set()
        user_profile_end_target_ages = set()
        for item_id in user_items:
            item_profile = self.__load_item_profile(item_id, dataset)
            #print(item_id)
            #pprint(item_profile)
            #print('%'*5)
            
            name_tokens = item_profile['name_tokens']
            user_profile_name_tokens = user_profile_name_tokens.union(name_tokens)
            
            author = item_profile['author']
            if isinstance(author, str):
                user_profile_authors.add(author)
                  
            keywords = item_profile['keywords']
            user_profile_keywords = user_profile_keywords.union(keywords)

            '''
            begin_target_age = item_profile['begin_target_age']
            end_target_age = item_profile['end_target_age']
            if isinstance(begin_target_age, str):
                user_profile_begin_target_ages.add(begin_target_age)
            if isinstance(end_target_age, str):
                user_profile_end_target_ages.add(end_target_age)
            '''
        '''
        profile = {'name_tokens' : user_profile_name_tokens, 'author' : user_profile_authors,
                   'keywords' : user_profile_keywords,
                   'begin_target_age' : user_profile_begin_target_ages,
                   'end_target_age' : user_profile_end_target_ages
                  }
        '''
        profile = {'name_tokens' : user_profile_name_tokens,
                   'author' : user_profile_authors,
                   'keywords' : user_profile_keywords,
                   'user_age' : user_age
                  }        
        return profile
    
    def get_jaccard_similarity(self, set_a, set_b):
        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)
        if len(union) != 0:
            return len(intersection) / len(union)
        else:
            return 0

    def __get_similarity_score(self, user_profile, item_profile):    
        name_tokens_similarity = self.get_jaccard_similarity(user_profile['name_tokens'],
                                                             item_profile['name_tokens'])        
        authors_similarity = self.get_jaccard_similarity(user_profile['author'],
                                                         item_profile['author'])
        keywords_similarity = self.get_jaccard_similarity(user_profile['keywords'],
                                                          item_profile['keywords'])
        
        score = name_tokens_similarity*0.25 + authors_similarity*0.25 + keywords_similarity*0.5
        '''
        print("\tname : {}, author : {}, keywords : {}, score : {} ".format(name_tokens_similarity,
                                                                          authors_similarity,
                                                                          keywords_similarity,
                                                                         score))  
        '''
        return (name_tokens_similarity, authors_similarity, keywords_similarity, score)

    def __generate_top_recommendations(self, items_interacted, user_profile):        
        #Get all items from train data and recommend them which are most similar to user_profile
        items_train = self.__get_all_items(dataset='train')
        item_scores = []
        for item_id in items_train:
            if item_id in items_interacted:#this item was already interacted by the user, hence do not recommend
                continue
            
            item_profile = self.__load_item_profile(item_id, dataset='train')                        
            #print("\n\t" + item_id)
            #print(item_profile)
            (name_tokens_similarity, authors_similarity, keywords_similarity, sim_score) = self.__get_similarity_score(user_profile, item_profile)
            item_scores.append({self.item_id_col : item_id,
                                'name_tokens_similarity': name_tokens_similarity,
                                'authors_similarity': authors_similarity,
                                'keywords_similarity': keywords_similarity,
                                'sim_score': sim_score})
        item_scores_df = pd.DataFrame(item_scores)
        #print(item_scores_df)
        #Sort the items based upon similarity scores
        item_scores_df = item_scores_df.sort_values(['sim_score', self.item_id_col],
                                                         ascending=[0, 1])        
        #print(item_scores_df)
        #Generate a recommendation rank based upon score
        item_scores_df['rank'] = item_scores_df['sim_score'].rank(ascending=0, method='first')
        item_scores_df.reset_index(drop=True, inplace=True)
        #print(item_scores_df)
        #return item_scores_df.head(len(items_interacted))
        return item_scores_df.head(self.no_of_recs)
       
    def recommend_items(self, user_id, dataset='test'):
        """Generate item recommendations for given user_id from chosen dataset"""
        self.__load_stats()
        start_time = default_timer()
        # Get all items with which user has interacted
        items_interacted = self.__get_items(user_id, dataset)
        user_age = self.__get_user_age(user_id, dataset)
        user_profile = self.get_user_profile(items_interacted, user_age, dataset)        
        print("User Profile")
        print(user_profile)
        print()
        
        self.recommendations = self.__generate_top_recommendations(items_interacted, user_profile)
        print(self.recommendations)
        recommended_items = self.recommendations[self.item_id_col].tolist()
        for item_id in recommended_items:
            print('*'*30)
            print("Recommended : ", item_id)
            item_profile = self.__load_item_profile(item_id, dataset='train')
            print(item_profile)
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated in : ",
                                   utilities.convert_sec(end_time - start_time)))
        return recommended_items

    def __get_known_items(self, items_interacted):
        """return filtered items which are present in training set"""
        known_items_interacted = []
        items_training_set = self.__get_all_items(dataset='train')
        for item in items_interacted:
            if item in items_training_set:
                known_items_interacted.append(item)
        return known_items_interacted

    def __get_items_for_eval(self, dataset='test', hold_out_ratio=0.5):
        """Generate recommended and interacted items for users"""
        '''
        eval_items_grps = dict()
        '''
        eval_items = dict()
        users = self.__get_all_users(dataset)
        no_of_users = len(users)
        no_of_users_considered = 0
        for user_id in users:
            #print(user_id)           
            
            # Get all items with which user has interacted
            items_interacted = self.__get_items(user_id, dataset)
            #print("items_interacted in test")
            #print("all_items_interacted in test set")
            #print(items_interacted)
            if dataset != 'train':
                items_interacted = self.__get_known_items(items_interacted)
                #print("known items in train")
                #print(items_interacted)
            #print("all_items_interacted which are present in train set")
            #print(items_interacted)
            assume_interacted_items, hold_out_items = self.split_items(items_interacted,
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

            user_age = self.__get_user_age(user_id, dataset)
            user_profile = self.get_user_profile(assume_interacted_items, user_age, dataset)
            #print(user_id)
            #print(user_profile)
            
            user_recommendations = self.__generate_top_recommendations(assume_interacted_items, user_profile)
            recommended_items = list(user_recommendations[self.item_id_col].values)
            eval_items[user_id]['items_recommended'] = recommended_items
            eval_items[user_id]['assume_interacted_items'] = assume_interacted_items
            eval_items[user_id]['items_interacted'] = hold_out_items
            
            '''
            no_of_items_recommended = len(recommended_items)
            reco = {user_id : eval_items[user_id]}
            if no_of_items_recommended not in eval_items_grps:
                eval_items_grps[no_of_items_recommended] = [reco]
            else:
                eval_items_grps[no_of_items_recommended].append(reco)
            '''
            
            
            #for debug purpose
            #Verify Evaluation
            '''
            no_of_items_to_recommend_list = [1, 2, 5, 10]
            for no_of_items_to_recommend in no_of_items_to_recommend_list:
                print("no_of_items_to_recommend : ", no_of_items_to_recommend)
                
                items_interacted = eval_items[user_id]['items_interacted']
                assume_interacted_items = eval_items[user_id]['assume_interacted_items']
                items_recommended = eval_items[user_id]['items_recommended'][0:no_of_items_to_recommend]            
                print("Items Expected to be Interacted : ", set(items_interacted))
                print("Assume Items Interacted : ", set(assume_interacted_items))
                print("Items Recommended: ", set(items_recommended))

                
                print("\n assume_interacted_items : ")
                for item_id in assume_interacted_items:
                    print(item_id)
                    item_profile = self.__load_item_profile(item_id, dataset='train')
                    print(item_profile)
                print("\n hold_out_items : ")
                for item_id in hold_out_items:
                    print(item_id)
                    item_profile = self.__load_item_profile(item_id, dataset='train')
                    print(item_profile)
                print("\n items_recommended : ")
                for item_id in items_recommended:
                    print(item_id)
                    item_profile = self.__load_item_profile(item_id, dataset='train')
                    print(item_profile)                    
                
                hitset = set(items_interacted).intersection(set(items_recommended))
                print("Hitset : ", hitset)
                no_of_items_interacted = len(items_interacted)
                #precision is the proportion of recommendations that are good recommendations
                precision = float(len(hitset))/no_of_items_to_recommend
                #recall is the proportion of good recommendations that appear in top recommendations
                recall = float(len(hitset))/no_of_items_interacted
                if (recall+precision) != 0:
                    f1_score = float(2*precision*recall)/(recall+precision)
                else:
                    f1_score = 0.0
                print("Precision : {}, Recall : {}, F1-Score : {}".format(precision, recall, f1_score))
                input()
            #for debug purpose
            '''
            
        print("Evaluation : No of users : ", no_of_users)
        print("Evaluation : No of users considered : ", no_of_users_considered)
        return eval_items

    def evaluate(self, no_of_recs_to_eval, dataset='test', hold_out_ratio=0.5):
        """Evaluate trained model"""
        print("Evaluating...")
        self.__load_stats()
        start_time = default_timer()

        #Generate recommendations for the users
        eval_items = self.__get_items_for_eval(dataset, hold_out_ratio)
        precision_recall_eval_file = os.path.join(self.model_dir, 'eval_items.json')
        utilities.dump_json_file(eval_items, precision_recall_eval_file)
        #precision_recall_eval_grp_file = os.path.join(self.model_dir, 'eval_items_grps.json')
        #utilities.dump_json_file(eval_items_grps, precision_recall_eval_grp_file)        
        #pprint(eval_items)

        precision_recall_intf = PrecisionRecall()
        #results = precision_recall_intf.compute_precision_recall1(eval_items_grps)
        results = precision_recall_intf.compute_precision_recall(no_of_recs_to_eval, eval_items)
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
    model = ContentBasedRecommender(results_dir, model_dir,
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
    model = ContentBasedRecommender(results_dir, model_dir,
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

    model = ContentBasedRecommender(results_dir, model_dir,
                                       train_data, test_data,
                                       user_id_col, item_id_col, no_of_recs)
       
    print("Items interactions for a user with user_id : {}".format(user_id))
    interacted_items = list(test_data[test_data[user_id_col] == user_id][item_id_col])
    for item in interacted_items:
        print(item)
            
    print()
    print("Items recommended for a user with user_id : {}".format(user_id))
    recommended_items = model.recommend_items(user_id, dataset='test')    
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
    model = ContentBasedRecommender(results_dir, model_dir,
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
    recommended_items = model.recommend_items(user_id, dataset='test')
    print()
    if recommended_items:
        for item in recommended_items:
            print(item)
    else:
        print("No items to recommend")
    print('*' * 80)

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

    no_of_recs = 10
    user_id_col = 'learner_id'
    item_id_col = 'book_code'
    train_test_dir = os.path.join(current_dir, 'train_test_data')
    
    if args.train:
        train(results_dir, model_dir, train_test_dir,
              user_id_col, item_id_col, no_of_recs=no_of_recs)
    elif args.eval:
        no_of_recs_to_eval = [1, 2, 5, 10]
        evaluate(results_dir, model_dir, train_test_dir,
                 user_id_col, item_id_col,
                 no_of_recs_to_eval, dataset='test', no_of_recs=no_of_recs)
    elif args.recommend and args.user_id:
        recommend(results_dir, model_dir, train_test_dir,
                  user_id_col, item_id_col,
                  args.user_id, no_of_recs=no_of_recs)
    else:
        no_of_recs_to_eval = [1, 2, 5, 10]
        train_eval_recommend(results_dir, model_dir, train_test_dir,
                             user_id_col, item_id_col,
                             no_of_recs_to_eval,
                             dataset='test', no_of_recs=no_of_recs)

if __name__ == '__main__':
    main()
