"""Module for Books Recommender"""
import os
import sys
import logging

import re
import nltk
from nltk.corpus import stopwords

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

try:
    ENGLISH_STOPWORDS = set(stopwords.words("english"))
except LookupError as err:
    print("Download NLTK Stop words corpus")
    nltk.download('stopwords')
    ENGLISH_STOPWORDS = set(stopwords.words("english"))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender.rec_interface import Recommender

class BooksRecommender(Recommender):
    """encapsulating common functionality for books recommender use case"""
    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)
        self.book_access_time_train = dict()
        self.book_access_time_test = dict()

    def derive_stats(self):
        """derive use case specific stats"""
        super().derive_stats()

        LOGGER.debug("Train Data       :: Getting Access Time for each User-Item")
        self.book_access_time_train = dict()
        for _, row in self.train_data.iterrows():
            user = row['learner_id']
            item = row['book_code']
            if user in self.book_access_time_train:
                self.book_access_time_train[user][item] = row['first_access_time']
            else:
                self.book_access_time_train[user] = {item : row['first_access_time']}
        #pprint(self.book_access_time_train)
        book_access_time_train_file = os.path.join(self.model_dir, 'book_access_time_train.json')
        utilities.dump_json_file(self.book_access_time_train, book_access_time_train_file)

        LOGGER.debug("Test Data        :: Getting Access Time for each User-Item")
        for _, row in self.test_data.iterrows():
            user = row['learner_id']
            item = row['book_code']
            if user in self.book_access_time_test:
                self.book_access_time_test[user][item] = row['first_access_time']
            else:
                self.book_access_time_test[user] = {item : row['first_access_time']}
        #pprint(self.book_access_time_test)
        book_access_time_test_file = os.path.join(self.model_dir, 'book_access_time_test.json')
        utilities.dump_json_file(self.book_access_time_test, book_access_time_test_file)

    def load_stats(self):
        """load use case specific stats"""
        super().load_stats()

        LOGGER.debug("Train Data       :: Loading Access Time for each User-Item")
        book_access_time_train_file = os.path.join(self.model_dir, 'book_access_time_train.json')
        self.book_access_time_train = utilities.load_json_file(book_access_time_train_file)

        LOGGER.debug("Test Data        :: Loading Access Time for each User-Item")
        book_access_time_test_file = os.path.join(self.model_dir, 'book_access_time_test.json')
        self.book_access_time_test = utilities.load_json_file(book_access_time_test_file)

    def get_items(self, user_id, dataset='train'):
        """Get unique items for a given user in sorted order of access time """
        if dataset == "train":
            items_access = self.book_access_time_train[user_id]
        else:#test
            items_access = self.book_access_time_test[user_id]
        #pprint(items_access)
        sorted_items = sorted(items_access.items(), key=lambda p: p[1])
        #pprint(sorted_items)
        items_access_ordered = []
        for item, _ in sorted_items:
            items_access_ordered.append(item)
        #print(items_access_ordered)
        #input()
        return items_access_ordered

def preprocess_token(token):
    """preprocess token while filtering stop words"""
    preprocessed_token = re.sub('[^a-zA-Z]', '', token)
    preprocessed_token = preprocessed_token.lower()
    if preprocessed_token not in ENGLISH_STOPWORDS:
        return preprocessed_token
    else:
        return None

def get_book_keywords(dataframe, book_code):
    """keywords for given book_code"""
    book_keywords = set()
    records = dataframe[dataframe['BOOK_CODE'] == book_code]
    for _, record in records.iterrows():
        first_record = record
        break
    keywords = first_record['T_KEYWORD']
    if isinstance(keywords, str):
        keywords = keywords.split('|')
        for keyword in keywords:
            tokens = keyword.split(' ')
            for token in tokens:
                preprocessed_token = preprocess_token(token)
                if preprocessed_token:
                    book_keywords.add(preprocessed_token)
    return book_keywords

def get_target_age_range(dataframe, book_code):
    """age range target for a given book_code"""
    records = dataframe[dataframe['BOOK_CODE'] == book_code]
    for _, record in records.iterrows():
        first_record = record
        break
    begin_target_age = first_record['BEGIN_TARGET_AGE']
    end_target_age = first_record['END_TARGET_AGE']
    return begin_target_age, end_target_age

def get_author(dataframe, book_code):
    """author name for a given book_code"""
    records = dataframe[dataframe['BOOK_CODE'] == book_code]
    for _, record in records.iterrows():
        first_record = record
        break
    author = first_record['T_AUTHOR']
    return author

def get_book_name(dataframe, book_code):
    """book name for a given book_code"""
    records = dataframe[dataframe['BOOK_CODE'] == book_code]
    for _, record in records.iterrows():
        first_record = record
        break
    book_name = first_record['T_BOOK_NAME']
    book_name_tokens = set()
    if isinstance(book_name, str):
        tokens = book_name.split(' ')
        for token in tokens:
            preprocessed_token = preprocess_token(token)
            if preprocessed_token:
                book_name_tokens.add(preprocessed_token)
    return book_name, book_name_tokens

def get_item_profile(dataframe, book_code):
    """item profile for a given book_code"""
    item_keywords = get_book_keywords(dataframe, book_code)
    name, item_name_tokens = get_book_name(dataframe, book_code)
    author = get_author(dataframe, book_code)
    item_author = set()
    if isinstance(author, str):
        item_author.add(author)
    return item_name_tokens, item_author, item_keywords

def get_jaccard_similarity(set_a, set_b):
    """jaccard similarity"""
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if len(union) != 0:
        return len(intersection) / len(union)
    else:
        return 0

def get_similarity_score(train_data, test_data, recommended_item, interacted_item):
    """content based similarity score"""
    item_profile_a = get_item_profile(train_data, recommended_item)
    item_name_tokens_a, item_author_a, item_keywords_a = item_profile_a
    item_profile_b = get_item_profile(test_data, interacted_item)
    item_name_tokens_b, item_author_b, item_keywords_b = item_profile_b

    authors_pair = (item_author_a, item_author_b)
    keywords_pair = (item_keywords_a, item_keywords_b)
    item_name_tokens_pair = (item_name_tokens_a, item_name_tokens_b)

    item_name_tokens_similarity = get_jaccard_similarity(item_name_tokens_pair[0],
                                                         item_name_tokens_pair[1])
    authors_similarity = get_jaccard_similarity(authors_pair[0], authors_pair[1])
    keywords_similarity = get_jaccard_similarity(keywords_pair[0], keywords_pair[1])
    print("\t {:30s} : {}".format('item_name_tokens_similarity',
                                  item_name_tokens_similarity))
    print("\t {:30s} : {}".format('authors_similarity', authors_similarity))
    print("\t {:30s} : {}".format('keywords_similarity', keywords_similarity))
    score = item_name_tokens_similarity*0.25 + authors_similarity*0.25 + keywords_similarity*0.5
    return score
