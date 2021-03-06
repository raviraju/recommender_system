"""Module for Books Recommender"""
import os
import sys
import logging

from collections import defaultdict
from collections import Counter
import math

from pprint import pprint

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
from recommender import rec_interface as generic_rec_interface

class BooksRecommender(generic_rec_interface.Recommender):
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
    preprocessed_token = re.sub('[^a-zA-Z0-9]', '', token)
    preprocessed_token = preprocessed_token.lower()
    if preprocessed_token not in ENGLISH_STOPWORDS:
        return preprocessed_token
    else:
        return None

def get_book_keywords(dataframe, book_code):
    """keywords for given book_code"""
    book_keywords = []
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
                    book_keywords.append(preprocessed_token)
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
    book_name_tokens_set = set()
    if isinstance(book_name, str):
        tokens = book_name.split(' ')
        for token in tokens:
            preprocessed_token = preprocess_token(token)
            if preprocessed_token:
                book_name_tokens_set.add(preprocessed_token)
    return book_name, book_name_tokens_set

def get_item_profile(dataframe, book_code):
    """item profile for a given book_code"""
    item_keywords = get_book_keywords(dataframe, book_code)
    _, item_name_tokens_set = get_book_name(dataframe, book_code)
    author = get_author(dataframe, book_code)
    item_author_set = set()
    if isinstance(author, str):
        item_author_set.add(author)
    item_profile = {'name_tokens': list(item_name_tokens_set),
                    'author': list(item_author_set),
                    'keywords': item_keywords}
    return item_profile

def get_user_profile(dataframe, user_items):
    """return user profile by merging item profiles
    for user interacted items"""
    user_profile_name_tokens_set = set()
    user_profile_authors_set = set()
    user_profile_keywords = []

    for item_id in user_items:
        item_profile = get_item_profile(dataframe, item_id)

        # print(item_id)
        # print(item_profile)
        # print('%'*5)

        name_tokens_set = set(item_profile['name_tokens'])
        user_profile_name_tokens_set |= name_tokens_set

        author_set = set(item_profile['author'])
        user_profile_authors_set |= author_set

        keywords = item_profile['keywords']
        user_profile_keywords.extend(keywords)

    user_profile = {'name_tokens': list(user_profile_name_tokens_set),
                    'author': list(user_profile_authors_set),
                    'keywords': user_profile_keywords
                   }
    #print(user_profile)
    return user_profile

def get_jaccard_similarity(set_a, set_b):
    """jaccard similarity"""
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if len(union) != 0:
        return len(intersection) / len(union)
    else:
        return 0

def get_log_freq_weight(term_frequency):
    """log term_frequency weight"""
    if term_frequency > 0:
        return 1 + math.log10(term_frequency)
    else:
        return 0

def get_term_freq_similarity1(word_list_a, word_list_b):
    """term frequency similarity"""
    tf_a = defaultdict(int)
    tf_b = defaultdict(int)
    set_a = set(word_list_a)
    set_b = set(word_list_b)

    for term in word_list_a:
        tf_a[term] += 1
    for term in word_list_b:
        tf_b[term] += 1

    #print(tf_a)
    #print(tf_b)
    intersection = set_a.intersection(set_b)
    score = 0
    for common_term in intersection:
        log_freq_weight_a = get_log_freq_weight(tf_a[common_term])
        log_freq_weight_b = get_log_freq_weight(tf_b[common_term])

        log_freq_weight_avg = (log_freq_weight_a+log_freq_weight_b)/2
        '''
        print(common_term,
              tf_a[common_term], log_freq_weight_a,
              tf_b[common_term], log_freq_weight_b,
              log_freq_weight_avg)
        '''
        score = score + log_freq_weight_avg
    return score

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return round(dotprod / (magA * magB), 2)

def length_similarity(lenc1, lenc2):
    return min(lenc1, lenc2) / float(max(lenc1, lenc2))

def get_term_freq_similarity(l1, l2):
    """term frequency similarity"""
    len1 = len(l1)
    len2 = len(l2)
    if len1 == 0 or len2 == 0:
        return 0.0
    c1, c2 = Counter(l1), Counter(l2)
    return length_similarity(len1, len2) * counter_cosine_similarity(c1, c2)

def get_subset_similarity(child_set, parent_set):
    """subset similarity"""
    return int(child_set.issubset(parent_set))

def get_sublist_similarity(child_list, parent_list):
    """term frequency similarity if child_list is contained in parent_list"""
    parent_list_len = len(parent_list)
    child_list_len = len(child_list)
    if child_list_len == 0:#empty lists
        return 1
    parent_counter = Counter(parent_list)
    child_counter = Counter(child_list)

    no_of_contained_terms = 0
    for term in child_counter:
        tf_child = child_counter[term]
        if term in parent_counter:
            tf_parent = parent_counter[term]
            #print(term, tf_child, tf_parent)
            if tf_child <= tf_parent:
                no_of_contained_terms += 1
    if no_of_contained_terms == len(child_counter):
        return 1
    else:
        return 0

def get_profile_similarity_score(user_profile, item_profile):
    """similarity scores bw user and item profile"""
    name_tokens_similarity = get_subset_similarity(set(item_profile['name_tokens']),
                                                   set(user_profile['name_tokens']))
    authors_similarity = get_subset_similarity(set(item_profile['author']),
                                               set(user_profile['author']))
#     keywords_similarity = get_sublist_similarity(item_profile['keywords'],
#                                                  user_profile['keywords'])
    keywords_similarity = get_subset_similarity(set(item_profile['keywords']),
                                                set(user_profile['keywords']))
    '''
    print("\tname : {}, author : {}, keywords : {}, score : {} ".format(name_tokens_similarity,
                                                                        authors_similarity,
                                                                        keywords_similarity))
    '''
    return (name_tokens_similarity, authors_similarity, keywords_similarity)

def get_similarity_score(train_data, test_data, recommended_item, interacted_item):
    """content based similarity score bw recommended and interacted item"""
    item_profile_a = get_item_profile(train_data, recommended_item)
    item_name_tokens_set_a, item_author_set_a, item_keywords_a = set(item_profile_a['name_tokens']), set(item_profile_a['author']), item_profile_a['keywords']
    item_profile_b = get_item_profile(test_data, interacted_item)
    item_name_tokens_set_b, item_author_set_b, item_keywords_b = set(item_profile_b['name_tokens']), set(item_profile_b['author']), item_profile_b['keywords']

    item_name_tokens_similarity = get_jaccard_similarity(item_name_tokens_set_a,
                                                         item_name_tokens_set_b)
    authors_similarity = get_jaccard_similarity(item_author_set_a, item_author_set_b)
    keywords_similarity = get_term_freq_similarity(item_keywords_a, item_keywords_b)
    print("\t {:30s} : {}".format('item_name_tokens_similarity',
                                  item_name_tokens_similarity))
    print("\t {:30s} : {}".format('authors_similarity', authors_similarity))
    print("\t {:30s} : {}".format('keywords_similarity', keywords_similarity))
    score = dict()
    score['item_name_similarity'] = item_name_tokens_similarity
    score['author_name_similarity'] = authors_similarity
    score['keywords_similarity'] = keywords_similarity
    return score

def recommend(recommender_obj,
              results_dir, model_dir,
              train_data_file, test_data_file,
              user_id_col, item_id_col,
              user_id, metadata_fields, **kwargs):
    """recommend items for user"""
    train_data, test_data = generic_rec_interface.load_train_test(train_data_file,
                                                                  test_data_file,
                                                                  user_id_col,
                                                                  item_id_col)
    recommender = recommender_obj(results_dir, model_dir,
                                  train_data, test_data,
                                  user_id_col, item_id_col,
                                  **kwargs)

    eval_items_file = os.path.join(model_dir, 'items_for_evaluation.json')
    eval_items = utilities.load_json_file(eval_items_file)
    if user_id in eval_items:
        assume_interacted_items = eval_items[user_id]['assume_interacted_items']
        items_interacted = eval_items[user_id]['items_interacted']

        print("Assumed Item interactions for a user with user_id : {}".format(user_id))
        for item in assume_interacted_items:
            print(item)
            if metadata_fields is not None:
                item_profile = get_item_profile(test_data, item)
                print("\t item_name_tokens : {}".format(item_profile['name_tokens']))
                print("\t item_author : {}".format(item_profile['author']))
                print("\t item_keywords : {}".format(item_profile['keywords']))
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
                item_profile = get_item_profile(test_data, item)
                print("\t item_name_tokens : {}".format(item_profile['name_tokens']))
                print("\t item_author : {}".format(item_profile['author']))
                print("\t item_keywords : {}".format(item_profile['keywords']))
                print()

                record = test_data[test_data[item_id_col] == item]
                if not record.empty:
                    for field in metadata_fields:
                        print("\t {} : {}".format(field, record[field].values[0]))
                print('\t '+ '#'*30)

        print()
        print("Items recommended for a user with user_id : {}".format(user_id))
        recommended_items = list(recommender.recommend_items(user_id)[item_id_col])
        print()        
        if len(recommended_items) > 0:
            for recommended_item in recommended_items:
                print(recommended_item)
                input()
                if metadata_fields is not None:
                    item_profile = get_item_profile(train_data, recommended_item)
                    print("\t item_name_tokens : {}".format(item_profile['name_tokens']))
                    print("\t item_author : {}".format(item_profile['author']))
                    print("\t item_keywords : {}".format(item_profile['keywords']))
                    print()

                    record = train_data[train_data[item_id_col] == recommended_item]
                    if not record.empty:
                        for field in metadata_fields:
                            print("\t {} : {}".format(field, record[field].values[0]))
                    for interacted_item in items_interacted:
                        score = get_similarity_score(train_data,
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
