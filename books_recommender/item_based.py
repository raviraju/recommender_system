"""Module for Item Based CF Books Recommender"""
import os
import sys
import argparse

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.reco_interface import load_train_test
from recommender import item_based_cf_opt

def preprocess_token(token):
    """preprocessing of tokens"""
    preprocessed_token = re.sub('[^a-zA-Z]', '', token)
    preprocessed_token = preprocessed_token.lower()
    #print(token, preprocessed_token)
    if preprocessed_token not in ENGLISH_STOPWORDS:
        return preprocessed_token
    else:
        return None

def get_book_keywords(dataframe, book_code):
    """book keywords"""
    book_keywords = set()
    records = dataframe[dataframe['BOOK_CODE'] == book_code]
    for _, record in records.iterrows():
        first_record = record
        break
    #print(first_record)
    keywords = first_record['T_KEYWORD']#.values[0]
    #print(book_code, keywords)
    if isinstance(keywords, str):
        keywords = keywords.split('|')
        #print(keywords)
        for keyword in keywords:
            tokens = keyword.split(' ')
            #print(tokens)
            for token in tokens:
                preprocessed_token = preprocess_token(token)
                if preprocessed_token:
                    book_keywords.add(preprocessed_token)
    #print(book_keywords)
    #print('*'*30)
    return book_keywords

def get_target_age_range(dataframe, book_code):
    """age range"""
    records = dataframe[dataframe['BOOK_CODE'] == book_code]
    for _, record in records.iterrows():
        first_record = record
        break
    #print(first_record)
    begin_target_age = first_record['BEGIN_TARGET_AGE']#.values[0]
    end_target_age = first_record['END_TARGET_AGE']#.values[0]
    return begin_target_age, end_target_age

def get_author(dataframe, book_code):
    """author name"""
    records = dataframe[dataframe['BOOK_CODE'] == book_code]
    for _, record in records.iterrows():
        first_record = record
        break
    #print(first_record)
    author = first_record['T_AUTHOR']#.values[0]
    return author

def get_book_name(dataframe, book_code):
    """book name"""
    records = dataframe[dataframe['BOOK_CODE'] == book_code]
    for _, record in records.iterrows():
        first_record = record
        break
    #print(first_record)
    book_name = first_record['T_BOOK_NAME']#.values
    #print(book_name)
    book_name_tokens = set()
    if isinstance(book_name, str):
        tokens = book_name.split(' ')
        for token in tokens:
            preprocessed_token = preprocess_token(token)
            if preprocessed_token:
                book_name_tokens.add(preprocessed_token)
    #print(book_name, book_name_tokens)
    return book_name, book_name_tokens

def get_item_profile(dataframe, book_code):
    """item profile"""
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
    item_name_tokens_a, item_author_a, item_keywords_a = get_item_profile(train_data, recommended_item)
    item_name_tokens_b, item_author_b, item_keywords_b = get_item_profile(test_data, interacted_item)

    authors_pair = (item_author_a, item_author_b)
    keywords_pair = (item_keywords_a, item_keywords_b)
    item_name_tokens_pair = (item_name_tokens_a, item_name_tokens_b)

    item_name_tokens_similarity = get_jaccard_similarity(item_name_tokens_pair[0],
                                                         item_name_tokens_pair[1])
    authors_similarity = get_jaccard_similarity(authors_pair[0], authors_pair[1])
    keywords_similarity = get_jaccard_similarity(keywords_pair[0], keywords_pair[1])
    print("\t {:30s} : {}".format('item_name_tokens_similarity', item_name_tokens_similarity))
    print("\t {:30s} : {}".format('authors_similarity', authors_similarity))
    print("\t {:30s} : {}".format('keywords_similarity', keywords_similarity))
    score = item_name_tokens_similarity*0.25 + authors_similarity*0.25 + keywords_similarity*0.5
    return score

def recommend(results_dir, model_dir, train_test_dir,
              user_id_col, item_id_col,
              user_id, no_of_recs=10, dataset='test', metadata_fields=None):
    """recommend items for user"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    model = item_based_cf_opt.ItemBasedCFRecommender(results_dir, model_dir,
                                                     train_data, test_data,
                                                     user_id_col, item_id_col, no_of_recs)

    print("Items interactions for a user with user_id : {}".format(user_id))
    interacted_items = list(test_data[test_data[user_id_col] == user_id][item_id_col])
    for item in interacted_items:
        print(item)
        if metadata_fields is not None:
            item_name_tokens, item_author, item_keywords = get_item_profile(test_data, item)
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
    recommended_items = model.recommend_items(user_id, dataset)
    print()
    if recommended_items:
        for recommended_item in recommended_items:
            print(recommended_item)
            if metadata_fields is not None:
                item_name_tokens, item_author, item_keywords = get_item_profile(train_data,
                                                                                recommended_item)
                print("\t item_name_tokens : {}".format(item_name_tokens))
                print("\t item_author : {}".format(item_author))
                print("\t item_keywords : {}".format(item_keywords))
                print()

                record = train_data[train_data[item_id_col] == recommended_item]
                if not record.empty:
                    for field in metadata_fields:
                        print("\t {} : {}".format(field, record[field].values[0]))
                for interacted_item in interacted_items:
                    score = get_similarity_score(train_data, test_data,
                                                     recommended_item,
                                                     interacted_item)
                    print("\t {:20s} | {:20s} | {}".format('recommended_item', 'interacted_item', 'score'))
                    print("\t {:20s} | {:20s} | {}".format(recommended_item, interacted_item, score))
                    print()
                print('\t '+ '#'*30)
    else:
        print("No items to recommend")
    print('*' * 80)

def main():
    """Item based recommender interface"""
    parser = argparse.ArgumentParser(description="Item Based Recommender")
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
    model_dir = os.path.join(current_dir, 'model/item_based')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    no_of_recs = 10
    user_id_col = 'learner_id'
    item_id_col = 'book_code'
    train_test_dir = os.path.join(current_dir, 'train_test_data')
    metadata_fields = None
    metadata_fields = ['T_BOOK_NAME', 'T_KEYWORD', 'T_AUTHOR']

    if args.train:
        item_based_cf_opt.train(results_dir, model_dir, train_test_dir,
                                user_id_col, item_id_col, no_of_recs=no_of_recs)
    elif args.eval:
        no_of_recs_to_eval = [1, 2, 5, 10]
        item_based_cf_opt.evaluate(results_dir, model_dir, train_test_dir,
                                   user_id_col, item_id_col,
                                   no_of_recs_to_eval, dataset='test',
                                   no_of_recs=no_of_recs, hold_out_ratio=0.5)
    elif args.recommend and args.user_id:
        recommend(results_dir, model_dir, train_test_dir,
                  user_id_col, item_id_col,
                  args.user_id, no_of_recs=no_of_recs,
                  metadata_fields=metadata_fields)
    else:
        no_of_recs_to_eval = [1, 2, 5, 10]
        item_based_cf_opt.train_eval_recommend(results_dir, model_dir, train_test_dir,
                                               user_id_col, item_id_col,
                                               no_of_recs_to_eval, dataset='test',
                                               no_of_recs=no_of_recs, hold_out_ratio=0.5)

if __name__ == '__main__':
    main()
