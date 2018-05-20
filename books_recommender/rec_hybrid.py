"""Module for Hybrid of Books Recommenders"""
import os
import sys
import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender import rec_interface as generic_rec_interface

from rec_random_based import RandomBasedRecommender
from rec_popularity_based import PopularityBasedRecommender

from rec_item_based_cf import ItemBasedCFRecommender
from rec_user_based_cf import UserBasedCFRecommender

from rec_content_based import ContentBasedRecommender
from rec_user_based_age_itp import UserBasedAgeItpRecommender

from rec_content_boosted_item_cf import ContentBoostedItemCFRecommender
from rec_content_boosted_user_cf import ContentBoostedUserCFRecommender

from rec_hybrid_user_based_cf_age_itp import Hybrid_UserBased_CF_AgeItp_Recommender

import rec_interface as books_rec_interface

from pprint import pprint

def get_user_item_meta_info(model_dir, user_id_col, item_id_col, all_data_file, learner_pref_file):
    """derive user and item info"""
    print("Loading All Data")
    if os.path.exists(all_data_file):
        all_data = pd.read_csv(all_data_file, dtype=object)
    else:
        print("Unable to find all data in : ", all_data_file)
        exit(0)

    items_meta_info_dict = all_data[['book_code', 'event_name', 'BEGIN_TARGET_AGE', 'END_TARGET_AGE']]\
                           .drop_duplicates()\
                           .fillna(value="0.0")\
                           .set_index('book_code')\
                           .transpose()\
                           .to_dict()
    items_meta_info_file = os.path.join(model_dir, 'kfold_experiments', 'items_meta_info.json')
    utilities.dump_json_file(items_meta_info_dict, items_meta_info_file)

    learner_pref_df = pd.read_csv(learner_pref_file)

    aggregate_file = os.path.join(model_dir, 'kfold_experiments', 'all_scores_aggregation.csv')
    all_aggregation_df = pd.read_csv(aggregate_file)

    items_for_evaluation_file = os.path.join(model_dir, 'kfold_experiments', 'all_items_for_evaluation.json')
    all_items_for_evaluation = utilities.load_json_file(items_for_evaluation_file)

    users = all_aggregation_df['user_id'].unique()
    print("Fetching All User Profiles...")
    user_profiles = dict()
    item_profiles = dict()
    for user_id in users:
        #print(user_id)
        assumed_set_items = all_items_for_evaluation[str(user_id)]['assume_interacted_items']
        #print(assumed_set_items)
        user_profile = books_rec_interface.get_user_profile(all_data, assumed_set_items)
        user_profiles[str(user_id)] = user_profile
        for item_id in assumed_set_items:
            item_profiles[item_id] = books_rec_interface.get_item_profile(all_data, item_id)

    user_profiles_file = os.path.join(model_dir, 'kfold_experiments', 'user_profiles.json')
    utilities.dump_json_file(user_profiles, user_profiles_file)

    all_aggregation_df['user_age'] = 0.0
    all_aggregation_df['user_audio_close'] = 0.0
    all_aggregation_df['user_book_close'] = 0.0
    all_aggregation_df['user_video_close'] = 0.0

    all_aggregation_df['item_event_name'] = 0.0
    all_aggregation_df['item_begin_target_age'] = 0.0
    all_aggregation_df['item_end_target_age'] = 0.0

    all_aggregation_df['name_tokens_similarity'] = 0.0
    all_aggregation_df['authors_similarity'] = 0.0
    all_aggregation_df['keywords_similarity'] = 0.0

    print("Computing Similarity bw User Profile and Recommended Item...")
    for index, row in all_aggregation_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']

        user_info = learner_pref_df[learner_pref_df[user_id_col] == user_id]
        all_aggregation_df.loc[index, 'user_age'] = float(user_info['age'])
        all_aggregation_df.loc[index, 'user_audio_close'] = float(user_info['audio_close'])
        all_aggregation_df.loc[index, 'user_book_close'] = float(user_info['book_close'])
        all_aggregation_df.loc[index, 'user_video_close'] = float(user_info['video_close'])

        items_meta_info = items_meta_info_dict[item_id]
        all_aggregation_df.loc[index, 'item_event_name'] = items_meta_info['event_name']
        all_aggregation_df.loc[index, 'item_begin_target_age'] = float(items_meta_info['BEGIN_TARGET_AGE'])
        all_aggregation_df.loc[index, 'item_end_target_age'] = float(items_meta_info['END_TARGET_AGE'])

        user_profile = user_profiles[str(user_id)]

        if item_id in item_profiles:
            item_profile = item_profiles[item_id]
        else:
            item_profile = books_rec_interface.get_item_profile(all_data, item_id)
            item_profiles[item_id] = item_profile

        similarity_scores = books_rec_interface.get_profile_similarity_score(user_profile, item_profile)
        (name_tokens_similarity, authors_similarity, keywords_similarity) = similarity_scores
        #print(similarity_scores)
        all_aggregation_df.loc[index, 'name_tokens_similarity'] = name_tokens_similarity
        all_aggregation_df.loc[index, 'authors_similarity'] = authors_similarity
        all_aggregation_df.loc[index, 'keywords_similarity'] = keywords_similarity

    item_profiles_file = os.path.join(model_dir, 'kfold_experiments', 'item_profiles.json')
    utilities.dump_json_file(item_profiles, item_profiles_file)

    all_aggregation_df.to_csv(aggregate_file, index=False)
    print("Updated :", aggregate_file)

def get_filtered_configs(configs, hold_out_strategy):
    """filter configs which adhere to hold_out_strategy"""
    print("Total no of configs : ", len(configs))
    print("Filtering configs which adhere to : ", hold_out_strategy)
    filtered_configs = []
    for config in configs:
        if hold_out_strategy in config['model_dir_name']:
            filtered_configs.append(config)
    print("Filtered no of configs : ", len(filtered_configs))
    pprint(filtered_configs)
    return filtered_configs

def main():
    """Hybrid of Songs Recommenders interface"""
    parser = argparse.ArgumentParser(description="Hybrid of Songs Recommender")
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
    parser.add_argument("--cross_eval",
                        help="Cross Evaluate Trained Model",
                        action="store_true")
    parser.add_argument("--kfolds",
                        help="No of kfold datasets to consider",
                        type=int)
    parser.add_argument("train_data",
                        help="Train Data")
    parser.add_argument("test_data",
                        help="Test Data")
    parser.add_argument("hold_out_strategy",
                        help="assume_ratio/assume_first_n/hold_last_n")
    parser.add_argument("hold_out_value",
                        help="assume_ratio=0.5/assume_first_n=5/hold_last_n=5")
    parser.add_argument("all_data",
                        help="All Data")
#     parser.add_argument("learner_pref",
#                         help="User Preferences")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    
    user_id_col = 'learner_id'
    item_id_col = 'book_code'

    kwargs = dict()
    kwargs['no_of_recs'] = 150 # max no_of_books read is 144

    kwargs['hold_out_strategy'] = args.hold_out_strategy
    if kwargs['hold_out_strategy'] == 'assume_ratio':
        kwargs['assume_ratio'] = float(args.hold_out_value)
    elif kwargs['hold_out_strategy'] == 'assume_first_n':
        kwargs['first_n'] = int(args.hold_out_value)
    elif kwargs['hold_out_strategy'] == 'hold_last_n':
        kwargs['last_n'] = int(args.hold_out_value)
    else:
        print("Invalid hold_out_strategy {} chosen".format(args.hold_out_strategy))
        exit(-1)

    no_of_recs_to_eval = [5, 6, 7, 8, 9, 10]

    model_name_prefix = 'model/' + kwargs['hold_out_strategy']
    configs = [
        {
            'model_dir_name' : model_name_prefix + '_hybrid_item_cf_user_cf_equal',
            'recommenders' : {
                ItemBasedCFRecommender : 0.5,
                UserBasedCFRecommender : 0.5}
        },
        ##########################################################################
        {
            'model_dir_name' : model_name_prefix + '_hybrid_item_cf_content_equal',
            'recommenders' : {
                ItemBasedCFRecommender : 0.5,
                ContentBasedRecommender : 0.5}
        },
        ##########################################################################
        {
            'model_dir_name' : model_name_prefix + '_hybrid_user_cf_content_equal',
            'recommenders' : {
                UserBasedCFRecommender : 0.5,
                ContentBasedRecommender : 0.5}
        },
        ##########################################################################
        {
            'model_dir_name' : model_name_prefix + '_hybrid_item_cf_sub_hybrid_user_cf_jaccard_cosine_equal',
            'recommenders' : {
                ItemBasedCFRecommender : 0.5,
                Hybrid_UserBased_CF_AgeItp_Recommender : 0.5}
        },
        ##########################################################################
        {
            'model_dir_name' : model_name_prefix + '_hybrid_item_cf_user_cf_jaccard_cosine_equal',
            'recommenders' : {
                ItemBasedCFRecommender : 0.34,
                UserBasedCFRecommender : 0.33,
                UserBasedAgeItpRecommender : 0.33}
        },
        ##########################################################################
        {
            'model_dir_name' : model_name_prefix + '_hybrid_item_cf_content_sub_hybrid_user_cf_jaccard_cosine_equal',
            'recommenders' : {
                ItemBasedCFRecommender : 0.34,
                ContentBasedRecommender : 0.33,
                Hybrid_UserBased_CF_AgeItp_Recommender : 0.33}
        },
        ##########################################################################
        {
            'model_dir_name' : model_name_prefix + '_hybrid_item_cf_content_user_cf_jaccard_cosine_equal',
            'recommenders' : {
                ItemBasedCFRecommender : 0.25,
                ContentBasedRecommender : 0.25,
                UserBasedCFRecommender : 0.25,
                UserBasedAgeItpRecommender : 0.25
            }
        },
        ##########################################################################
        {
            'model_dir_name' : model_name_prefix + '_hybrid_all_recommenders_equal',
            'recommenders' : {
                RandomBasedRecommender : 0.125,
                PopularityBasedRecommender : 0.125,

                ItemBasedCFRecommender : 0.125,
                UserBasedCFRecommender : 0.125,

                ContentBasedRecommender : 0.125,
                UserBasedAgeItpRecommender : 0.125,

                ContentBoostedItemCFRecommender : 0.125,
                ContentBoostedUserCFRecommender : 0.125
            }
        },
        {
            'model_dir_name' : model_name_prefix + '_hybrid_all_recommenders_sub_hybrid_equal',
            'recommenders' : {
                RandomBasedRecommender : 0.143,
                PopularityBasedRecommender : 0.143,

                ItemBasedCFRecommender : 0.143,
                ContentBasedRecommender : 0.142,

                Hybrid_UserBased_CF_AgeItp_Recommender : 0.143,

                ContentBoostedItemCFRecommender : 0.143,
                ContentBoostedUserCFRecommender : 0.143
            }
        },
    ]

    kwargs['age_or_itp'] = 'age'
    kwargs['user_features'] = []
    configs = get_filtered_configs(configs, kwargs['hold_out_strategy'])

    data = pd.read_csv(args.all_data)
    all_items = set(data[item_id_col].unique())
    for config in configs:
        model_dir = os.path.join(current_dir, config['model_dir_name'])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        recommenders = config['recommenders']

        if args.cross_eval and args.kfolds:
            generic_rec_interface.hybrid_kfold_evaluation(recommenders,
                                                          args.kfolds,
                                                          results_dir, model_dir,
                                                          args.train_data, args.test_data,
                                                          user_id_col, item_id_col,
                                                          no_of_recs_to_eval, **kwargs)
            generic_rec_interface.hybrid_kfold_evaluation_using_auto_weights(model_dir,
                                                                             all_items,
                                                                             item_id_col,
                                                                             kwargs['no_of_recs'],
                                                                             no_of_recs_to_eval)
            #get_user_item_meta_info(model_dir, user_id_col, item_id_col, args.all_data, args.learner_pref)
        elif args.train:
            generic_rec_interface.hybrid_train(recommenders,
                                               results_dir, model_dir,
                                               args.train_data, args.test_data,
                                               user_id_col, item_id_col,
                                               **kwargs)
        elif args.eval:
            generic_rec_interface.hybrid_evaluate(recommenders,
                                                  results_dir, model_dir,
                                                  args.train_data, args.test_data,
                                                  user_id_col, item_id_col,
                                                  no_of_recs_to_eval,
                                                  eval_res_file='evaluation_results.json',
                                                  **kwargs)
        elif args.recommend and args.user_id:
            generic_rec_interface.hybrid_recommend(recommenders,
                                                   results_dir, model_dir,
                                                   args.train_data, args.test_data,
                                                   user_id_col, item_id_col,
                                                   args.user_id, **kwargs)
        else:
            generic_rec_interface.hybrid_train_eval_recommend(recommenders,
                                                              results_dir, model_dir,
                                                              args.train_data, args.test_data,
                                                              user_id_col, item_id_col,
                                                              no_of_recs_to_eval, **kwargs)

if __name__ == '__main__':
    main()
