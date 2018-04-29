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
from books_recommender import rec_item_based_cf as books_rec_item_based_cf
from books_recommender import rec_user_based_cf as books_rec_user_based_cf
from books_recommender import rec_popularity_based as books_rec_popularity_based
from rec_hybrid_user_based_cf_age_itp import Hybrid_UserBased_CF_AgeItp_Recommender
from rec_content_based import ContentBasedRecommender
import rec_interface as books_rec_interface

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
    parser.add_argument("all_data",
                        help="All Data")
    parser.add_argument("learner_pref",
                        help="User Preferences")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    
    user_id_col = 'learner_id'
    item_id_col = 'book_code'

    kwargs = dict()
    kwargs['no_of_recs'] = 150 # max no_of_books read is 144

    # kwargs['hold_out_strategy'] = 'hold_out_ratio'
    # kwargs['hold_out_ratio'] = 0.5

    # kwargs['hold_out_strategy'] = 'assume_first_n'
    # kwargs['first_n'] = 5 #each user has atleast 10 items interacted, so there shall be equal split if no_of_items = 10

    kwargs['hold_out_strategy'] = 'hold_last_n'
    kwargs['last_n'] = 5 #each user has atleast 10 items interacted, so there shall be equal split if no_of_items = 10

    no_of_recs_to_eval = [5, 6, 7, 8, 9, 10]

    configs = [
        {
            'model_dir_name' : 'model/hybrid_item_cf_user_cf',
            'recommenders' : {
                books_rec_item_based_cf.ItemBasedCFRecommender : 0.5,
                books_rec_user_based_cf.UserBasedCFRecommender : 0.5}
        },
        {
            'model_dir_name' : 'model/hybrid_item_cf_content',
            'recommenders' : {
                books_rec_item_based_cf.ItemBasedCFRecommender : 0.5,
                ContentBasedRecommender : 0.5}
        },
        {
            'model_dir_name' : 'model/hybrid_user_cf_content',
            'recommenders' : {
                books_rec_user_based_cf.UserBasedCFRecommender : 0.5,
                ContentBasedRecommender : 0.5}
        },
        {
            'model_dir_name' : 'model/hybrid_item_cf_user_age',
            'recommenders' : {
                books_rec_item_based_cf.ItemBasedCFRecommender : 0.5,
                Hybrid_UserBased_CF_AgeItp_Recommender : 0.5}
        },
        {
            'model_dir_name' : 'model/hybrid_item_cf_content_user_age',
            'recommenders' : {
                books_rec_item_based_cf.ItemBasedCFRecommender : 0.35,
                ContentBasedRecommender : 0.35,
                Hybrid_UserBased_CF_AgeItp_Recommender : 0.3}
        }
    ]
    kwargs['age_or_itp'] = 'age'
    
    for config in configs:
        model_dir = os.path.join(current_dir, config['model_dir_name'])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        recommenders = config['recommenders']

        if args.cross_eval and args.kfolds:
            '''
            get_user_item_meta_info(model_dir, user_id_col, item_id_col, args.all_data, args.learner_pref)
            print("remove this line")
            return
            '''
            generic_rec_interface.hybrid_kfold_evaluation(recommenders,
                                                          args.kfolds,
                                                          results_dir, model_dir,
                                                          args.train_data, args.test_data,
                                                          user_id_col, item_id_col,
                                                          no_of_recs_to_eval, **kwargs)
            get_user_item_meta_info(model_dir, user_id_col, item_id_col, args.all_data, args.learner_pref)
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
