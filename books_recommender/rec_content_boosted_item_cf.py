"""Module for Content Boosted Item Based CF Books Recommender"""
import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rec_content_based import ContentBasedRecommender
from rec_item_based_cf import ItemBasedCFRecommender

from recommender import rec_interface as generic_rec_interface
import rec_interface as books_rec_interface


class ContentBoostedRecommender(ContentBasedRecommender,
                                ItemBasedCFRecommender):
    """Content boosted User based cf recommender system model for Books"""

    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)
        self.threshold_similarity = 0.8
        self.model_file = os.path.join(self.model_dir,
                                       'content_boosted_item_cf_model.pkl')        
    #######################################
    def compute_uim(self):
        """compute psedo uim by using similar items from content based recommendations"""
        print("Computing User Item Matrix...")
        uim_df = super().compute_uim()
        # self.save_uim(uim_df)

        print("Computing Pseudo User Item Matrix...")
        pseudo_uim_df = pd.DataFrame(uim_df)
        for user_id, items in uim_df.iterrows():
            #print(user_id)
            #print(items)
            interacted_items = list(items.iloc[items.nonzero()[0]].index)
            #print(interacted_items)
            user_content_based_recs = super().generate_top_recommendations(user_id,
                                                                           interacted_items,
                                                                           user_dataset='train')
            user_content_based_recs = user_content_based_recs[
                user_content_based_recs['score'] > self.threshold_similarity]
            if not user_content_based_recs.empty:
                #print(user_content_based_recs)
                similar_items = list(user_content_based_recs[self.item_id_col])
                #print(similar_items)
                for similar_item_id in similar_items:
                    pseudo_uim_df.loc[user_id][similar_item_id] = 1
            #pseudo_interacted_items = pseudo_uim_df.loc[user_id]
            #pseudo_interacted_items = list(pseudo_interacted_items.iloc[pseudo_interacted_items.nonzero()[0]].index)
            #print(interacted_items)
            #print(pseudo_interacted_items)
        pseudo_uim = pseudo_uim_df.as_matrix()
        non_zero_count = np.count_nonzero(pseudo_uim)
        count = pseudo_uim.size
        density = non_zero_count / count
        print("Density of Pseudo User Item Matrix : ", density)
        return pseudo_uim_df

    def train(self):
        """train the content boosted user cf recommender system model"""
        ItemBasedCFRecommender.train(self)

    def recommend_items(self, user_id):
        """recommend items for given user_id from test dataset"""
        return ItemBasedCFRecommender.recommend_items(self, user_id)

    def evaluate(self, no_of_recs_to_eval, eval_res_file='evaluation_results.json'):
        """evaluate trained model for different no of ranked recommendations"""
        return ItemBasedCFRecommender.evaluate(self, no_of_recs_to_eval, eval_res_file)

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
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')

    model_dir = os.path.join(current_dir, 'model/content_boosted_item_cf')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    user_id_col = 'learner_id'
    item_id_col = 'book_code'

    no_of_recs = 10
    hold_out_ratio = 0.5
    kwargs = {'no_of_recs': no_of_recs,
              'hold_out_ratio': hold_out_ratio
             }
    no_of_recs_to_eval = [1, 2, 5, 10]
    recommender_obj = ContentBoostedRecommender

    if args.cross_eval and args.kfolds:
        generic_rec_interface.kfold_evaluation(recommender_obj,
                                               args.kfolds,
                                               results_dir, model_dir,
                                               args.train_data, args.test_data,
                                               user_id_col, item_id_col,
                                               no_of_recs_to_eval, **kwargs)
        return
    if args.train:
        generic_rec_interface.train(recommender_obj,
                                    results_dir, model_dir,
                                    args.train_data, args.test_data,
                                    user_id_col, item_id_col,
                                    **kwargs)
    elif args.eval:
        generic_rec_interface.evaluate(recommender_obj,
                                       results_dir, model_dir,
                                       args.train_data, args.test_data,
                                       user_id_col, item_id_col,
                                       no_of_recs_to_eval,
                                       eval_res_file='evaluation_results.json',
                                       **kwargs)
    elif args.recommend and args.user_id:
        # generic_rec_interface.recommend(recommender_obj,
        #                                 results_dir, model_dir,
        #                                 args.train_data, args.test_data,
        #                                 user_id_col, item_id_col,
        #                                 args.user_id, **kwargs)
        # metadata_fields = None
        metadata_fields = ['T_BOOK_NAME', 'T_KEYWORD', 'T_AUTHOR']
        books_rec_interface.recommend(recommender_obj,
                                      results_dir, model_dir,
                                      args.train_data, args.test_data,
                                      user_id_col, item_id_col,
                                      args.user_id, metadata_fields, **kwargs)
    else:
        generic_rec_interface.train_eval_recommend(recommender_obj,
                                                   results_dir, model_dir,
                                                   args.train_data, args.test_data,
                                                   user_id_col, item_id_col,
                                                   no_of_recs_to_eval, **kwargs)

if __name__ == '__main__':
    main()
