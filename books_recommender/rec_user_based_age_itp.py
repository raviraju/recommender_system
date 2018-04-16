"""Module for User Based Age and Item Type Preference Books Recommender"""
import os
import sys
import argparse
import logging
from timeit import default_timer

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender import rec_interface as generic_rec_interface
from recommender import rec_user_based_cf as generic_rec_user_based_cf
import rec_interface as books_rec_interface

class UserBasedAgeItpRecommender(books_rec_interface.BooksRecommender,
                                 generic_rec_user_based_cf.UserBasedCFRecommender):
    """User based age and item type preference recommender system model for Books"""

    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)
        self.kwargs = kwargs
    #######################################
    def compute_user_similarity(self):
        """construct matrix using cosine similarity of user age and item type Preference"""
        #Compute User Item Matrix
        print()
        print("Combining train_data with test_data_for_evaluation...")
        train_test_data = self.train_data.append(self.test_data_for_evaluation,
                                                 ignore_index=True)
        print("Computing User Item Matrix...")
        self.uim_df = pd.get_dummies(train_test_data[self.item_id_col])\
                        .groupby(train_test_data[self.user_id_col])\
                        .apply(max)
        self.save_uim(self.uim_df)

        #stats
        print()
        items = [str(col) for col in self.uim_df.columns]
        no_of_items = len(items)
        users = [str(idx) for idx in self.uim_df.index]
        no_of_users = len(users)
        print("No of Items : ", no_of_items)
        print("No of Users : ", no_of_users)

        #Compute User-User Similarity Matrix with Cosine Similarity of user features
        print()
        print("""Computing User-User Similarity Matrix with Cosine Similarity of age, item type preferences...""")
        measures_df = pd.read_csv('preprocessed_metadata/learner_measures.csv')
        measures_df.set_index('learner_id', inplace=True)
        #print(measures_df.head())

        '''
        audio_users = measures_df[(measures_df['audio_close'] == 1.0) & \
                                  (measures_df['book_close'] == 0)    & \
                                  (measures_df['video_close'] == 0)]\
                                  [['audio_close', 'book_close', 'video_close']]
        book_users = measures_df[(measures_df['book_close'] == 1.0) & \
                                 (measures_df['audio_close'] == 0)  & \
                                 (measures_df['video_close'] == 0)]\
                                 [['audio_close', 'book_close', 'video_close']]
        video_users = measures_df[(measures_df['video_close'] == 1.0) & \
                                  (measures_df['book_close'] == 0)    & \
                                  (measures_df['audio_close'] == 0)]\
                                  [['audio_close', 'book_close', 'video_close']]

        print("len(all_users) : ", len(measures_df))
        print("len(audio_users) : ", len(audio_users))
        print("len(video_users) : ", len(video_users))
        print("len(book_users) : ", len(book_users))
        '''

        measures_df_train_test = measures_df[measures_df.index.isin(users)]
        #print(measures_df_train_test.head())
        if self.kwargs['age_or_itp'] == 'itp':
            measures_df_train_test = measures_df_train_test[['audio_close', 'book_close', 'video_close']]
        if self.kwargs['age_or_itp'] == 'age':
            measures_df_train_test = measures_df_train_test[['age']]
        if self.kwargs['age_or_itp'] == 'age_and_itp':
            measures_df_train_test = measures_df_train_test
        #print(measures_df_train_test.head())
        #input()

        users_cosine_similarity = cosine_similarity(measures_df_train_test.as_matrix())
        users_cosine_similarity_df = pd.DataFrame(users_cosine_similarity,
                                                  columns=users,
                                                  index=users)
        #print(users_cosine_similarity_df.head())
        return users_cosine_similarity_df

def main():
    """User based recommender interface"""
    parser = argparse.ArgumentParser(description="User Based Recommender")
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
    parser.add_argument("--age_or_itp",
                        help="select age or item_type preference")    
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
    
    if args.age_or_itp == 'itp':
        kwargs['age_or_itp'] = 'itp'
        model_dir = os.path.join(current_dir, 'model/user_based_itp')
    if args.age_or_itp == 'age':
        kwargs['age_or_itp'] = 'age'
        model_dir = os.path.join(current_dir, 'model/user_based_age')
    if args.age_or_itp == 'age_and_itp':
        kwargs['age_or_itp'] = 'age_and_itp'
        model_dir = os.path.join(current_dir, 'model/user_based_age_itp')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    recommender_obj = UserBasedAgeItpRecommender

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
