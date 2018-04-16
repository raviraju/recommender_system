"""Module for Hybrid of User Based CF with (Age and/or Item Type Preference) Books Recommender"""
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
from recommender.evaluation import PrecisionRecall

class Hybrid_UserBased_CF_AgeItp_Recommender(books_rec_interface.BooksRecommender,
                                             generic_rec_user_based_cf.UserBasedCFRecommender):
    """Hybrid of User Based CF with (Age and/or Item Type Preference) Books Recommender"""

    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)
        self.kwargs = kwargs
    #######################################
    def compute_user_cosine_similarity(self):
        """construct matrix using cosine similarity of user age and item type Preference"""
        #Fetch User Item Matrix
        self.uim_df = self.load_uim()
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

        measures_df_train_test = measures_df[measures_df.index.isin(users)]
        #print(len(measures_df_train_test))
        #print(measures_df_train_test.head())
        
        if self.kwargs['age_or_itp'] == 'itp':
            measures_df_train_test = measures_df_train_test[['audio_close', 'book_close', 'video_close']]
        if self.kwargs['age_or_itp'] == 'age':
            measures_df_train_test = measures_df_train_test[['age']]
        if self.kwargs['age_or_itp'] == 'age_and_itp':
            measures_df_train_test = measures_df_train_test
        #print(measures_df_train_test.head())
        
        users_cosine_similarity = cosine_similarity(measures_df_train_test.as_matrix())
        user_cosine_similarity_df = pd.DataFrame(users_cosine_similarity,
                                                 columns=users,
                                                 index=users)
        #print(user_cosine_similarity_df.head())
        return user_cosine_similarity_df

    def train(self):
        """train the user similarity based recommender system model"""
        super().train()

        # Compute user similarity matrix of size, len(users) X len(users)
        print("Compute user cosine similarity matrix...")
        start_time = default_timer()
        self.user_cosine_similarity_df = self.compute_user_cosine_similarity()
        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))
        #print(self.user_cosine_similarity_df.shape)        
        user_based_itp_model_file = os.path.join(self.model_dir, 'user_based_itp_model.pkl')
        joblib.dump(self.user_cosine_similarity_df, user_based_itp_model_file)
        LOGGER.debug("Saved Model : " + user_based_itp_model_file)
    #######################################
    def __get_similar_users(self, user_id):
        """retrieve similar users for a given user_id"""
        #print(user_id)
        similar_jaccard_users = self.user_similarity_matrix_df[user_id]
        #print('similar_jaccard_users')
        #print(similar_jaccard_users['101653.0'])
        similar_cosine_users = self.user_cosine_similarity_df[user_id]
        #print('similar_cosine_users')
        #print(similar_cosine_users['101653.0'])
        df = pd.DataFrame({'similar_jaccard_users':similar_jaccard_users,
                           'similar_cosine_users':similar_cosine_users})
        df = df[(df['similar_jaccard_users'] > 0) & (df['similar_cosine_users'] > 0)]
        df['combined_similarity'] = df['similar_jaccard_users'] * 0.75 + df['similar_cosine_users'] * 0.25
        df = df[df['combined_similarity'] > 0]
        
        df.sort_values(by=['combined_similarity'], inplace=True, ascending=False)        
        if user_id in df.index:
            df.drop(user_id, inplace=True)                
        
        most_similar_users = df['combined_similarity']
        #print(len(most_similar_users))
        return most_similar_users

    def __generate_top_recommendations(self, user_id, user_interacted_items):
        """Use the cooccurence matrix to make top recommendations"""
        # Calculate a weighted average of the scores in cooccurence matrix for
        # all user items.
        #print(user_id)
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']

        similar_users_weights = self.__get_similar_users(user_id)
        similar_user_ids = similar_users_weights.index
        #print(similar_user_ids)
        #top_10_users = list(similar_users_weights.head(10).index)
        #print(top_10_users)
        #input()
        sub_uim_df = self.uim_df.loc[similar_user_ids]
        weighted_sub_uim_df = sub_uim_df.mul(similar_users_weights, axis='index')
        no_of_similar_users = weighted_sub_uim_df.shape[0]
        if no_of_similar_users != 0:
            item_scores = weighted_sub_uim_df.sum(axis=0) / float(no_of_similar_users)
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
        """recommend items for given user_id from test dataset"""
        super().recommend_items(user_id)
        #pprint(self.items_for_evaluation[user_id])
        self.uim_df = self.load_uim()

        if os.path.exists(self.model_file):
            self.user_similarity_matrix_df = joblib.load(self.model_file)
            #print(self.user_similarity_matrix_df.shape)

            user_based_itp_model_file = os.path.join(self.model_dir, 'user_based_itp_model.pkl')
            self.user_cosine_similarity_df = joblib.load(user_based_itp_model_file)
            #print(self.user_cosine_similarity_df.shape)
            LOGGER.debug("Loaded Trained Model")

            # Use the cooccurence matrix to make recommendations
            start_time = default_timer()
            assume_interacted_items = self.items_for_evaluation[user_id]['assume_interacted_items']
            user_recommendations = self.__generate_top_recommendations(user_id,
                                                                       assume_interacted_items)
            recommended_items = list(user_recommendations[self.item_id_col].values)
            end_time = default_timer()
            print("{:50}    {}".format("Recommendations generated. ",
                                       utilities.convert_sec(end_time - start_time)))
            return user_recommendations
        else:
            print("Trained Model not found !!!. Failed to generate recommendations")
            return None
    #######################################
    def __recommend_items_to_evaluate(self):
        """recommend items for all users from test dataset"""
        for user_id in self.items_for_evaluation:
            assume_interacted_items = self.items_for_evaluation[user_id]['assume_interacted_items']
            user_recommendations = self.__generate_top_recommendations(user_id,
                                                                       assume_interacted_items)

            recommended_items = list(user_recommendations[self.item_id_col].values)
            self.items_for_evaluation[user_id]['items_recommended'] = recommended_items
            
            recommended_items_dict = dict()
            for i, recs in user_recommendations.iterrows():
                item_id = recs[self.item_id_col]
                score = round(recs['score'], 3)
                rank = recs['rank']
                recommended_items_dict[item_id] = {'score' : score, 'rank' : rank}
            self.items_for_evaluation[user_id]['items_recommended_score'] = recommended_items_dict
        return self.items_for_evaluation

    def evaluate(self, no_of_recs_to_eval, eval_res_file='evaluation_results.json'):
        """evaluate trained model for different no of ranked recommendations"""
        super().load_stats()
        super().load_items_for_evaluation()
        self.uim_df = self.load_uim()

        if os.path.exists(self.model_file):
            self.user_similarity_matrix_df = joblib.load(self.model_file)
            #print(self.user_similarity_matrix_df.shape)
            
            user_based_itp_model_file = os.path.join(self.model_dir, 'user_based_itp_model.pkl')
            self.user_cosine_similarity_df = joblib.load(user_based_itp_model_file)
            #print(self.user_cosine_similarity_df.shape)
            LOGGER.debug("Loaded Trained Model")            

            start_time = default_timer()
            #Generate recommendations for the users
            self.items_for_evaluation = self.__recommend_items_to_evaluate()
            self.save_items_for_evaluation()

            precision_recall_intf = PrecisionRecall()
            evaluation_results = precision_recall_intf.compute_precision_recall(
                no_of_recs_to_eval, self.items_for_evaluation, self.items_all)
            end_time = default_timer()
            print("{:50}    {}".format("Evaluation Completed. ",
                                       utilities.convert_sec(end_time - start_time)))

            results_file = os.path.join(self.model_dir, eval_res_file)
            utilities.dump_json_file(evaluation_results, results_file)

            return evaluation_results
        else:
            print("Trained Model not found !!!. Failed to evaluate")
            evaluation_results = {'status' : "Trained Model not found !!!. Failed to evaluate"}

            results_file = os.path.join(self.model_dir, 'evaluation_results.json')
            utilities.dump_json_file(evaluation_results, results_file)

            return evaluation_results
    #######################################
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
        model_dir = os.path.join(current_dir, 'model/user_based_cf_itp')
    elif args.age_or_itp == 'age':
        kwargs['age_or_itp'] = 'age'
        model_dir = os.path.join(current_dir, 'model/user_based_cf_age')
    elif args.age_or_itp == 'age_and_itp':
        kwargs['age_or_itp'] = 'age_and_itp'
        model_dir = os.path.join(current_dir, 'model/user_based_cf_age_itp')
    else:
        print("Invalid arguments, refer --help")
        exit(0)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    recommender_obj = Hybrid_UserBased_CF_AgeItp_Recommender

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
