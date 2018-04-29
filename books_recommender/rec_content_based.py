"""Module for Content Based Books Recommender"""
import os
import sys
import argparse
import logging
from timeit import default_timer

from sklearn.preprocessing import minmax_scale

import pandas as pd

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import utilities
from recommender import rec_interface as generic_rec_interface
from recommender.evaluation import PrecisionRecall
import rec_interface as books_rec_interface

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

        self.item_profile_all_dict = dict()
        self.user_profiles_dict = dict()
    #######################################
    def derive_stats(self):
        """derive stats"""
        super().derive_stats()

        LOGGER.debug("All Data       :: Getting Item Profiles")
        #fetch items from train dataframe
        for item_id in self.items_train:
            item_profile = books_rec_interface.get_item_profile(self.train_data, item_id)
            self.item_profile_all_dict[item_id] = item_profile
        #fetch remaining items (only in test) from test dataframe
        missing_items_from_test = set(self.items_all) - set(self.items_train)
        for item_id in missing_items_from_test:
            item_profile = books_rec_interface.get_item_profile(self.test_data, item_id)
            self.item_profile_all_dict[item_id] = item_profile

        item_profile_all_file = os.path.join(self.model_dir, 'item_profile_all.json')
        utilities.dump_json_file(self.item_profile_all_dict, item_profile_all_file)


    def load_stats(self):
        """load stats"""
        super().load_stats()
        
        LOGGER.debug("All Data       :: Loading Item Profiles")
        item_profile_all_file = os.path.join(self.model_dir, 'item_profile_all.json')
        self.item_profile_all_dict = utilities.load_json_file(item_profile_all_file)

    #######################################
    def train(self):
        """train the content based recommender system model"""
        self.derive_stats()
        self.get_test_data_for_evaluation()
    #######################################
    def __get_item_profile(self, item_id):
        """private function, return item profile saved for given item_id"""
        if item_id in self.item_profile_all_dict:
            return self.item_profile_all_dict[item_id]
        else:
            return {}

    def __get_user_profile(self, user_items):
        """private function, return user profile by merging item profiles
        for user interacted items"""
        user_profile_name_tokens_set = set()
        user_profile_authors_set = set()
        user_profile_keywords = []

        for item_id in user_items:
            item_profile = self.__get_item_profile(item_id)
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

        profile = {'name_tokens': list(user_profile_name_tokens_set),
                   'author': list(user_profile_authors_set),
                   'keywords': user_profile_keywords
                  }
        return profile

    def __weighted_avg(self, data_frame, columns_weights_dict):
        """compute weighted average defined by columns_weights_dict"""
        data_frame['sim_score'] = 0.0
        for col_name in columns_weights_dict:
            weighted_col = data_frame[col_name] * columns_weights_dict[col_name]
            data_frame['sim_score'] = data_frame['sim_score'] + weighted_col
            # print(data_frame['sim_score'])
        return data_frame

    def generate_top_recommendations(self, user_id, user_interacted_items):
        """Get all items from train data and recommend them
        which are most similar to user_profile"""
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col,
                   'name_tokens_similarity', 'authors_similarity', 'keywords_similarity',
                   'score', 'rank']
        user_profile = self.__get_user_profile(user_interacted_items)
        self.user_profiles_dict[user_id] = user_profile
        # print("User Profile")
        # print(user_profile)

        items_all = self.get_all_items(dataset='all')
        item_scores = []
        for item_id in items_all:
            item_profile = self.__get_item_profile(item_id)
            # print("\n\t" + item_id)
            # print(item_profile)
            similarity_scores = books_rec_interface.get_profile_similarity_score(user_profile, item_profile)
            # print(similarity_scores)
            (name_tokens_similarity, authors_similarity, keywords_similarity) = similarity_scores
            item_scores.append({self.item_id_col: item_id,
                                'name_tokens_similarity': name_tokens_similarity,
                                'authors_similarity': authors_similarity,
                                'keywords_similarity': keywords_similarity})
        item_scores_df = pd.DataFrame(item_scores)

        # print("scaling...")
        '''
        values = item_scores_df[['name_tokens_similarity']].as_matrix()
        scaled_values = minmax_scale(values)
        item_scores_df['name_tokens_similarity_scaled'] = scaled_values

        values = item_scores_df[['authors_similarity']].as_matrix()
        scaled_values = minmax_scale(values)
        item_scores_df['authors_similarity_scaled'] = scaled_values

        values = item_scores_df[['keywords_similarity']].as_matrix()
        scaled_values = minmax_scale(values)
        item_scores_df['keywords_similarity_scaled'] = scaled_values

        columns_weights_dict = {'name_tokens_similarity_scaled': 0.25,
                                'authors_similarity_scaled': 0.25,
                                'keywords_similarity_scaled': 0.5
                               }
        '''
        columns_weights_dict = {'name_tokens_similarity': 0.25,
                                'authors_similarity': 0.25,
                                'keywords_similarity': 0.5
                               }
        # print("weighted_avg...")
        item_scores_df = self.__weighted_avg(item_scores_df, columns_weights_dict)

        # print("sorting...")
        # Sort the items based upon similarity scores
        item_scores_df = item_scores_df.sort_values(['sim_score', self.item_id_col],
                                                    ascending=[0, 1])
        item_scores_df.reset_index(drop=True, inplace=True)
        #print(item_scores_df[item_scores_df['sim_score'] > 0])

        # print(item_scores_df.head())
        rank = 1
        for _, item_score in item_scores_df.iterrows():
            item_id = item_score[self.item_id_col]
            if item_id in user_interacted_items:  # to avoid items which user has already aware
                continue
            if rank > self.no_of_recs:  # limit no of recommendations
                break
            item_dict = {
                self.user_id_col: user_id,
                self.item_id_col: item_id,
                'name_tokens_similarity': item_score['name_tokens_similarity'],
                'authors_similarity': item_score['authors_similarity'],
                'keywords_similarity': item_score['keywords_similarity'],
                'score': item_score['sim_score'],
                'rank': rank
            }
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
        self.load_stats()
        self.load_items_for_evaluation()

        start_time = default_timer()
        assume_interacted_items = self.items_for_evaluation[user_id]['assume_interacted_items']
        user_recommendations = self.generate_top_recommendations(user_id,
                                                                 assume_interacted_items)

        #recommended_items = list(user_recommendations[self.item_id_col].values)
        end_time = default_timer()
        print("{:50}    {}".format("Recommendations generated. ",
                                   utilities.convert_sec(end_time - start_time)))
        return user_recommendations
    #######################################
    def __recommend_items_to_evaluate(self):
        """recommend items for all users from test dataset"""
        all_user_recommendations_list = []
        for user_id in self.items_for_evaluation:
            assume_interacted_items = self.items_for_evaluation[
                user_id]['assume_interacted_items']
            user_recommendations = self.generate_top_recommendations(user_id,
                                                                     assume_interacted_items)
            all_user_recommendations_list.append(user_recommendations)
            recommended_items = list(user_recommendations[self.item_id_col].values)
            self.items_for_evaluation[user_id]['items_recommended'] = recommended_items
            
            recommended_items_dict = dict()
            for i, recs in user_recommendations.iterrows():
                item_id = recs[self.item_id_col]
                score = round(recs['score'], 3)
                rank = recs['rank']
                recommended_items_dict[item_id] = {'score' : score, 'rank' : rank}
            self.items_for_evaluation[user_id]['items_recommended_score'] = recommended_items_dict

            items_interacted_set = set(self.items_for_evaluation[user_id]['items_interacted'])
            items_recommended_set = set(recommended_items)
            correct_recommendations = items_interacted_set & items_recommended_set
            no_of_correct_recommendations = len(correct_recommendations)
            self.items_for_evaluation[user_id]['no_of_correct_recommendations'] = no_of_correct_recommendations
            self.items_for_evaluation[user_id]['correct_recommendations'] = list(correct_recommendations)

        all_user_recommendations_df = pd.concat(all_user_recommendations_list, axis=0)
        #print(len(all_user_recommendations_df[self.user_id_col].unique()))
        all_user_recommendations_file = os.path.join(self.model_dir, 'all_user_recommendation_details.csv')
        all_user_recommendations_df.to_csv(all_user_recommendations_file, index=False)

        user_profiles_file = os.path.join(self.model_dir, 'user_profiles.json')
        utilities.dump_json_file(self.user_profiles_dict, user_profiles_file)

        return self.items_for_evaluation

    def evaluate(self, no_of_recs_to_eval, eval_res_file='evaluation_results.json'):
        """evaluate trained model for different no of ranked recommendations"""
        self.load_stats()
        self.load_items_for_evaluation()

        start_time = default_timer()
        # Generate recommendations for the users
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
    #######################################

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

    model_dir = os.path.join(current_dir, 'model/content_based')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

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
    recommender_obj = ContentBasedRecommender

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
