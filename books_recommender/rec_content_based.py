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

        self.item_profile_train_dict = dict()
        self.item_profile_test_dict = dict()
    #######################################
    def derive_stats(self):
        """derive stats"""
        super().derive_stats()

        LOGGER.debug("Train Data       :: Getting Item Profiles")
        for item_id in self.items_train:
            item_profile = books_rec_interface.get_item_profile(
                self.train_data, item_id)
            name_tokens_set, author_set, keywords = item_profile
            profile = {'name_tokens': list(name_tokens_set),
                       'author': list(author_set),
                       'keywords': keywords}
            self.item_profile_train_dict[item_id] = profile

        item_profile_train_file = os.path.join(
            self.model_dir, 'item_profile_train.json')
        utilities.dump_json_file(
            self.item_profile_train_dict, item_profile_train_file)

        LOGGER.debug("Test Data        :: Getting Item Profiles")
        for item_id in self.items_test:
            item_profile = books_rec_interface.get_item_profile(
                self.test_data, item_id)
            name_tokens_set, author_set, keywords = item_profile
            profile = {'name_tokens': list(name_tokens_set),
                       'author': list(author_set),
                       'keywords': keywords}
            self.item_profile_test_dict[item_id] = profile

        item_profile_test_file = os.path.join(
            self.model_dir, 'item_profile_test.json')
        utilities.dump_json_file(
            self.item_profile_test_dict, item_profile_test_file)

    def load_stats(self):
        """load stats"""
        super().load_stats()

        LOGGER.debug("Train Data       :: Loading Item Profiles")
        item_profile_train_file = os.path.join(
            self.model_dir, 'item_profile_train.json')
        self.item_profile_train_dict = utilities.load_json_file(
            item_profile_train_file)

        LOGGER.debug("Test Data        :: Loading Item Profiles")
        item_profile_test_file = os.path.join(
            self.model_dir, 'item_profile_test.json')
        self.item_profile_test_dict = utilities.load_json_file(
            item_profile_test_file)
    #######################################
    def train(self):
        """train the content based recommender system model"""
        self.derive_stats()
        self.get_test_data_for_evaluation()
    #######################################
    def __get_item_profile(self, item_id, dataset='train'):
        """private function, return item profile saved for given item_id"""
        if dataset == "train":
            if item_id in self.item_profile_train_dict:
                return self.item_profile_train_dict[item_id]
            else:
                return {}
        else:  # test
            if item_id in self.item_profile_test_dict:
                return self.item_profile_test_dict[item_id]
            else:
                return {}

    def __get_user_profile(self, user_items, dataset='train'):
        """private function, return user profile by merging item profiles
        for user interacted items"""
        user_profile_name_tokens_set = set()
        user_profile_authors_set = set()
        user_profile_keywords = []

        for item_id in user_items:
            item_profile = self.__get_item_profile(item_id, dataset)
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

    def __get_similarity_score(self, user_profile, item_profile):
        """similarity scores bw user and item profile"""
        name_tokens_similarity = books_rec_interface.get_jaccard_similarity(set(user_profile['name_tokens']),
                                                                            set(item_profile['name_tokens']))
        authors_similarity = books_rec_interface.get_jaccard_similarity(set(user_profile['author']),
                                                                        set(item_profile['author']))
        keywords_similarity = books_rec_interface.get_term_freq_similarity(user_profile['keywords'],
                                                                           item_profile['keywords'])
        '''
        print("\tname : {}, author : {}, keywords : {}, score : {} ".format(name_tokens_similarity,
                                                                            authors_similarity,
                                                                            keywords_similarity))
        '''
        return (name_tokens_similarity, authors_similarity, keywords_similarity)

    def __weighted_avg(self, data_frame, columns_weights_dict):
        """compute weighted average defined by columns_weights_dict"""
        data_frame['sim_score'] = 0.0
        for col_name in columns_weights_dict:
            weighted_col = data_frame[col_name] * \
                columns_weights_dict[col_name]
            data_frame['sim_score'] = data_frame['sim_score'] + weighted_col
            # print(data_frame['sim_score'])
        return data_frame

    def generate_top_recommendations(self, user_id, user_interacted_items, user_dataset='test'):
        """Get all items from train data and recommend them
        which are most similar to user_profile"""
        items_to_recommend = []
        columns = [self.user_id_col, self.item_id_col, 'score', 'rank']

        user_profile = self.__get_user_profile(user_interacted_items,
                                               user_dataset)
        #print("User Profile")
        # print(user_profile)
        # print()
        items_train = self.get_all_items(dataset='train')
        item_scores = []
        for item_id in items_train:
            item_profile = self.__get_item_profile(item_id, dataset='train')
            #print("\n\t" + item_id)
            # print(item_profile)
            similarity_scores = self.__get_similarity_score(
                user_profile, item_profile)
            (name_tokens_similarity, authors_similarity,
             keywords_similarity) = similarity_scores
            item_scores.append({self.item_id_col: item_id,
                                'name_tokens_similarity': name_tokens_similarity,
                                'authors_similarity': authors_similarity,
                                'keywords_similarity': keywords_similarity})
        item_scores_df = pd.DataFrame(item_scores)

        # print("scaling...")
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
        # print("weighted_avg...")
        item_scores_df = self.__weighted_avg(
            item_scores_df, columns_weights_dict)

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
        for user_id in self.items_for_evaluation:
            assume_interacted_items = self.items_for_evaluation[
                user_id]['assume_interacted_items']
            user_recommendations = self.generate_top_recommendations(user_id,
                                                                       assume_interacted_items)

            recommended_items = list(user_recommendations[self.item_id_col].values)
            self.items_for_evaluation[user_id][
                'items_recommended'] = recommended_items
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
            no_of_recs_to_eval, self.items_for_evaluation)
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

    no_of_recs = 10
    hold_out_ratio = 0.5
    kwargs = {'no_of_recs': no_of_recs,
              'hold_out_ratio': hold_out_ratio
             }
    no_of_recs_to_eval = [1, 2, 5, 10]
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
