"""Module for Content Based Articles Recommender"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                #binary_rec     #articles_recommender
import argparse
from timeit import default_timer
from pprint import pprint

import joblib
import pandas as pd

import logging
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

from lib import utilities
import rec_interface as articles_rec_interface
from recommender import rec_interface as generic_rec_interface
from recommender.evaluation import PrecisionRecall

import spacy
spacy_lemmatizer = spacy.load('en', disable=['parser', 'ner'])
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pandas as pd
pd.set_option('display.max_colwidth', 150)

class ContentBasedRecommender(articles_rec_interface.ArticlesRecommender):
    """Content based recommender system model for Articles"""
    def __init__(self, results_dir, model_dir,
                 train_data, test_data, meta_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)
        self.model_file = os.path.join(self.model_dir,
                                        'content_based_model.pkl')
        self.meta_data = meta_data
        self.item_profiles_dict = dict()
        self.user_profiles_dict = dict()

    #######################################
    def derive_stats(self):
        """derive stats"""
        super().derive_stats()

    def load_stats(self):
        """load stats"""
        super().load_stats()
        
        LOGGER.debug("All Data       :: Loading Item Profiles")
        item_profiles_file = os.path.join(self.model_dir, 'item_profiles.json')
        self.item_profiles_dict = utilities.load_json_file(item_profiles_file)
    #######################################
    def __lemmatize_spacy(self, text):
        """Apply Lemmatization using Spacy Lemmatization"""
        lemma_text = None
        if isinstance(text, str):
            lemma_text = ""
    
            # Parse the sentence using the loaded 'en' model object `nlp`
            doc = spacy_lemmatizer(text)
    
            # Extract the lemma for each token
            lemmas = []
            for token in doc:
                lemma = token.lemma_
                if lemma == '-PRON-': #https://spacy.io/api/annotation#lemmatization
                    lemma = token.text
                if lemma.isalpha():
                    lemmas.append(lemma)
            lemma_text = " ".join(lemmas)
            if len(lemma_text) == 0:
                return None
        return lemma_text

    def __run_topic_modelling(self):
        no_of_topics = 5
        no_of_top_words_per_topic = 30
        
        self.meta_data['text_content'] = self.meta_data[['title', 'text']].apply(lambda x: '. '.join(x), axis=1)            
        meta_data_text_df = self.meta_data['text_content'].drop_duplicates()\
                                                          .reset_index()\
                                                          .rename(columns={'index' : 'text_content_id'})
        self.meta_data = self.meta_data.merge(meta_data_text_df)
        # pprint(self.meta_data.iloc[0].to_dict)
        # print(meta_data_text_df.head())
        # print(self.meta_data.shape, meta_data_text_df.shape)

        print("Lemmatize Text...")
        meta_data_text_df.loc[:, 'processed_text'] = meta_data_text_df['text_content'].apply(self.__lemmatize_spacy)
        # print(meta_data_text_df.head())
        # print(meta_data_text_df.shape)
        # input()

        print("Infer Topics...")
        tf_vectorizer = CountVectorizer(strip_accents = 'ascii',
                                        stop_words = 'english',
                                        lowercase = True,
                                        analyzer = 'word',
                                        token_pattern = r'\b[a-zA-Z]{5,}\b',
                                        ngram_range = (1, 1),
                                        max_df = 0.8, min_df = 1)        
        doc_term_freq_matrix = tf_vectorizer.fit_transform(meta_data_text_df['processed_text'])
        # print(doc_term_freq_matrix.shape)
        feature_names = tf_vectorizer.get_feature_names()

        lda = LatentDirichletAllocation(n_components=no_of_topics, n_jobs=-1, random_state=0)
        doc_topics_df = pd.DataFrame(lda.fit_transform(doc_term_freq_matrix), 
                                     columns=['topic_' + str(i) for i in range(lda.n_components)])
        def get_topic_names(x):
            """ignore topics with less 0.1"""
            return x[x>0.1].to_dict()
        doc_topics_df.loc[:, 'mostly_about'] = doc_topics_df.apply(get_topic_names, axis=1)
        doc_topics_df.loc[:, 'no_of_topics'] = doc_topics_df['mostly_about'].apply(lambda x: len(x))
        # print(doc_topics_df.iloc[0].to_dict())
        # print(doc_topics_df.shape)
        # input()
        
        doc_text_topics_df = doc_topics_df.join(meta_data_text_df, how='inner')
        # print(doc_text_topics_df.iloc[0].to_dict())
        # print(doc_text_topics_df.shape)
        # input()
        
        topic_df = pd.DataFrame(lda.components_, columns=feature_names)
        topic_top_words_dict = dict()
        for i, topic_words in topic_df.iterrows():
            topic_id_str = 'topic_' + str(i)        
            topic_words_dict = topic_words.sort_values(ascending=False).head(no_of_top_words_per_topic).to_dict()
            # topic_words_str = ', '.join(['#'+str(word) for word in topic_words_dict.keys()])
            # print(topic_id_str + " : " + topic_words_str)
            topic_top_words_dict[topic_id_str] = dict()
            # topic_top_words_dict[topic_id_str]['top_words_str'] = topic_words_str
            topic_top_words_dict[topic_id_str]['top_words'] = topic_words_dict
        # pprint(topic_top_words_dict)
        # input()

        return tf_vectorizer, lda, doc_text_topics_df, topic_top_words_dict

    def __generate_item_profile(self, item_id, doc_topics_df, topic_top_words_dict):
        item_meta_df = self.meta_data[self.meta_data[self.item_id_col] == item_id]
        item_meta_df = item_meta_df[[self.item_id_col, 'url', 'title', 'text', 'text_content_id']]
        item_meta_df.drop_duplicates(inplace=True)
        item_profile_dict = dict()
        item_profile_dict['url'] = item_meta_df['url'].values[0]
        item_profile_dict['title'] = item_meta_df['title'].values[0]
        item_profile_dict['text'] = item_meta_df['text'].values[0]

        text_content_id = item_meta_df['text_content_id'].values[0]
        item_profile_dict['text_content_id'] = int(text_content_id)

        item_topics_df = doc_topics_df[doc_topics_df['text_content_id'] == text_content_id]
        mostly_about_topics = item_topics_df['mostly_about'].values[0]
        item_profile_dict['topics'] = dict()
        all_topics_top_words = []
        for topic_id_str in mostly_about_topics:
            item_profile_dict['topics'][topic_id_str] = dict()
            item_profile_dict['topics'][topic_id_str]['topic_conf'] = mostly_about_topics[topic_id_str]
            item_profile_dict['topics'][topic_id_str]['top_words'] = dict(topic_top_words_dict[topic_id_str]['top_words'])
            all_topics_top_words.extend(list(item_profile_dict['topics'][topic_id_str]['top_words'].keys()))
        item_profile_dict['all_topics_top_words'] = all_topics_top_words
        return item_profile_dict

    def train(self):
        """train the content based recommender system model"""
        super().train()

        print()
        print("*"*80)
        print("\tContent Based : Recommending Similar items ...")
        print("*"*80)
        start_time = default_timer()

        vectorizer, lda, doc_topics_df, topic_top_words_dict = self.__run_topic_modelling()

        LOGGER.debug("All Data       :: Getting Item Profiles")        
        for item_id in self.items_all:
            item_profile = self.__generate_item_profile(item_id, doc_topics_df, topic_top_words_dict)
            # pprint(item_profile)
            self.item_profiles_dict[item_id] = item_profile
        item_profiles_file = os.path.join(self.model_dir, 'item_profiles.json')
        utilities.dump_json_file(self.item_profiles_dict, item_profiles_file)

        end_time = default_timer()
        print("{:50}    {}".format("Completed. ",
                                   utilities.convert_sec(end_time - start_time)))
        trained_models = dict()
        trained_models['vectorizer'] = vectorizer
        trained_models['lda'] = lda
        joblib.dump(trained_models, self.model_file)
        LOGGER.debug("Saved Model")
    #######################################
    def __get_item_profile(self, item_id):
        if item_id in self.item_profiles_dict:
            return self.item_profiles_dict[item_id]
        else:
            return None

    def __get_user_profile(self, user_items):
        """return user profile by merging item profiles for user interacted items"""
        user_urls = []
        user_titles = []
        # user_texts = []
        # user_topic_words = []
        user_all_topics_top_words = []
        for item_id in user_items:
            item_profile_dict = self.__get_item_profile(item_id)
            if item_profile_dict is not None:
                item_url = item_profile_dict['url']
                user_urls.append(item_url)
                
                item_title = item_profile_dict['title']
                user_titles.append(item_title)
                
                # item_text = item_profile_dict['text']
                # user_texts.append(item_text)

                # item_topics = item_profile_dict['topics']
                # for topic_id_str in item_topics:
                #     top_words = list(item_topics[topic_id_str]['top_words'].keys())
                #     user_topic_words.extend(top_words)
                
                all_topics_top_words = item_profile_dict['all_topics_top_words']
                user_all_topics_top_words.extend(all_topics_top_words)
        user_profie_dict = dict()
        user_profie_dict['urls'] = user_urls
        user_profie_dict['titles'] = user_titles
        # user_profie_dict['texts'] = user_texts
        # user_profie_dict['topic_words'] = user_topic_words
        user_profie_dict['all_topics_top_words'] = user_all_topics_top_words
        return user_profie_dict          

    def __get_subset_similarity(self, child_set, parent_set):
        """subset similarity"""
        return int(child_set.issubset(parent_set))

    def __get_profile_similarity_score(self, user_profile, item_profile):
        """similarity scores bw user and item profile"""
        topics_subset_similarity = self.__get_subset_similarity(set(item_profile['all_topics_top_words']),
                                                         set(user_profile['all_topics_top_words']))
        similarity_scores_dict = dict()
        similarity_scores_dict['topics_subset_similarity'] = topics_subset_similarity

        return similarity_scores_dict

    def __weighted_avg(self, item_scores_df, columns_weights_dict):
        """compute weighted average defined by columns_weights_dict"""
        item_scores_df['sim_score'] = 0.0
        for col_name in columns_weights_dict:
            weighted_col = item_scores_df[col_name] * columns_weights_dict[col_name]
            item_scores_df['sim_score'] = item_scores_df['sim_score'] + weighted_col
        return item_scores_df

    def __generate_top_recommendations(self, user_id, known_interacted_items):
        items_to_recommend = []

        user_profile_dict = self.__get_user_profile(known_interacted_items)
        # pprint(user_profile_dict)

        item_scores = []
        items_all = self.get_all_items(dataset='all')
        for item_id in items_all:
            item_profile_dict = self.__get_item_profile(item_id)
            # print("\n\t" + item_id)
            # print(item_profile)
            similarity_scores = self.__get_profile_similarity_score(user_profile_dict, 
                                                                    item_profile_dict)
            item_scores.append({self.item_id_col: item_id,
                                'topics_subset_similarity': similarity_scores['topics_subset_similarity']
                               })
        item_scores_df = pd.DataFrame(item_scores)
        # print(item_scores_df.head())
        # print(item_scores_df['topics_subset_similarity'].value_counts())
        # input()

        columns_weights_dict = dict()
        columns_weights_dict['topics_subset_similarity'] = 1

        # print("weighted_avg...")
        item_scores_df = self.__weighted_avg(item_scores_df, columns_weights_dict)

        # print("sorting...")
        # Sort the items based upon similarity scores
        item_scores_df = item_scores_df.sort_values(['sim_score', self.item_id_col],
                                                    ascending=[0, 1])
        item_scores_df.reset_index(drop=True, inplace=True)
        #print(item_scores_df[item_scores_df['sim_score'] > 0])
        
        rank = 1
        for _, item_score in item_scores_df.iterrows():
            item_id = item_score[self.item_id_col]

            if not self.allow_recommending_known_items and item_id in known_interacted_items:#to avoid items which user has already aware
                continue            
            if rank > self.no_of_recs:  # limit no of recommendations
                break
            item_dict = {
                self.item_id_col: item_id,
                'score': round(item_score['sim_score'], 3),
                'rank': rank
            }
            items_to_recommend.append(item_dict)
            rank += 1
        if len(items_to_recommend) > 0:
            items_to_recommend_df = pd.DataFrame(items_to_recommend)
        else:
            items_to_recommend_df = pd.DataFrame(columns = [self.item_id_col, 'score', 'rank'])
        return items_to_recommend_df
    
    def recommend_items(self, user_id):
        """recommend items for given user_id from test dataset"""
        super().recommend_items(user_id)

        if os.path.exists(self.model_file):
            # trained_models = joblib.load(self.model_file)
            # LOGGER.debug("Loaded Trained Model")
            start_time = default_timer()
            known_interacted_items = self.items_for_evaluation[user_id]['known_interacted_items']            
            items_to_recommend_df = self.__generate_top_recommendations(user_id, known_interacted_items)
            end_time = default_timer()
            print("{:50}    {}".format("Recommendations generated. ",
                                       utilities.convert_sec(end_time - start_time)))
            return items_to_recommend_df
        else:
            print("Trained Model not found !!!. Failed to generate recommendations")
            return None
    #######################################
    def __recommend_items_to_evaluate(self):
        """recommend items for all users from test dataset"""
        for user_id in self.items_for_evaluation:
            known_interacted_items = self.items_for_evaluation[user_id]['known_interacted_items']
            items_to_recommend_df = self.__generate_top_recommendations(user_id, known_interacted_items)
            recommended_items_dict = items_to_recommend_df.set_index(self.item_id_col).to_dict('index')            

            self.items_for_evaluation[user_id]['items_recommended'] = list(recommended_items_dict.keys())
            self.items_for_evaluation[user_id]['items_recommended_score'] = recommended_items_dict

            items_to_be_interacted_set = set(self.items_for_evaluation[user_id]['items_to_be_interacted'])
            items_recommended_set = set(self.items_for_evaluation[user_id]['items_recommended'])
            correct_recommendations = items_to_be_interacted_set.intersection(items_recommended_set)
            no_of_correct_recommendations = len(correct_recommendations)
            self.items_for_evaluation[user_id]['no_of_correct_recommendations'] = no_of_correct_recommendations
            self.items_for_evaluation[user_id]['correct_recommendations'] = list(correct_recommendations)
        return self.items_for_evaluation

    def evaluate(self, no_of_recs_to_eval, eval_res_file='evaluation_results.json'):
        """Evaluate trained model for different no of ranked recommendations"""
        super().evaluate(no_of_recs_to_eval, eval_res_file)

        if os.path.exists(self.model_file):
            # trained_models = joblib.load(self.model_file)
            # LOGGER.debug("Loaded Trained Model")

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

            results_file = os.path.join(self.model_dir, eval_res_file)
            utilities.dump_json_file(evaluation_results, results_file)

            return evaluation_results
    #######################################

def load_data(train_data_file, test_data_file, meta_data_file, item_id_col):
    print("Loading Train Data...")
    train_data = generic_rec_interface.load_data(train_data_file)
    if train_data is None:
        exit(-1)
    print("Loading Test Data...")
    test_data = generic_rec_interface.load_data(test_data_file)
    if test_data is None:
        exit(-1)
    print("Loading Meta Data...")
    meta_data = generic_rec_interface.load_data(meta_data_file)
    if meta_data is None:
        exit(-1)
    meta_data = meta_data[meta_data['eventType'] == 'CONTENT SHARED']
    meta_data = meta_data[meta_data['lang'] == 'en']
    meta_data = meta_data[[item_id_col, 'url', 'title', 'text']]
    return train_data, test_data, meta_data

def train(recommender_obj,
          results_dir, model_dir,
          train_data_file, test_data_file, meta_data_file,
          user_id_col, item_id_col,
          **kwargs):
    """train recommender"""
    train_data, test_data, meta_data = load_data(train_data_file, 
                                                 test_data_file, 
                                                 meta_data_file, item_id_col)
    recommender = recommender_obj(results_dir, model_dir,
                                  train_data, test_data, meta_data,
                                  user_id_col, item_id_col,
                                  **kwargs)
    recommender.train()
    print('*' * 80)

def recommend(recommender_obj,
              results_dir, model_dir,
              train_data_file, test_data_file, meta_data_file,
              user_id_col, item_id_col,
              user_id, **kwargs):
    """recommend items for user"""
    train_data, test_data, meta_data = load_data(train_data_file, 
                                                 test_data_file, 
                                                 meta_data_file, item_id_col)
    recommender = recommender_obj(results_dir, model_dir,
                                  train_data, test_data, meta_data,
                                  user_id_col, item_id_col,
                                  **kwargs)
    eval_items_file = os.path.join(model_dir, 'items_for_evaluation.json')
    eval_items = utilities.load_json_file(eval_items_file)
    if user_id in eval_items:
        items_interacted_in_train = eval_items[user_id]['items_interacted_in_train']
        assume_interacted_items = eval_items[user_id]['assume_interacted_items']
        items_to_be_interacted = eval_items[user_id]['items_to_be_interacted']

        print("\nTrain Item interactions for a user with user_id   : {}".format(user_id))        
        if meta_data is not None:
            cols = [item_id_col]
            items_meta_data = meta_data[meta_data[item_id_col].isin(items_interacted_in_train)]
            if 'meta_data_fields' in kwargs:
                meta_data_fields = kwargs['meta_data_fields']
                cols.extend(meta_data_fields)
            print(items_meta_data[cols])
        else:
            for item in items_interacted_in_train:
                print(item)

        print("\nAssumed Item interactions for a user with user_id : {}".format(user_id))
        if meta_data is not None:
            cols = [item_id_col]
            items_meta_data = meta_data[meta_data[item_id_col].isin(assume_interacted_items)]
            if 'meta_data_fields' in kwargs:
                meta_data_fields = kwargs['meta_data_fields']
                cols.extend(meta_data_fields)
            print(items_meta_data[cols])
        else:
            for item in assume_interacted_items:
                print(item)

        print()
        print("\nItems to be interacted for a user with user_id    : {}".format(user_id))
        if meta_data is not None:
            cols = [item_id_col]
            items_meta_data = meta_data[meta_data[item_id_col].isin(items_to_be_interacted)]
            if 'meta_data_fields' in kwargs:
                meta_data_fields = kwargs['meta_data_fields']
                cols.extend(meta_data_fields)
            print(items_meta_data[cols])
        else:
            for item in items_to_be_interacted:
                print(item)

        print()
        print("\nTop {} Items recommended for a user with user_id  : {}".format(recommender.no_of_recs, user_id))
        items_to_recommend_df = recommender.recommend_items(user_id)
        if items_to_recommend_df is not None:
            recommended_items = list(items_to_recommend_df[item_id_col].values)
          
            if meta_data is not None and 'meta_data_fields' in kwargs:
                cols = [item_id_col]
                cols.extend(kwargs['meta_data_fields'])
                items_to_recommend_df = items_to_recommend_df.merge(meta_data[cols], how='inner')
                pprint(items_to_recommend_df)#.to_dict(orient='index'))
            else:                
                for item in recommended_items:
                    print(item)

            print()
            print("\nItems correctly recommended for a user with user_id  : {}".format(user_id))
            correct_recommendations = set(items_to_be_interacted).intersection(set(recommended_items))
            if meta_data is not None and 'meta_data_fields' in kwargs:
                correct_items_to_recommend_df = items_to_recommend_df[items_to_recommend_df[item_id_col].isin(correct_recommendations)]
                print(correct_items_to_recommend_df)
            else:
                for item in correct_recommendations:
                    print(item)
        else:
            print("No items to recommend")
        print('*' * 80)
    else:
        print("""Cannot generate recommendations as either items assumed to be interacted or items held out are None""")

def evaluate(recommender_obj,
             results_dir, model_dir,
             train_data_file, test_data_file, meta_data_file,
             user_id_col, item_id_col,
             no_of_recs_to_eval,
             eval_res_file, **kwargs):
    """evaluate recommender"""
    train_data, test_data, meta_data = load_data(train_data_file, 
                                                 test_data_file, 
                                                 meta_data_file, item_id_col)
    recommender = recommender_obj(results_dir, model_dir,
                                  train_data, test_data, meta_data,
                                  user_id_col, item_id_col,
                                  **kwargs)
    evaluation_results = recommender.evaluate(no_of_recs_to_eval,
                                              eval_res_file)
    pprint(evaluation_results)
    print('*' * 80)
    return evaluation_results

def train_eval_recommend(recommender_obj,
                         results_dir, model_dir,
                         train_data_file, test_data_file, meta_data_file,
                         user_id_col, item_id_col,
                         no_of_recs_to_eval,
                         **kwargs):
    """train, evaluate and recommend"""
    train_data, test_data, meta_data = load_data(train_data_file, 
                                                 test_data_file, 
                                                 meta_data_file, item_id_col)
    recommender = recommender_obj(results_dir, model_dir,
                                  train_data, test_data, meta_data,
                                  user_id_col, item_id_col,
                                  **kwargs)
    print("Training Recommender...")
    recommender.train()
    print('*' * 80)

    print("Evaluating Recommender System")
    evaluation_results = recommender.evaluate(no_of_recs_to_eval)
    pprint(evaluation_results)
    print('*' * 80)

    print("One of the Best Recommendations")
    items_for_evaluation_file = os.path.join(model_dir, 'items_for_evaluation.json')
    items_for_evaluation = utilities.load_json_file(items_for_evaluation_file)
    users = list(items_for_evaluation.keys())

    best_user_id = users[0]
    max_no_of_correct_recommendations = 0
    for user_id in items_for_evaluation:
        no_of_correct_recommendations = items_for_evaluation[user_id]['no_of_correct_recommendations']
        if no_of_correct_recommendations > max_no_of_correct_recommendations:
            max_no_of_correct_recommendations = no_of_correct_recommendations
            best_user_id = user_id
    print("Top {} Items recommended for a user with user_id : {}".format(recommender.no_of_recs, best_user_id))
    items_to_recommend_df = recommender.recommend_items(best_user_id)        
    if items_to_recommend_df is not None:
        recommended_items = list(items_to_recommend_df[item_id_col].values)
      
        if meta_data is not None and 'meta_data_fields' in kwargs:
            cols = [item_id_col]
            cols.extend(kwargs['meta_data_fields'])
            items_to_recommend_df = items_to_recommend_df.merge(meta_data[cols], how='left')
            pprint(items_to_recommend_df)#.to_dict(orient='index'))
        else:                
            for item in recommended_items:
                print(item)

        items_to_be_interacted = items_for_evaluation[best_user_id]['items_to_be_interacted']
        print()
        print("\nItems correctly recommended for a user with user_id  : {}".format(best_user_id))
        correct_recommendations = set(items_to_be_interacted).intersection(set(recommended_items))
        if meta_data is not None and 'meta_data_fields' in kwargs:
            correct_items_to_recommend_df = items_to_recommend_df[items_to_recommend_df[item_id_col].isin(correct_recommendations)]
            print(correct_items_to_recommend_df)
        else:
            for item in correct_recommendations:
                print(item)
    else:
        print("No items to recommend")
    print('*' * 80)

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

    parser.add_argument("train_data", help="Train Data")
    parser.add_argument("test_data", help="Test Data")
    parser.add_argument("meta_data", help="Meta Data")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')

    user_id_col = 'personId'
    item_id_col = 'contentId'
    
    kwargs = dict()
    kwargs['no_of_recs'] = 10
    kwargs['hold_out_strategy'] = 'hold_all'

    # kwargs['hold_out_strategy'] = 'assume_ratio'
    # kwargs['assume_ratio'] = 0.5

    # kwargs['hold_out_strategy'] = 'assume_first_n'
    # kwargs['assume_first_n'] = 5

    # kwargs['hold_out_strategy'] = 'hold_last_n'
    # kwargs['hold_last_n'] = 5

    no_of_recs_to_eval = [5, 10]
    recommender_obj = ContentBasedRecommender

    # if args.meta_data:
    #     print(args.meta_data)
    #     kwargs['meta_data_file'] = args.meta_data
    #     kwargs['meta_data_fields'] = ['url']#, 'title']
    
    model_name = 'models/' + kwargs['hold_out_strategy'] + '_content_based'
    model_dir = os.path.join(current_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if args.train:
        train(recommender_obj,
              results_dir, model_dir,
              args.train_data, args.test_data, args.meta_data,
              user_id_col, item_id_col,
              **kwargs)
    elif args.eval:
        evaluate(recommender_obj,
                results_dir, model_dir,
                args.train_data, args.test_data, args.meta_data,
                user_id_col, item_id_col,
                no_of_recs_to_eval,
                eval_res_file='evaluation_results.json',
                **kwargs)
    elif args.recommend and args.user_id:
        recommend(recommender_obj,
                  results_dir, model_dir,
                  args.train_data, args.test_data, args.meta_data,
                  user_id_col, item_id_col,
                  args.user_id, **kwargs)
    else:
        train_eval_recommend(recommender_obj,
                             results_dir, model_dir,
                             args.train_data, args.test_data, args.meta_data,
                             user_id_col, item_id_col,
                             no_of_recs_to_eval, **kwargs)
if __name__ == '__main__':
    main()
