"""Module for Item Based CF Books Recommender"""
import os
import sys
import pandas
from sklearn.cross_validation import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.item_based_cf import ItemBasedCFRecommender

def load_train_test(learner_books_file):
    """Loads data and returns training and test set"""
    #Read learner_id-book_code-view_count triplets
    learner_books_df = pandas.read_csv(learner_books_file)

    train_data, test_data = train_test_split(learner_books_df, test_size = 0.20, random_state=0)

    return train_data, test_data

def main():
    """Method for Item Based Recommender"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_data = os.path.join(current_dir, 'preprocessed_data')
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    learner_books_file = os.path.join(preprocessed_data, 'learner_books.csv')
    train_data, test_data = load_train_test(learner_books_file)
    #print(train_data.head(5))
    #print(test_data.head(5))
    item_based_cf_reco = ItemBasedCFRecommender(results_dir)
    item_based_cf_reco.train(train_data, 'learner_id', 'book_code')

    learners = test_data['learner_id'].unique()
    learner_id = learners[0]
    #learner_id = 5
    recommendations = item_based_cf_reco.recommend(learner_id)
    #print(recommendations)
    print("Item Based Recommendations are found in results/")

if __name__ == '__main__':
    main()
