"""Module for Item Based CF Books Recommender"""
import os
import sys
import pandas
from sklearn.cross_validation import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.item_based_cf import ItemBasedCFRecommender

def load_data(data_dir):
    """Loads data and returns training and test set"""
    #Read learner_id-book_code-view_count triplets
    bookclub_events = os.path.join(data_dir, 'bookclub_events.csv')

    bookclub_events_df = pandas.read_csv(bookclub_events,
                                         parse_dates=['event_time', 'receipt_time'],
                                         dtype={'contents_code': str,
                                                'library_source' : str,
                                                'media_fm' : str})
    books_df = bookclub_events_df.groupby(['learner_id','book_code']).size().reset_index().rename(columns={0:'view_count'})

    train_data, test_data = train_test_split(books_df, test_size = 0.20, random_state=0)

    return train_data, test_data

def main():
    """Method for Item Based Recommender"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train_data, test_data = load_data(data_dir)
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
