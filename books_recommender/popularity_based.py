"""Module for Popularity Based Books Recommender"""
import os
import sys
import pandas
from sklearn.cross_validation import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.popularity_based import PopularityBasedRecommender

def load_data():
    """Loads data and returns training and test set"""
    #Read learner_id-book_code-view_count triplets
    bookclub_events = 'data/bookclub_events.csv'

    bookclub_events_df = pandas.read_csv(bookclub_events,
                                         parse_dates=['event_time', 'receipt_time'],
                                         dtype={'contents_code': str,
                                                'library_source' : str,
                                                'media_fm' : str})
    books_df = bookclub_events_df.groupby(['learner_id','book_code']).size().reset_index().rename(columns={0:'view_count'})

    train_data, test_data = train_test_split(books_df, test_size = 0.20, random_state=0)

    return train_data, test_data

def main():
    """Method for Popularity Based Recommender"""
    train_data, test_data = load_data()
    #print(train_data.head(5))
    #print(test_data.head(5))
    popularity_reco = PopularityBasedRecommender()
    popularity_reco.train(train_data, 'learner_id', 'book_code')

    learners = test_data['learner_id'].unique()
    learner_id = learners[0]
    #learner_id = 5
    recommendations = popularity_reco.recommend(learner_id)
    print(recommendations)

if __name__ == '__main__':
    main()
