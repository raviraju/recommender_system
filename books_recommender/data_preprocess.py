"""Module for Preprocessing data for Book Recommendations"""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_learner_books(bookclub_events):
    """Extract learner_id-book_code-events_count triplets"""
    dataframe = pd.read_csv(bookclub_events,
                            parse_dates=['event_time', 'receipt_time'],
                            dtype={'learner_id': int,
                                   'contents_code': str,
                                   'library_source': str,
                                   'media_fm': str})
    dataframe = dataframe.drop_duplicates(['row_id'])
    # eliminate learners with null id
    dataframe = dataframe[dataframe['learner_id'].notnull()]
    # eliminate books with null code
    dataframe = dataframe[dataframe['book_code'].notnull()]
    learner_book_dataframe = dataframe.groupby(['learner_id', 'book_code']).size(
    ).reset_index().rename(columns={0: 'events_count'})

    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_data_dir = os.path.join(current_dir, 'preprocessed_data')
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)
    learner_books_file = os.path.join(
        preprocessed_data_dir, 'learner_books.csv')
    learner_book_dataframe.to_csv(learner_books_file)
    print("Preprocessed data available in preprocessed_data/")

if __name__ == '__main__':
    bookclub_events = os.path.join('data/', 'bookclub_events.csv')
    extract_learner_books(bookclub_events)
