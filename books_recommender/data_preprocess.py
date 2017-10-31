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
    print("No of records : ", len(dataframe))
    
    dataframe = dataframe.drop_duplicates(['row_id'])
    print("Removed Duplicate Row_ID, No of records : ", len(dataframe))
    
    # handle books with null code
    print("No of Records with NULL book_code : ", len(dataframe[dataframe['book_code'].isnull()]))#10136500
    dataframe.book_code.fillna(dataframe.contents_code, inplace=True)
    
    # eliminate learners with null id
    dataframe = dataframe[dataframe['learner_id'].notnull()]
    print("Removed learners with null id, No of Records : ", len(dataframe))
    
    #since no of open and close events are unequal, just filter on close events
    events_filter = ((dataframe['event_name'] == 'book_close') |\
                     (dataframe['event_name'] == 'video_close') |\
                     (dataframe['event_name'] == 'audio_close'))
    dataframe = dataframe[events_filter]
    print("After filtering events for close events, No of Records : ", len(dataframe))

    learner_book_df = dataframe.groupby(['learner_id', 'book_code'])\
                               .size().reset_index()\
                               .rename(columns={0: 'events_count'})

    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_data_dir = os.path.join(current_dir, 'preprocessed_data')
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)
    learner_books_file = os.path.join(preprocessed_data_dir, 'learner_books_close_events.csv')
    learner_book_df.to_csv(learner_books_file, index=False)
    print("Preprocessed data available in preprocessed_data/")

if __name__ == '__main__':
    bookclub_events = os.path.join('data/', 'bookclub_events.csv')
    extract_learner_books(bookclub_events)
