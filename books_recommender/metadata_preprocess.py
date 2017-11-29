"""Module for Preprocessing data for Book Recommendations"""
import os
import sys
import pandas as pd
import numpy as np
import datetime as DT

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_learner_books(bookclub_events, demographics, meta_data_file):
    """Extract learner_id-book_code-events_count triplets with demographics and metadata"""
    

    dataframe = pd.read_csv(bookclub_events,
                            parse_dates=['event_time', 'receipt_time'])

    print("No of records : ", len(dataframe))
    dataframe = dataframe.drop_duplicates(['row_id'])
    print("Removed Duplicate Row_ID, No of records : ", len(dataframe))

    # eliminate books with null id
    dataframe = dataframe[dataframe['book_code'].notnull()]
    print("Removed books with null id, No of Records : ", len(dataframe))

    # eliminate learners with null id
    dataframe = dataframe[dataframe['learner_id'].notnull()]
    print("Removed learners with null id, No of Records : ", len(dataframe))

    #since no of open and close events are unequal, just filter on close events
    events_filter = ((dataframe['event_name'] == 'book_close') |\
                     (dataframe['event_name'] == 'video_close') |\
                     (dataframe['event_name'] == 'audio_close'))
    dataframe = dataframe[events_filter]
    print("After filtering events for close events, No of Records : ", len(dataframe))

    learner_books_df = dataframe.groupby(['learner_id', 'book_code'])\
                               .size().reset_index()\
                               .rename(columns={0: 'events_count'})

    #merge with demographics info
    demograph = pd.read_csv(demographics, 
                            parse_dates=['learner_birthday'])
    now = pd.Timestamp(DT.datetime.now())
    demograph['dob'] = pd.to_datetime(demograph['learner_birthday'], format='%m%d%y')
    demograph['dob'] = demograph['dob'].where(demograph['dob'] < now, demograph['dob'] -  np.timedelta64(100, 'Y'))
    demograph['age'] = (now - demograph['dob']).astype('<m8[Y]')

    learner_books_info_df = pd.merge(learner_books_df, demograph, how='inner', on='learner_id')

    metadata = pd.read_csv(meta_data_file)
    learner_books_info_meta_df = pd.merge(learner_books_info_df, metadata, how='inner', left_on='book_code', right_on='BOOK_CODE')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_data_dir = os.path.join(current_dir, 'preprocessed_metadata')
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    learner_books_info_file = os.path.join(preprocessed_data_dir, 'learner_books_info_close_events.csv')
    learner_books_info_meta_df.to_csv(learner_books_info_file, index=False)

    learner_books_info_min_3_events_df = learner_books_info_meta_df[learner_books_info_meta_df['events_count']>=3]
    learner_books_info_min_3_file = os.path.join(preprocessed_data_dir, 'learner_books_info_close_min_3_events.csv')
    learner_books_info_min_3_events_df.to_csv(learner_books_info_min_3_file, index=False)

    learner_books_info_min_10_events_df = learner_books_info_meta_df[learner_books_info_meta_df['events_count']>=10]
    learner_books_info_min_10_file = os.path.join(preprocessed_data_dir, 'learner_books_info_close_min_10_events.csv')
    learner_books_info_min_10_events_df.to_csv(learner_books_info_min_10_file, index=False)

    print("Preprocessed data available in ", preprocessed_data_dir)

if __name__ == '__main__':
    bookclub_events = os.path.join('data/', 'bookclub_events.csv')
    demographics = os.path.join('data/', 'bc.demographics.csv')
    meta_data_file = os.path.join('data/', 'BOOKINFORMATION_META.csv')
    extract_learner_books(bookclub_events, demographics, meta_data_file)
