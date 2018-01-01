"""Module for Preprocessing data for Book Recommendations"""
import os
import sys
import datetime as DT
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_learner_books(bookclub_events, demographics, meta_data_file):
    """Extract learner_id-book_code-events_count triplets with demographics and metadata"""

    print("Loading : {} ...".format(bookclub_events))
    dataframe = pd.read_csv(bookclub_events,
                            parse_dates=['event_time', 'receipt_time'])

    # eliminate rows with duplicate row_id
    print("{:50s}   {}".format("Total No of records : ", len(dataframe)))
    dataframe = dataframe.drop_duplicates(['row_id'])
    print("{:50s}   {}".format("Removed Duplicate Row_ID, No of records : ",
                               len(dataframe)))

    # eliminate books with null id
    dataframe = dataframe[dataframe['book_code'].notnull()]
    print("{:50s}   {}".format("Removed books with null id, No of Records : ",
                               len(dataframe)))

    # eliminate learners with null id
    dataframe = dataframe[dataframe['learner_id'].notnull()]
    print("{:50s}   {}".format("Removed learners with null id, No of Records : ",
                               len(dataframe)))

    # since no of open and close events are unequal, just filter on close
    # events
    events_filter = ((dataframe['event_name'] == 'book_close') |
                     (dataframe['event_name'] == 'video_close') |
                     (dataframe['event_name'] == 'audio_close'))
    dataframe = dataframe[events_filter]
    print("{:50s}   {}".format("After filtering events for close events, No of Records : ",
                               len(dataframe)))

    # sort data by close event time
    print("Sorting records by event time...")
    dataframe.sort_values(by='event_time', inplace=True)

    item_type_stats = dataframe.groupby(['learner_id', 'event_name'])\
                               .agg({'book_code' : (lambda x: len(x.unique()))})\
                               .rename(columns={'book_code' : 'no_of_items'})\
                               .reset_index()

    total_item_type_stats = item_type_stats.groupby('learner_id')\
                                           .agg({'no_of_items' : np.sum})\
                                           .rename(columns={'no_of_items' : 'total_no_of_items'})\
                                           .reset_index()

    learner_item_type_stats = item_type_stats.merge(total_item_type_stats,
                                                    on='learner_id')
    learner_item_type_stats['percentage'] = learner_item_type_stats[
        'no_of_items'] / learner_item_type_stats['total_no_of_items']

    learner_books_df = dataframe.groupby(['learner_id', 'book_code'])\
                                .size()\
                                .reset_index()\
                                .rename(columns={0: 'events_count'})

    first_closure_event_df = dataframe.groupby(['learner_id', 'book_code'])['event_time']\
                                      .min()\
                                      .reset_index()\
                                      .rename(columns={'event_time': 'first_access_time'})

    learner_books_first_closure_df = learner_books_df.merge(first_closure_event_df,
                                                            on=['learner_id', 'book_code'])

    print("Loading : {} ...".format(demographics))
    # merge with demographics info
    demograph = pd.read_csv(demographics,
                            parse_dates=['learner_birthday'])
    now = pd.Timestamp(DT.datetime.now())
    demograph['dob'] = pd.to_datetime(
        demograph['learner_birthday'], format='%m%d%y')
    demograph['dob'] = demograph['dob'].where(demograph['dob'] < now,
                                              demograph['dob'] - np.timedelta64(100, 'Y'))
    demograph['age'] = (now - demograph['dob']).astype('<m8[Y]')

    learner_age_item_type_stats = learner_item_type_stats.merge(demograph,
                                                                on='learner_id')

    all_learners = set(learner_age_item_type_stats['learner_id'].unique())

    media_preference = dict()
    for learner_id in all_learners:
        media_preference[learner_id] = {'book_close' : 0.0,
                                        'audio_close' : 0.0,
                                        'video_close' : 0.0,
                                        'age' : 0
                                       }
    for _, row in learner_age_item_type_stats.iterrows():
        learner_id = row['learner_id']
        event_name = row['event_name']
        preference = row['percentage']
        media_preference[learner_id][event_name] = preference
        media_preference[learner_id]['age'] = row['age']

    list_of_measures = []
    for learner_id in media_preference:
        details = {'learner_id' : learner_id,
                   'age' : media_preference[learner_id]['age'],
                   'book_close' : media_preference[learner_id]['book_close'],
                   'audio_close' : media_preference[learner_id]['audio_close'],
                   'video_close' : media_preference[learner_id]['video_close'],
                  }
        list_of_measures.append(details)
    learner_measures_df = pd.DataFrame(list_of_measures)
    learner_measures_df.set_index('learner_id', inplace=True)


    learner_books_info_df = pd.merge(learner_books_first_closure_df, demograph,
                                     how='inner',
                                     on='learner_id')

    print("Loading : {} ...".format(meta_data_file))
    metadata = pd.read_csv(meta_data_file)
    learner_books_info_meta_df = pd.merge(learner_books_info_df, metadata,
                                          how='inner',
                                          left_on='book_code',
                                          right_on='BOOK_CODE')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_data_dir = os.path.join(current_dir, 'preprocessed_metadata')
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    learner_age_item_type_file = os.path.join(preprocessed_data_dir,
                                              'learner_age_item_type_stats.csv')
    learner_age_item_type_stats.to_csv(learner_age_item_type_file, index=False)

    learner_measures_file = os.path.join(preprocessed_data_dir,
                                         'learner_measures.csv')
    learner_measures_df.to_csv(learner_measures_file)

    learner_books_info_file = os.path.join(preprocessed_data_dir,
                                           'learner_books_info_close_events.csv')
    learner_books_info_meta_df.to_csv(learner_books_info_file, index=False)

    learner_books_info_min_3_events_df = learner_books_info_meta_df[
        learner_books_info_meta_df['events_count'] >= 3]
    learner_books_info_min_3_file = os.path.join(preprocessed_data_dir,
                                                 'learner_books_info_close_min_3_events.csv')
    learner_books_info_min_3_events_df.to_csv(learner_books_info_min_3_file, index=False)

    learner_books_info_min_10_events_df = learner_books_info_meta_df[
        learner_books_info_meta_df['events_count'] >= 10]
    learner_books_info_min_10_file = os.path.join(preprocessed_data_dir,
                                                  'learner_books_info_close_min_10_events.csv')
    learner_books_info_min_10_events_df.to_csv(learner_books_info_min_10_file, index=False)

    print("Preprocessed data available in ", preprocessed_data_dir)

def main():
    """preprocess data"""
    bookclub_events = os.path.join('data/', 'bookclub_events.csv')
    demographics = os.path.join('data/', 'bc.demographics.csv')
    meta_data_file = os.path.join('data/', 'BOOKINFORMATION_META.csv')
    extract_learner_books(bookclub_events, demographics, meta_data_file)

if __name__ == '__main__':
    main()
