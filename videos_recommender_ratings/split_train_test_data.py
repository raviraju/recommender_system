"""Module to split data into train and test"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import defaultdict

def split_items(items_interacted, hold_last_n=5):
    """return assume_interacted_items, hold_out_items"""
    assume_interacted_items = []
    hold_out_items = []
    
    # print("items_interacted : ", items_interacted)
    no_of_items_interacted = len(items_interacted)
    # print("no_of_items_interacted : ", no_of_items_interacted)

    if no_of_items_interacted <= hold_last_n:
        return assume_interacted_items, hold_out_items
   
    no_of_items_assumed_interacted = no_of_items_interacted - hold_last_n
    no_of_items_to_be_held = hold_last_n

    if no_of_items_assumed_interacted != 0:
        assume_interacted_items = items_interacted[:no_of_items_assumed_interacted]
    if no_of_items_to_be_held != 0:
        hold_out_items = items_interacted[no_of_items_assumed_interacted:(no_of_items_assumed_interacted+no_of_items_to_be_held)]

    # print("no_of_items_assumed_interacted : ", no_of_items_assumed_interacted)
    # print("assume_interacted_items : ", assume_interacted_items)
    
    # print("no_of_items_to_be_held : ", no_of_items_to_be_held)
    # print("hold_out_items : ", hold_out_items)
    # input()
    return assume_interacted_items, hold_out_items

def apply_hold_out_strategy(train_data, test_data,
                            user_id_col, item_id_col,
                            hold_out_strategy = "hold_last_n"):
    """Items Hold Out strategy"""
    if hold_out_strategy == "hold_last_n":
        
        assumed_items_test_data = pd.DataFrame()
        hold_out_items_test_data = pd.DataFrame()

        for user_id in test_data[user_id_col].unique():
            user_test_data = test_data[test_data[user_id_col] == user_id]
            
            items_interacted = list(user_test_data[item_id_col])
            assume_interacted_items, hold_out_items = split_items(items_interacted)

            assume_items_data   = user_test_data[user_test_data[item_id_col].isin(assume_interacted_items)]
            hold_out_items_data = user_test_data[user_test_data[item_id_col].isin(hold_out_items)]

            assumed_items_test_data = assumed_items_test_data.append(assume_items_data, ignore_index=True)
            hold_out_items_test_data = hold_out_items_test_data.append(hold_out_items_data, ignore_index=True)
    
        training_data = train_data.copy()
        training_data = training_data.append(assumed_items_test_data, ignore_index=True)
        testing_data = hold_out_items_test_data
        return training_data, testing_data
    
    else:
        print("Invalid hold_out_strategy : ", hold_out_strategy)
        exit(-1)    

def generate_users_split(data, user_id_col, item_id_col,
                         validation_size = 0.2, test_size=0.2, min_no_of_items=1):
    """Loads data and returns training and test set by random selection of users"""
    current_dir = os.path.dirname(os.path.abspath(__file__))       
    train_test_dir = os.path.join(current_dir, 'train_test_data/users_split')
    if not os.path.exists(train_test_dir):
        os.makedirs(train_test_dir)

    #Read user_id-item_id-events
    print("Loading Data...")
    events_df = pd.read_csv(data)    
    print("{:30} : {:20} : {}".format("Data", "No of records", len(events_df)))
    print("{:30} : {:20} : {}".format("Data", "No of users", len(events_df[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Data", "No of items", len(events_df[item_id_col].unique())))
    
    events_df = events_df.drop(['Unnamed: 0'], axis=1)
    
    #filtering data to be imported
    print('*'*80)
    print("Filtering Data on Min no of items being rated = ", min_no_of_items)
    user_items_df = events_df.groupby([user_id_col])\
                             .agg({item_id_col: (lambda x: len(x.unique()))})\
                             .rename(columns={item_id_col : 'no_of_items'})\
                             .reset_index()
    dist = user_items_df['no_of_items'].describe()
    no_of_items_list = [dist['min'], dist['25%'], dist['50%'], dist['75%'], dist['max']]
    print("Distribution of item_counts (min, 25%, 50%, 75%, max)")
    for no_of_items in no_of_items_list:
        no_of_users = len(user_items_df[user_items_df['no_of_items'] <= no_of_items])
        print("{:15} : {:5} {:15} : {:5}".format("No of items", no_of_items, " No of users", no_of_users))
    print()
    input()
    user_min_items_df = user_items_df[user_items_df['no_of_items'] >= min_no_of_items]        
    filtered_events_df = pd.merge(user_min_items_df, events_df, how='inner', on=user_id_col)
    print()
    print("{:30} : {:20} : {}".format("Filtered Data", "No of records", len(filtered_events_df)))
    print("{:30} : {:20} : {}".format("Filtered Data", "No of users", len(filtered_events_df[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Filtered Data", "No of items", len(filtered_events_df[item_id_col].unique())))
    
    print('*'*80)
    print("Generating Training and Test Data by randomly selecting test users...")
    users = user_min_items_df[user_id_col].unique()
    no_of_users = len(users)    
    no_of_validation_users = int(no_of_users * validation_size)
    no_of_test_users = int(no_of_users * test_size)    
    no_of_train_users = no_of_users - no_of_validation_users - no_of_test_users
    no_of_train_n_validation_users = no_of_train_users + no_of_validation_users

    users_set = set(users)
    #randomly select validation users
    validation_users_set = set(np.random.choice(users, no_of_validation_users, replace=False))
    remaining_users_set = users_set - validation_users_set
    remaining_users = list(remaining_users_set)
    #randomly select test users
    test_users_set = set(np.random.choice(remaining_users, no_of_test_users, replace=False))
    #assign train users to those who do not belong to validation and test
    train_users_set = users_set - validation_users_set - test_users_set

    common_users = train_users_set & validation_users_set & test_users_set
    print("No of users             : {}".format(len(users_set)))
    print("No of train users       : {}".format(len(train_users_set)))
    print("No of validation users  : {}".format(len(validation_users_set)))
    print("No of test users        : {}".format(len(test_users_set)))
    print("No of common users      : {}".format(len(common_users)))

    train_data = events_df[events_df[user_id_col].isin(train_users_set)]
    validation_data = events_df[events_df[user_id_col].isin(validation_users_set)]
    test_data = events_df[events_df[user_id_col].isin(test_users_set)]
    
    train_n_validation_data = events_df[events_df[user_id_col].isin(train_users_set) |
                                           events_df[user_id_col].isin(validation_users_set)]

    common_users = set(train_data[user_id_col].unique()) & \
                   set(validation_data[user_id_col].unique()) & \
                   set(test_data[user_id_col].unique())
    common_items = set(train_data[item_id_col].unique()) & \
                   set(validation_data[item_id_col].unique()) & \
                   set(test_data[item_id_col].unique())

    print()
    print("{:30} : {:20} : {}".format("Train Data", "No of records", len(train_data)))
    print("{:30} : {:20} : {}".format("Train Data", "No of users",
                                      len(train_data[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Train Data", "No of items",
                                      len(train_data[item_id_col].unique())))
    print()
    print("{:30} : {:20} : {}".format("Validation Data", "No of records", len(test_data)))
    print("{:30} : {:20} : {}".format("Validation Data", "No of users",
                                      len(validation_data[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Validation Data", "No of items",
                                      len(validation_data[item_id_col].unique())))    
    print()
    print("{:30} : {:20} : {}".format("Test Data", "No of records", len(test_data)))
    print("{:30} : {:20} : {}".format("Test Data", "No of users",
                                      len(test_data[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Test Data", "No of items",
                                      len(test_data[item_id_col].unique())))
    print()
    print("{:30} : {:20} : {}".format("Common ", "No of users", len(common_users)))
    print("{:30} : {:20} : {}".format("Common ", "No of items", len(common_items)))
    print()
    print("{:30} : {:20} : {}".format("Train and Validation Data", "No of records", len(train_n_validation_data)))
    print("{:30} : {:20} : {}".format("Train and Validation Data", "No of users",
                                      len(train_n_validation_data[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Train and Validation Data", "No of items",
                                      len(train_n_validation_data[item_id_col].unique())))
    
    '''
    train_data_file = os.path.join(train_test_dir, 'train_data.csv')    
    train_data.to_csv(train_data_file, index=False)
    validation_data_file = os.path.join(train_test_dir, 'validation_data.csv')    
    validation_data.to_csv(validation_data_file, index=False)
    test_data_file = os.path.join(train_test_dir, 'test_data.csv')    
    test_data.to_csv(test_data_file, index=False)
    '''
    #######################################################################################################
    print('*'*80)
    print("Preparing Train and Validation Data...")
    print("Applying hold_out_strategy")
    training_data_for_validation, validating_data = apply_hold_out_strategy(train_data, 
                                                                            validation_data, 
                                                                            user_id_col, item_id_col)
    common_users = set(training_data_for_validation[user_id_col].unique()) & set(validating_data[user_id_col].unique())
    common_items = set(training_data_for_validation[item_id_col].unique()) & set(validating_data[item_id_col].unique())

    print("{:30} : {:20} : {}".format("Training Data For Validation", "No of records", len(training_data_for_validation)))
    print("{:30} : {:20} : {}".format("Training Data For Validation", "No of users",
                                      len(training_data_for_validation[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Training Data For Validation", "No of items",
                                      len(training_data_for_validation[item_id_col].unique())))
    print()
    print("{:30} : {:20} : {}".format("Validating Data", "No of records", len(validating_data)))
    print("{:30} : {:20} : {}".format("Validating Data", "No of users",
                                      len(validating_data[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Validating Data", "No of items",
                                      len(validating_data[item_id_col].unique())))
    print()
    print("{:30} : {:20} : {}".format("Common ", "No of users", len(common_users)))
    print("{:30} : {:20} : {}".format("Common ", "No of items", len(common_items)))
    
    '''
    training_data_for_validation_file = os.path.join(train_test_dir, 'training_data_for_validation.csv')    
    training_data_for_validation.to_csv(training_data_for_validation_file, index=False)
    validating_data_file = os.path.join(train_test_dir, 'validating_data.csv')    
    validating_data.to_csv(validating_data_file, index=False)
    '''
    
    user_item_rating = [user_id_col, item_id_col, 'like_rating']    
    training_for_validation_uir_file = os.path.join(train_test_dir, 'training_for_validation_uir_data.csv')
    training_data_for_validation[user_item_rating].to_csv(training_for_validation_uir_file, index=False)
    validating_data_uir_file = os.path.join(train_test_dir, 'validation_uir_data.csv')
    validating_data[user_item_rating].to_csv(validating_data_uir_file, index=False)
    #######################################################################################################    
    print('*'*80)
    print("Preparing Train and Test Data...")
    print("Applying hold_out_strategy")
    training_data_for_test, testing_data = apply_hold_out_strategy(train_data, test_data, user_id_col, item_id_col)
    common_users = set(training_data_for_test[user_id_col].unique()) & set(testing_data[user_id_col].unique())
    common_items = set(training_data_for_test[item_id_col].unique()) & set(testing_data[item_id_col].unique())

    print("{:30} : {:20} : {}".format("Training Data For Test", "No of records", len(training_data_for_test)))
    print("{:30} : {:20} : {}".format("Training Data For Test", "No of users",
                                      len(training_data_for_test[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Training Data For Test", "No of items",
                                      len(training_data_for_test[item_id_col].unique())))
    print()
    print("{:30} : {:20} : {}".format("Testing Data", "No of records", len(testing_data)))
    print("{:30} : {:20} : {}".format("Testing Data", "No of users",
                                      len(testing_data[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Testing Data", "No of items",
                                      len(testing_data[item_id_col].unique())))
    print()
    print("{:30} : {:20} : {}".format("Common ", "No of users", len(common_users)))
    print("{:30} : {:20} : {}".format("Common ", "No of items", len(common_items)))
    
    '''
    training_data_for_test_file = os.path.join(train_test_dir, 'training_data_for_test.csv')    
    training_data_for_test.to_csv(training_data_for_test_file, index=False)
    testing_data_file = os.path.join(train_test_dir, 'testing_data.csv')    
    testing_data.to_csv(testing_data_file, index=False)
    '''
    
    user_item_rating = [user_id_col, item_id_col, 'like_rating']    
    training_for_test_uir_file = os.path.join(train_test_dir, 'training_for_test_uir_data.csv')
    training_data_for_test[user_item_rating].to_csv(training_for_test_uir_file, index=False)
    testing_data_uir_file = os.path.join(train_test_dir, 'test_uir_data.csv')
    testing_data[user_item_rating].to_csv(testing_data_uir_file, index=False)
    #######################################################################################################    
    print('*'*80)
    print("Preparing Train+Validation and Test Data...")
    print("Applying hold_out_strategy")
    training_all, testing_all = apply_hold_out_strategy(train_n_validation_data, test_data, 
                                                        user_id_col, item_id_col)
    common_users = set(training_all[user_id_col].unique()) & set(testing_all[user_id_col].unique())
    common_items = set(training_all[item_id_col].unique()) & set(testing_all[item_id_col].unique())

    print("{:30} : {:20} : {}".format("Train n Validation Data For Test", "No of records",
                                      len(training_all)))
    print("{:30} : {:20} : {}".format("Train n Validation Data For Test", "No of users",
                                      len(training_all[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Train n Validation Data For Test", "No of items",
                                      len(training_all[item_id_col].unique())))
    print()
    print("{:30} : {:20} : {}".format("Testing Data", "No of records",
                                      len(testing_all)))
    print("{:30} : {:20} : {}".format("Testing Data", "No of users",
                                      len(testing_all[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Testing Data", "No of items",
                                      len(testing_all[item_id_col].unique())))
    print()
    print("{:30} : {:20} : {}".format("Common ", "No of users", len(common_users)))
    print("{:30} : {:20} : {}".format("Common ", "No of items", len(common_items)))
    
    '''
    training_all_file = os.path.join(train_test_dir, 'training_all.csv')    
    training_all.to_csv(training_all_file, index=False)
    testing_all_file = os.path.join(train_test_dir, 'testing_all.csv')    
    testing_all.to_csv(testing_all_file, index=False)
    '''
    
    user_item_rating = [user_id_col, item_id_col, 'like_rating']    
    training_all_uir_file = os.path.join(train_test_dir,'training_all_uir_data.csv')
    training_all[user_item_rating].to_csv(training_all_uir_file, index=False)

    testing_all_uir_file = os.path.join(train_test_dir, 'testing_all_uir_data.csv')
    testing_all[user_item_rating].to_csv(testing_all_uir_file, index=False)
    print("Train and Test Data are in ", train_test_dir)

def generate_kfolds_split(data, user_id_col, item_id_col, rating_col,
                          validation_size = 0.2, no_of_kfolds=10, min_no_of_items=1):
    """Loads data and returns training and test sets by kfolds selection of users"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_test_dir = os.path.join(current_dir, 'train_test_data/kfolds_split')
    if not os.path.exists(train_test_dir):
        os.makedirs(train_test_dir)

    #Read user_id-item_id-events
    print("Loading Data...")    
    events_df = pd.read_csv(data)    
    print("{:30} : {:20} : {}".format("Data", "No of records", len(events_df)))
    print("{:30} : {:20} : {}".format("Data", "No of users", len(events_df[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Data", "No of items", len(events_df[item_id_col].unique())))
    
    events_df = events_df.drop(['Unnamed: 0'], axis=1)
    
    #filtering data to be imported
    print('*'*80)
    print("Filtering Data...")
    user_items_df = events_df.groupby([user_id_col])\
                             .agg({item_id_col: (lambda x: len(x.unique()))})\
                             .rename(columns={item_id_col : 'no_of_items'})\
                             .reset_index()
    dist = user_items_df['no_of_items'].describe()
    no_of_items_list = [dist['min'], dist['25%'], dist['50%'], dist['75%'], dist['max']]
    print("Distribution of item_counts (min, 25%, 50%, 75%, max)")
    for no_of_items in no_of_items_list:
        no_of_users = len(user_items_df[user_items_df['no_of_items'] >= no_of_items])
        print("No of items : ", no_of_items, " No of users : ", no_of_users)
    print()

    user_min_items_df = user_items_df[user_items_df['no_of_items'] >= min_no_of_items]        
    data = pd.merge(user_min_items_df, events_df, how='inner', on=user_id_col)
    print("Min no of items : ", min_no_of_items)
    print()
    print("{:30} : {:20} : {}".format("Filtered Data", "No of records", len(data)))
    print("{:30} : {:20} : {}".format("Filtered Data", "No of users", len(data[user_id_col].unique())))
    print("{:30} : {:20} : {}".format("Filtered Data", "No of items", len(data[item_id_col].unique())))
    
    print('*'*80)
    print("Generating Training and Test Data by kfolding on users...")
    users = user_min_items_df[user_id_col].unique()
    kfolds = KFold(n_splits=no_of_kfolds)
    i = 1
    experiments = dict()
    experiments['train'] = dict()
    experiments['validation'] = dict()
    experiments['test'] = dict()
    experiments['train_n_validation'] = dict()
    features = [user_id_col, item_id_col, 'like_rating']
    for train_n_validation_indices, test_indices in kfolds.split(users):
        #print("%s %s" % (train_n_validation_indices, test_indices))
        
        train_n_validation_users = users[train_n_validation_indices]
        no_of_validation_users = int(len(train_n_validation_users) * validation_size)
        #randomly select validation users from train_n_validation
        validation_users_set = set(np.random.choice(train_n_validation_users, 
                                                    no_of_validation_users, replace=False))
        train_n_validation_users_set = set(train_n_validation_users)
        train_users_set = train_n_validation_users_set - validation_users_set
        test_users_set = set(users[test_indices])
        #print(train, test)
        experiments['train'][i] = train_users_set
        experiments['validation'][i] = validation_users_set
        experiments['test'][i] = test_users_set
        experiments['train_n_validation'][i] = train_n_validation_users_set

        train_data = events_df[events_df[user_id_col].isin(train_users_set)]
        validation_data = events_df[events_df[user_id_col].isin(validation_users_set)]
        test_data = events_df[events_df[user_id_col].isin(test_users_set)]
        train_n_validation_data = events_df[events_df[user_id_col].isin(train_n_validation_users_set)]
        
        train_data_users = set(train_data[user_id_col].unique())
        validation_data_users = set(validation_data[user_id_col].unique())
        test_data_users = set(test_data[user_id_col].unique())
        common_users = train_data_users & validation_data_users & test_data_users        
        train_n_validation_data_users = set(train_n_validation_data[user_id_col].unique())

        train_data_items = set(train_data[item_id_col].unique())
        validation_data_items = set(validation_data[item_id_col].unique())
        test_data_items = set(test_data[item_id_col].unique())
        common_items = train_data_items & validation_data_items & test_data_items
        train_n_validation_data_items = set(train_n_validation_data[item_id_col].unique())

        print()
        print("{} {:30} : {:20} : {}".format(i, "Train Data",
                                             "No of records", len(train_data)))
        print("{} {:30} : {:20} : {}".format(i, "Train Data",
                                             "No of users", len(train_data_users)))
        print("{} {:30} : {:20} : {}".format(i, "Train Data",
                                             "No of items", len(train_data_items)))
        print()
        print("{} {:30} : {:20} : {}".format(i, "Validation Data",
                                             "No of records", len(validation_data)))
        print("{} {:30} : {:20} : {}".format(i, "Validation Data",
                                             "No of users", len(validation_data_users)))
        print("{} {:30} : {:20} : {}".format(i, "Validation Data",
                                             "No of items", len(validation_data_items)))
        print()
        print("{} {:30} : {:20} : {}".format(i, "Test Data",
                                             "No of records", len(test_data)))
        print("{} {:30} : {:20} : {}".format(i, "Test Data",
                                             "No of users", len(test_data_users)))
        print("{} {:30} : {:20} : {}".format(i, "Test Data",
                                             "No of items", len(test_data_items)))
        print()
        print("{} {:30} : {:20} : {}".format(i, "Common ",
                                             "No of users", len(common_users)))
        print("{} {:30} : {:20} : {}".format(i, "Common ",
                                             "No of items", len(common_items)))
        print()
        print("{} {:30} : {:20} : {}".format(i, "Train and Validation Data", 
                                             "No of records", len(train_n_validation_data)))
        print("{} {:30} : {:20} : {}".format(i, "Train and Validation Data", 
                                             "No of users", len(train_n_validation_data_users)))
        print("{} {:30} : {:20} : {}".format(i, "Train and Validation Data", 
                                             "No of items", len(train_n_validation_data_items)))

        '''
        print("Dump train and test files...")
        train_data_file = os.path.join(train_test_dir, str(i) + '_train_data.csv')
        train_data.to_csv(train_data_file, index=False)
        validation_data_file = os.path.join(train_test_dir, str(i) + '_validation_data.csv')    
        validation_data.to_csv(validation_data_file, index=False)    
        test_data_file = os.path.join(train_test_dir, str(i) + '_test_data.csv')
        test_data.to_csv(test_data_file, index=False)
        train_n_validation_data_file = os.path.join(train_test_dir, str(i) + '_train_n_validation_data.csv')    
        train_n_validation_data.to_csv(train_n_validation_data_file, index=False)
        '''

        #######################################################################################################
        print('*'*80)
        print("Preparing Train and Validation Data...")
        print("Applying hold_out_strategy")
        training_data_for_validation, validating_data = apply_hold_out_strategy(train_data, 
                                                                                validation_data, 
                                                                                user_id_col, item_id_col)
        training_data_for_validation_users = set(training_data_for_validation[user_id_col].unique())
        training_data_for_validation_items = set(training_data_for_validation[item_id_col].unique())
        
        validating_data_users = set(validating_data[user_id_col].unique())
        validating_data_items = set(validating_data[item_id_col].unique())
        
        common_users = training_data_for_validation_users & validating_data_users
        common_items = training_data_for_validation_items & validating_data_items

        print("{:30} : {:20} : {}".format("Training Data For Validation", "No of records", len(training_data_for_validation)))
        print("{:30} : {:20} : {}".format("Training Data For Validation", "No of users",
                                          len(training_data_for_validation_users)))
        print("{:30} : {:20} : {}".format("Training Data For Validation", "No of items",
                                          len(training_data_for_validation_items)))
        print()
        print("{:30} : {:20} : {}".format("Validating Data", "No of records", len(validating_data)))
        print("{:30} : {:20} : {}".format("Validating Data", "No of users",
                                          len(validating_data_users)))
        print("{:30} : {:20} : {}".format("Validating Data", "No of items",
                                          len(validating_data_items)))
        print()
        print("{:30} : {:20} : {}".format("Common ", "No of users", len(common_users)))
        print("{:30} : {:20} : {}".format("Common ", "No of items", len(common_items)))
        
        '''
        known_train_data = dict()
        for _, row in training_data_for_validation.iterrows():
            #print(row[user_id_col], row[item_id_col], row[rating_col])
            known_train_data[(row[user_id_col], row[item_id_col])] = row[rating_col]
            
        known_validating_data = defaultdict(int)
        for i, row in validating_data.iterrows():
            #print(row[user_id_col], row[item_id_col], row[rating_col])
            known_validating_data[(row[user_id_col], row[item_id_col])] = row[rating_col]
        
        all_items = training_data_for_validation_items | validating_data_items
        validating_data_triplets = []
        for user_id in validating_data_users:
            for item_id in all_items:
                if (user_id, item_id) in known_train_data:
                    continue
                else:
                    rating = known_validating_data[(user_id, item_id)]
                    validating_data_triplet = {
                        user_id_col : user_id,
                        item_id_col : item_id,
                        rating_col : rating
                    }
                    validating_data_triplets.append(validating_data_triplet)
        validating_data_triplets_df = pd.DataFrame(validating_data_triplets)
        print(validating_data_triplets_df.head())
        '''

        '''
        training_data_for_validation_file = os.path.join(train_test_dir, str(i) + '_training_data_for_validation.csv')    
        training_data_for_validation.to_csv(training_data_for_validation_file, index=False)
        validating_data_file = os.path.join(train_test_dir, str(i) + '_validating_data.csv')    
        validating_data.to_csv(validating_data_file, index=False)
        '''

        user_item_rating = [user_id_col, item_id_col, 'like_rating']    
        training_for_validation_uir_file = os.path.join(train_test_dir, str(i) + '_training_for_validation_uir_data.csv')
        training_data_for_validation[user_item_rating].to_csv(training_for_validation_uir_file, index=False)
        validating_data_uir_file = os.path.join(train_test_dir, str(i) + '_validation_uir_data.csv')
        validating_data[user_item_rating].to_csv(validating_data_uir_file, index=False)
        
        #all_validating_data_uir_file = os.path.join(train_test_dir, str(i) + '_all_validation_uir_data.csv')
        #validating_data_triplets_df[user_item_rating].to_csv(all_validating_data_uir_file, index=False)
        #######################################################################################################    
        print('*'*80)
        print("Preparing Train and Test Data...")
        print("Applying hold_out_strategy")
        training_data_for_test, testing_data = apply_hold_out_strategy(train_data, test_data, user_id_col, item_id_col)
        common_users = set(training_data_for_test[user_id_col].unique()) & set(testing_data[user_id_col].unique())
        common_items = set(training_data_for_test[item_id_col].unique()) & set(testing_data[item_id_col].unique())

        print("{:30} : {:20} : {}".format("Training Data For Test", "No of records", len(training_data_for_test)))
        print("{:30} : {:20} : {}".format("Training Data For Test", "No of users",
                                          len(training_data_for_test[user_id_col].unique())))
        print("{:30} : {:20} : {}".format("Training Data For Test", "No of items",
                                          len(training_data_for_test[item_id_col].unique())))
        print()
        print("{:30} : {:20} : {}".format("Testing Data", "No of records", len(testing_data)))
        print("{:30} : {:20} : {}".format("Testing Data", "No of users",
                                          len(testing_data[user_id_col].unique())))
        print("{:30} : {:20} : {}".format("Testing Data", "No of items",
                                          len(testing_data[item_id_col].unique())))
        print()
        print("{:30} : {:20} : {}".format("Common ", "No of users", len(common_users)))
        print("{:30} : {:20} : {}".format("Common ", "No of items", len(common_items)))

        '''
        training_data_for_test_file = os.path.join(train_test_dir, str(i) + '_training_data_for_test.csv')    
        training_data_for_test.to_csv(training_data_for_test_file, index=False)
        testing_data_file = os.path.join(train_test_dir, str(i) + '_testing_data.csv')    
        testing_data.to_csv(testing_data_file, index=False)
        '''
        user_item_rating = [user_id_col, item_id_col, 'like_rating']    
        training_for_test_uir_file = os.path.join(train_test_dir, str(i) + '_training_for_test_uir_data.csv')
        training_data_for_test[user_item_rating].to_csv(training_for_test_uir_file, index=False)
        testing_data_uir_file = os.path.join(train_test_dir, str(i) + '_test_uir_data.csv')
        testing_data[user_item_rating].to_csv(testing_data_uir_file, index=False)
        #######################################################################################################    
        print('*'*80)
        print("Preparing Train+Validation and Test Data...")
        print("Applying hold_out_strategy")
        training_all, testing_all = apply_hold_out_strategy(train_n_validation_data, test_data, 
                                                            user_id_col, item_id_col)
        common_users = set(training_all[user_id_col].unique()) & set(testing_all[user_id_col].unique())
        common_items = set(training_all[item_id_col].unique()) & set(testing_all[item_id_col].unique())

        print("{:30} : {:20} : {}".format("Train n Validation Data For Test", "No of records",
                                          len(training_all)))
        print("{:30} : {:20} : {}".format("Train n Validation Data For Test", "No of users",
                                          len(training_all[user_id_col].unique())))
        print("{:30} : {:20} : {}".format("Train n Validation Data For Test", "No of items",
                                          len(training_all[item_id_col].unique())))
        print()
        print("{:30} : {:20} : {}".format("Testing Data", "No of records",
                                          len(testing_all)))
        print("{:30} : {:20} : {}".format("Testing Data", "No of users",
                                          len(testing_all[user_id_col].unique())))
        print("{:30} : {:20} : {}".format("Testing Data", "No of items",
                                          len(testing_all[item_id_col].unique())))
        print()
        print("{:30} : {:20} : {}".format("Common ", "No of users", len(common_users)))
        print("{:30} : {:20} : {}".format("Common ", "No of items", len(common_items)))
        '''
        training_all_file = os.path.join(train_test_dir, str(i) + '_training_all.csv')    
        training_all.to_csv(training_all_file, index=False)
        testing_all_file = os.path.join(train_test_dir, str(i) + '_testing_all.csv')    
        testing_all.to_csv(testing_all_file, index=False)
        '''

        user_item_rating = [user_id_col, item_id_col, 'like_rating']    
        training_all_uir_file = os.path.join(train_test_dir,str(i) + '_training_all_uir_data.csv')
        training_all[user_item_rating].to_csv(training_all_uir_file, index=False)

        testing_all_uir_file = os.path.join(train_test_dir, str(i) + '_testing_all_uir_data.csv')
        testing_all[user_item_rating].to_csv(testing_all_uir_file, index=False)
        #######################################################################################################
        i += 1

        print('*'*30)
    print("Train and Test Data are in ", train_test_dir)
    #Validation of kfold splits
    all_users = set(users)
    all_test_users = set()
    for i in experiments['test']:
        all_test_users |= set(experiments['test'][i])
    print("Users not considered in testing : ", all_users - all_test_users)
    all_train_users = set()
    for i in experiments['train_n_validation']:
        all_train_users |= set(experiments['train_n_validation'][i])
    print("Users not considered in train_n_validation : ", all_users - all_train_users)

def main():
    """interface to load and split data into train and test"""
    parser = argparse.ArgumentParser(description="Split train and test data")
    parser.add_argument("--random_split",
                        help="Random split data into train and test",
                        action="store_true")
    parser.add_argument("--users_split",
                        help="split users into train and test",
                        action="store_true")
    parser.add_argument("--kfolds_split",
                        help="generate cross validation train and test",
                        action="store_true")
    parser.add_argument("--min_no_of_items",
                        help="min_no_of_items", type=int)
    parser.add_argument("--validation_size",
                        help="validation_size ratio", type=float)
    parser.add_argument("--test_size",
                        help="test_size ratio", type=float)
    parser.add_argument("--no_of_kfolds",
                        help="no of k folds", type=int)
    parser.add_argument("data", help="data used to split into train and test")
    parser.add_argument("user_id_col", help="user_id column name")
    parser.add_argument("item_id_col", help="item_id column name")
    parser.add_argument("rating_col",  help="rating column name")
    args = parser.parse_args()
    
    if args.min_no_of_items is None:
        min_no_of_items = 15
    else:
        min_no_of_items = args.min_no_of_items     

    if args.users_split and args.validation_size and args.test_size and args.data:
        generate_users_split(args.data, args.user_id_col, args.item_id_col, args.rating_col,
                             args.validation_size, args.test_size, min_no_of_items)
    elif args.kfolds_split and args.validation_size and args.no_of_kfolds and args.data:
        generate_kfolds_split(args.data, args.user_id_col, args.item_id_col, args.rating_col,
                              args.validation_size, args.no_of_kfolds, min_no_of_items)
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()
