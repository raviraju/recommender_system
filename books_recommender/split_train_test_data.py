"""Module to split data into train and test"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_random(train_test_dir, data, test_size=0.2):
    """Loads data and returns training and test set"""
    print("Generate Training and Test Data")
    #Read learner_id-book_code-view_count triplets
    learner_books_df = pd.read_csv(data)
    print("{:10} : {:20} : {}".format("Total", "No of records",
                                      len(learner_books_df)))
    #filtering data to be imported
    #split train and test data
    print("{:10} : {:20} : {}".format("Filtered", "No of records",
                                      len(learner_books_df)))
    train_data, test_data = train_test_split(learner_books_df,
                                             test_size=test_size,
                                             random_state=None)
                                             #random_state=0)
                                             #If int, random_state is the seed used
                                             #by the random number generator
    print("{:10} : {:20} : {}".format("Train Data", "No of records", len(train_data)))
    print("{:10} : {:20} : {}".format("Test Data", "No of records", len(test_data)))
    train_data_file = os.path.join(train_test_dir, 'train_data.csv')
    train_data.to_csv(train_data_file)
    test_data_file = os.path.join(train_test_dir, 'test_data.csv')
    test_data.to_csv(test_data_file)
    print("Train and Test Data are in ", train_test_dir)

def main():
    """interface to load and split data into train and test"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_test_dir = os.path.join(current_dir, 'train_test_data')
    if not os.path.exists(train_test_dir):
        os.makedirs(train_test_dir)

    data_dir = os.path.join(current_dir, 'preprocessed_data')
    data = os.path.join(data_dir, 'learner_books_close_min_events.csv')

    generate_random(train_test_dir, data)

if __name__ == "__main__":
    main()
