"""Module for Popularity Based Books Recommender"""
import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.reco_interface import load_train_test
from recommender import popularity_based

def recommend(results_dir, model_dir, train_test_dir,
              user_id_col, item_id_col,
              user_id, no_of_recs=10, metadata_fields=None):
    """recommend items for user"""
    train_data, test_data = load_train_test(train_test_dir, user_id_col, item_id_col)

    model = popularity_based.PopularityBasedRecommender(results_dir, model_dir,
                                       train_data, test_data,
                                       user_id_col, item_id_col, no_of_recs)
       
    print("Items interactions for a user with user_id : {}".format(user_id))
    interacted_items = list(test_data[test_data[user_id_col] == user_id][item_id_col])
    for item in interacted_items:
        print(item)
        if (metadata_fields is not None):
            record = test_data[test_data[item_id_col] == item]
            if not record.empty:
                for field in metadata_fields:
                    print("\t{} : {}".format(field, record[field].values[0]))
            
    print()
    print("Items recommended for a user with user_id : {}".format(user_id))
    recommended_items = model.recommend_items()    
    if recommended_items:
        for item in recommended_items:
            print(item)
            if (metadata_fields is not None):
                record = train_data[train_data[item_id_col] == item]
                if not record.empty:
                    for field in metadata_fields:
                        print("\t{} : {}".format(field, record[field].values[0]))
    else:
        print("No items to recommend")
    print('*' * 80)

def main():
    """Popularity based recommender interface"""
    parser = argparse.ArgumentParser(description="Popularity Based Recommender")
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
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_dir = os.path.join(current_dir, 'model/popularity_based')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    no_of_recs = 10
    user_id_col = 'learner_id'
    item_id_col = 'book_code'
    train_test_dir = os.path.join(current_dir, 'train_test_data')
    metadata_fields = ['T_BOOK_NAME', 'T_KEYWORD', 'T_AUTHOR']
    
    if args.train:
        popularity_based.train(results_dir, model_dir, train_test_dir,
                               user_id_col, item_id_col, no_of_recs=no_of_recs)
    elif args.eval:
        no_of_recs_to_eval = [1, 2, 5, 10, 20, 50]
        popularity_based.evaluate(results_dir, model_dir, train_test_dir,
                                  user_id_col, item_id_col,
                                  no_of_recs_to_eval, dataset='test', no_of_recs=no_of_recs)
    elif args.recommend and args.user_id:
        recommend(results_dir, model_dir, train_test_dir,
                  user_id_col, item_id_col,
                  args.user_id, no_of_recs=no_of_recs,
                  metadata_fields=metadata_fields)
    else:
        no_of_recs_to_eval = [1, 2, 5, 10, 20, 50]
        popularity_based.train_eval_recommend(results_dir, model_dir, train_test_dir,
                                              user_id_col, item_id_col,
                                              no_of_recs_to_eval,
                                              dataset='test', no_of_recs=no_of_recs)

if __name__ == '__main__':
    main()
