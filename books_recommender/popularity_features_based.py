"""Module for Popularity Based Books Recommender with User features"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import popularity_based_features

def main():
    """Popularity based recommender interface"""
    parser = argparse.ArgumentParser(description="Popularity Based Recommender with User features")
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

    no_of_recs = 10
    user_id_col = 'learner_id'
    item_id_col = 'book_code'
    
    #features = ['learner_gender']
    #features = ['age']
    #features = ['learner_gender', 'age']
    
    features_configs = [['learner_gender'], ['age'], ['learner_gender', 'age']]
    
    for features in features_configs:        
        features_str = '_'.join(features)
        model_dir = os.path.join(current_dir, 'model/popularity_based_' + features_str)
        print(model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        train_test_dir = os.path.join(current_dir, 'train_test_data')

        if args.train:
            popularity_based_features.train(results_dir, model_dir, train_test_dir,
                                            user_id_col, item_id_col, features, no_of_recs=no_of_recs)
        elif args.eval:
            no_of_recs_to_eval = [1, 2, 5, 10]
            popularity_based_features.evaluate(results_dir, model_dir, train_test_dir,
                                               user_id_col, item_id_col,
                                               no_of_recs_to_eval, features, dataset='test', no_of_recs=no_of_recs)
        elif args.recommend and args.user_id:
            popularity_based_features.recommend(results_dir, model_dir, train_test_dir,
                                                user_id_col, item_id_col,
                                                args.user_id, features, no_of_recs=no_of_recs)
        else:
            no_of_recs_to_eval = [1, 2, 5, 10]
            popularity_based_features.train_eval_recommend(results_dir, model_dir, train_test_dir,
                                                           user_id_col, item_id_col,
                                                           no_of_recs_to_eval, features,
                                                           dataset='test', no_of_recs=no_of_recs)

if __name__ == '__main__':
    main()
