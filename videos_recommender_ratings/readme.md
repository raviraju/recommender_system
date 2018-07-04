python split_train_test_data.py --users_split --test_size 0.2 --min_no_of_items 10 --validation_size 0.2 preprocessed_data/latest_rating_new.csv learner_id media_id > preprocessed_data/users_split.txt
python split_train_test_data.py --kfolds_split --no_of_kfolds 10 --min_no_of_items 10 --validation_size 0.2 preprocessed_data/latest_rating_new.csv learner_id media_id > preprocessed_data/kfolds_split.txt