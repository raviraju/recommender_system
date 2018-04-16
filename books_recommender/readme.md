## Data Preprocessing
python preprocess_metadata.py

## Split Data into Train and Test
python split_train_test_data.py --random_split --min_no_of_books 10 --test_size 0.2 preprocessed_metadata/learner_books_info_close_events.csv
python split_train_test_data.py --users_split --min_no_of_books 10 --test_size 0.2 preprocessed_metadata/learner_books_info_close_events.csv
python split_train_test_data.py --kfold_split --kfolds 10 --min_no_of_books 10 preprocessed_metadata/learner_books_info_close_events.csv


## Generate recommendations
python rec_random_based.py train_test_data/train_data.csv train_test_data/test_data.csv
python rec_popularity_based.py train_test_data/train_data.csv train_test_data/test_data.csv
python rec_item_based_cf.py train_test_data/train_data.csv train_test_data/test_data.csv
python rec_user_based_cf.py train_test_data/train_data.csv train_test_data/test_data.csv
python rec_content_based.py train_test_data/train_data.csv train_test_data/test_data.csv

python rec_user_based_age_itp.py --age_or_itp itp train_test_data/train_data.csv train_test_data/test_data.csv
python rec_user_based_age_itp.py --age_or_itp age train_test_data/train_data.csv train_test_data/test_data.csv
python rec_user_based_age_itp.py --age_or_itp age_and_itp train_test_data/train_data.csv train_test_data/test_data.csv

python rec_content_boosted_item_cf.py train_test_data/train_data.csv train_test_data/test_data.csv
python rec_content_boosted_user_cf.py train_test_data/train_data.csv train_test_data/test_data.csv

python rec_hybrid.py train_test_data/train_data.csv train_test_data/test_data.csv
python rec_hybrid_user_based_cf_age_itp.py --age_or_itp itp train_test_data/train_data.csv train_test_data/test_data.csv
python rec_hybrid_user_based_cf_age_itp.py --age_or_itp age train_test_data/train_data.csv train_test_data/test_data.csv
python rec_hybrid_user_based_cf_age_itp.py --age_or_itp age_and_itp train_test_data/train_data.csv train_test_data/test_data.csv

python ../compare_models.py ../books_recommender/

## Generate recommendations using kfold cross validation
python rec_random_based.py --cross_eval --kfolds 10 train_test_data/ train_test_data/
python rec_popularity_based.py --cross_eval --kfolds 10 train_test_data/ train_test_data/
python rec_item_based_cf.py --cross_eval --kfolds 10 train_test_data/ train_test_data/
python rec_user_based_cf.py --cross_eval --kfolds 10 train_test_data/ train_test_data/
python rec_content_based.py --cross_eval --kfolds 10 train_test_data/ train_test_data/

python rec_user_based_age_itp.py --cross_eval --kfolds 10 --age_or_itp itp train_test_data/ train_test_data/
python rec_user_based_age_itp.py --cross_eval --kfolds 10 --age_or_itp age train_test_data/ train_test_data/
python rec_user_based_age_itp.py --cross_eval --kfolds 10 --age_or_itp age_and_itp train_test_data/ train_test_data/
python rec_content_boosted_item_cf.py --cross_eval --kfolds 10 train_test_data/ train_test_data/
python rec_content_boosted_user_cf.py --cross_eval --kfolds 10 train_test_data/ train_test_data/

python rec_hybrid.py --cross_eval --kfolds 10 train_test_data/ train_test_data/
python rec_hybrid_user_based_cf_age_itp.py --cross_eval --kfolds 10 --age_or_itp itp train_test_data/ train_test_data/
python rec_hybrid_user_based_cf_age_itp.py --cross_eval --kfolds 10 --age_or_itp age train_test_data/ train_test_data/
python rec_hybrid_user_based_cf_age_itp.py --cross_eval --kfolds 10 --age_or_itp age_and_itp train_test_data/ train_test_data/

python ../compare_models.py --kfold ../books_recommender/

## Evaluate recommendations
cd ..
python compare_models.py books_recommender/
python compare_models.py --kfold books_recommender/

## APIs to test train,eval and recommend
python rec_random_based.py --train train_test_data/train_data.csv train_test_data/test_data.csv
python rec_random_based.py --eval train_test_data/train_data.csv train_test_data/test_data.csv
python rec_random_based.py --recommend --user_id 393735.0 train_test_data/train_data.csv train_test_data/test_data.csv
python rec_random_based.py --cross_eval --kfolds 10 train_test_data/ train_test_data/