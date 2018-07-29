## Download dataset from the following urls and place in `data` directory

10000.txt : https://static.turi.com/datasets/millionsong/10000.txt
song_data.csv : https://static.turi.com/datasets/millionsong/song_data.csv

## Data Preprocessing
python data_preprocess.py 
Preprocessed data available in preprocessed_data/

## Split Data into Train and Test
python split_train_test_data.py --users_split --test_size 0.2 --min_no_of_songs 10 preprocessed_data/user_songs.csv
python split_train_test_data.py --kfold_split --kfolds 10 --min_no_of_songs 10 preprocessed_data/user_songs.csv

## Generate recommendations
python rec_random_based.py train_test_data/train_data.csv train_test_data/test_data.csv
python rec_popularity_based.py train_test_data/train_data.csv train_test_data/test_data.csv
python rec_item_based_cf.py train_test_data/train_data.csv train_test_data/test_data.csv
python rec_user_based_cf.py train_test_data/train_data.csv train_test_data/test_data.csv

## Generate recommendations using kfold cross validation
python rec_random_based.py --cross_eval --kfolds 10 train_test_data/ train_test_data/
python rec_popularity_based.py --cross_eval --kfolds 10 train_test_data/ train_test_data/
python rec_item_based_cf.py --cross_eval --kfolds 10 train_test_data/ train_test_data/
python rec_user_based_cf.py --cross_eval --kfolds 10 train_test_data/ train_test_data/

## Evaluate recommendations
cd ..
python compare_models.py songs_recommender/
python compare_models.py --kfold songs_recommender/