## Place the following data files in `data/`
`bc.demographics.csv  bookclub_events.csv  BOOKINFORMATION.csv  BOOKMETA.csv  demograph.csv`

## Data Preprocessing

### INSTALL [Google Cloud Translation API](https://cloud.google.com/translate/docs/reference/libraries#client-libraries-install-python)
`pip install --upgrade google-cloud-translate`

`cd data_preprocessing/`

Obtain service account key in json file and place in directory `data_preprocessing/Translate-df698c42870a.json`

### Test google translation using 
`python test_google_translation.py`

### Store all korean fields to be converted in json files in korean_dir/
Load korean fields from ../data/BOOKMETA.csv into korean_dir
`python book_meta.py --store`

### Obtain google translations for json files in korean_dir, update translations in translations.json
`python translate_korean_fields.py`

### Convert korean fields using google translations in translations.json to produce ../data/T_BOOKMETA.csv
`python book_meta.py --convert`

### Merge data/T_BOOKMETA.csv and data/BOOKINFORMATION.csv to produce data/BOOKINFORMATION_META.csv
`python merge_book_info_meta.py`

### Perform Data Cleaning and Preprocessing
`python preprocess_metadata.py`

### Split Data into Train and Test
```
python split_train_test_data.py --users_split --min_no_of_books 10 --test_size 0.2 preprocessed_metadata/learner_books_info_close_min_10_events.csv
python split_train_test_data.py --kfold_split --kfolds 10 --min_no_of_books 10 preprocessed_metadata/learner_books_info_close_min_10_events.csv
```

## Generate recommendations
```
python rec_random_based.py train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_popularity_based.py train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_item_based_cf.py train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_user_based_cf.py train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_content_based.py train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5

python rec_user_based_age_itp.py --age_or_itp itp train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_user_based_age_itp.py --age_or_itp age train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_user_based_age_itp.py --age_or_itp age_and_itp train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5

python rec_content_boosted_item_cf.py train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_content_boosted_user_cf.py train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_hybrid_user_based_cf_age_itp.py --age_or_itp itp train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_hybrid_user_based_cf_age_itp.py --age_or_itp age train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_hybrid_user_based_cf_age_itp.py --age_or_itp age_and_itp train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
```
### Hybrid Recommenders
```
python rec_hybrid.py train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5 data_preprocessing/preprocessed_metadata/learner_books_info_close_min_10_events.csv
```

## Generate recommendations using kfold cross validation
```
python rec_random_based.py --cross_eval --kfolds 10 train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
python rec_popularity_based.py --cross_eval --kfolds 10 train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
python rec_item_based_cf.py --cross_eval --kfolds 10 train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
python rec_user_based_cf.py --cross_eval --kfolds 10 train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
python rec_content_based.py --cross_eval --kfolds 10 train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5

python rec_user_based_age_itp.py --cross_eval --kfolds 10 --age_or_itp itp train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
python rec_user_based_age_itp.py --cross_eval --kfolds 10 --age_or_itp age train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
python rec_user_based_age_itp.py --cross_eval --kfolds 10 --age_or_itp age_and_itp train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
python rec_content_boosted_item_cf.py --cross_eval --kfolds 10 train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
python rec_content_boosted_user_cf.py --cross_eval --kfolds 10 train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
python rec_hybrid_user_based_cf_age_itp.py --cross_eval --kfolds 10 --age_or_itp itp train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
python rec_hybrid_user_based_cf_age_itp.py --cross_eval --kfolds 10 --age_or_itp age train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
python rec_hybrid_user_based_cf_age_itp.py --cross_eval --kfolds 10 --age_or_itp age_and_itp train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
```
### Hybrid Recommenders
```
python rec_hybrid.py --cross_eval --kfolds 10 train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5 data_preprocessing/preprocessed_metadata/learner_books_info_close_min_10_events.csv
```

## Evaluate recommendations
```
cd ../binary_recommender
python lib/compare_models.py books_recommender/
python lib/compare_models.py --kfold books_recommender/
python lib/compare_models.py --kfold books_recommender/ --hybrid_only
```

## APIs to test train,eval and recommend
```
python rec_item_based_cf.py --train train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_item_based_cf.py --eval train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_item_based_cf.py --recommend --user_id 239525.0 train_test_data/users_split/train_data.csv train_test_data/users_split/test_data.csv hold_last_n 5
python rec_item_based_cf.py --cross_eval --kfolds 10 train_test_data/kfold_split/ train_test_data/kfold_split/ hold_last_n 5
```