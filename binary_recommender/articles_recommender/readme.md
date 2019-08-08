# About

Articles Recommender for [Articles sharing and reading from CI&T DeskDrop](https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop)

# Data

Download data and place in `data/` directory 
* [shared_articles.csv](https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop/downloads/shared_articles.csv/5)
* [users_interactions.csv](https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop/downloads/users_interactions.csv/5)

# Generate Train and Test Datasets

Use Prepare Data.ipynb in `notebooks/` to generate train and test datasets for various time periods

```
ls -lR train_test_datasets/
train_test_datasets/:
total 16
drwxr-xr-x 2 ravi ravi 4096 Aug  6 20:27 dataset_1
drwxr-xr-x 2 ravi ravi 4096 Aug  6 20:27 dataset_2
drwxr-xr-x 2 ravi ravi 4096 Aug  6 20:27 dataset_3
drwxr-xr-x 2 ravi ravi 4096 Aug  6 20:27 dataset_4

train_test_datasets/dataset_1:
total 524
-rw-r--r-- 1 ravi ravi 508568 Aug  6 20:27 test_2016Q2.csv
-rw-r--r-- 1 ravi ravi  22554 Aug  6 20:27 train_2016Q1.csv

train_test_datasets/dataset_2:
total 888
-rw-r--r-- 1 ravi ravi 376792 Aug  6 20:27 test_2016Q3.csv
-rw-r--r-- 1 ravi ravi 529674 Aug  6 20:27 train_2016Q1_2016Q2.csv

train_test_datasets/dataset_3:
total 1076
-rw-r--r-- 1 ravi ravi 200270 Aug  6 20:27 test_2016Q4.csv
-rw-r--r-- 1 ravi ravi 900120 Aug  6 20:27 train_2016Q1_2016Q2_2016Q3.csv

train_test_datasets/dataset_4:
total 1176
-rw-r--r-- 1 ravi ravi  102952 Aug  6 20:27 test_2017Q1.csv
-rw-r--r-- 1 ravi ravi 1093990 Aug  6 20:27 train_2016Q1_2016Q2_2016Q3_2016Q4.csv
```
# Run Train, Eval and Recommendation
```
python rec_random_based.py  train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv
python rec_popularity_based.py train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv
python rec_user_based_cf.py train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv
python rec_item_based_cf.py train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv

python rec_random_based.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv
python rec_popularity_based.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv
python rec_user_based_cf.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv
python rec_item_based_cf.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv
```

# Individual APIs for train, recommend and evaluate
```
python rec_random_based.py  train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv --train
python rec_random_based.py  train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv --recommend --user_id -9223121837663643404
python rec_random_based.py  train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv --eval
```