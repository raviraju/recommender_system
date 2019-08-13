# About

Articles Recommender for [Articles sharing and reading from CI&T DeskDrop](https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop)

# Fetch Data

Download data and place in `data/` directory 
* [shared_articles.csv](https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop/downloads/shared_articles.csv/5)
* [users_interactions.csv](https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop/downloads/users_interactions.csv/5)

# Notebooks for Analysis of Data
* [Exploratory Data Analysis of User_Interactions](notebooks/EDA_User_Interactions.ipynb)
* [Exploratory Data Analysis of Articles_Shared](notebooks/EDA_Articles_Shared.ipynb)
* [Exploring Topic Modelling on Articles_Shared](notebooks/Topic_Modelling_on_Shared_Articles.ipynb)

# 

# Generate Train and Test Datasets for different Time Periods

Use [Prepare_Data.ipynb](notebooks/Prepare_Data.ipynb) in `notebooks/` to generate train and test datasets for various time periods

```
ls -lR train_test_datasets/
train_test_datasets/:
total 16
drwxr-xr-x 2 ravi ravi 4096 Aug 12 21:06 dataset_1
drwxr-xr-x 2 ravi ravi 4096 Aug 12 21:06 dataset_2
drwxr-xr-x 2 ravi ravi 4096 Aug 12 21:06 dataset_3
drwxr-xr-x 2 ravi ravi 4096 Aug 12 21:06 dataset_4

train_test_datasets/dataset_1:
total 524
-rw-r--r-- 1 ravi ravi 508568 Aug 12 21:06 test_2016Q2.csv
-rw-r--r-- 1 ravi ravi  22554 Aug 12 21:06 train_2016Q1.csv

train_test_datasets/dataset_2:
total 888
-rw-r--r-- 1 ravi ravi 376792 Aug 12 21:06 test_2016Q3.csv
-rw-r--r-- 1 ravi ravi 529674 Aug 12 21:06 train_2016Q1_2016Q2.csv

train_test_datasets/dataset_3:
total 1076
-rw-r--r-- 1 ravi ravi 200270 Aug 12 21:06 test_2016Q4.csv
-rw-r--r-- 1 ravi ravi 900120 Aug 12 21:06 train_2016Q1_2016Q2_2016Q3.csv

train_test_datasets/dataset_4:
total 1176
-rw-r--r-- 1 ravi ravi  102952 Aug 12 21:06 test_2017Q1.csv
-rw-r--r-- 1 ravi ravi 1093990 Aug 12 21:06 train_2016Q1_2016Q2_2016Q3_2016Q4.csv
```

# Run Train, Eval and Recommendation for each DataSet
1. Create the following directories to store models produced from each dataset
    ```
    mkdir models_datasets
    mkdir models_datasets/dataset_1
    mkdir models_datasets/dataset_2
    mkdir models_datasets/dataset_3
    mkdir models_datasets/dataset_4
    ```
2. Move generated model and evaluation results in `models/` into 'models_datasets/dataset_1, models_datasets/dataset_2...'
    ```
    python rec_random_based.py train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv --meta_data data/shared_articles.csv
    python rec_popularity_based.py train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv --meta_data data/shared_articles.csv
    python rec_user_based_cf.py train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv --meta_data data/shared_articles.csv
    python rec_item_based_cf.py train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv --meta_data data/shared_articles.csv
    python rec_content_based.py train_test_datasets/dataset_1/train_2016Q1.csv train_test_datasets/dataset_1/test_2016Q2.csv data/shared_articles.csv

    mv models models_datasets/dataset_1/

    python rec_random_based.py train_test_datasets/dataset_2/train_2016Q1_2016Q2.csv train_test_datasets/dataset_2/test_2016Q3.csv --meta_data data/shared_articles.csv
    python rec_popularity_based.py train_test_datasets/dataset_2/train_2016Q1_2016Q2.csv train_test_datasets/dataset_2/test_2016Q3.csv --meta_data data/shared_articles.csv
    python rec_user_based_cf.py train_test_datasets/dataset_2/train_2016Q1_2016Q2.csv train_test_datasets/dataset_2/test_2016Q3.csv --meta_data data/shared_articles.csv
    python rec_item_based_cf.py train_test_datasets/dataset_2/train_2016Q1_2016Q2.csv train_test_datasets/dataset_2/test_2016Q3.csv --meta_data data/shared_articles.csv
    python rec_content_based.py train_test_datasets/dataset_2/train_2016Q1_2016Q2.csv train_test_datasets/dataset_2/test_2016Q3.csv data/shared_articles.csv

    mv models models_datasets/dataset_2/

    python rec_random_based.py train_test_datasets/dataset_3/train_2016Q1_2016Q2_2016Q3.csv train_test_datasets/dataset_3/test_2016Q4.csv --meta_data data/shared_articles.csv
    python rec_popularity_based.py train_test_datasets/dataset_3/train_2016Q1_2016Q2_2016Q3.csv train_test_datasets/dataset_3/test_2016Q4.csv --meta_data data/shared_articles.csv
    python rec_user_based_cf.py train_test_datasets/dataset_3/train_2016Q1_2016Q2_2016Q3.csv train_test_datasets/dataset_3/test_2016Q4.csv --meta_data data/shared_articles.csv
    python rec_item_based_cf.py train_test_datasets/dataset_3/train_2016Q1_2016Q2_2016Q3.csv train_test_datasets/dataset_3/test_2016Q4.csv --meta_data data/shared_articles.csv
    python rec_content_based.py train_test_datasets/dataset_3/train_2016Q1_2016Q2_2016Q3.csv train_test_datasets/dataset_3/test_2016Q4.csv data/shared_articles.csv

    mv models models_datasets/dataset_3/

    python rec_random_based.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv --meta_data data/shared_articles.csv
    python rec_popularity_based.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv --meta_data data/shared_articles.csv
    python rec_user_based_cf.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv --meta_data data/shared_articles.csv
    python rec_item_based_cf.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv --meta_data data/shared_articles.csv
    python rec_content_based.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv data/shared_articles.csv

    mv models models_datasets/dataset_4/
    ```

# Compare Evaluation Results
* [Compare Models with various evaluation metrics](notebooks/Analyse_Evaluation.ipynb)

# Individual APIs to experiment with train, evaluate and recommend

    python rec_user_based_cf.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv --meta_data data/shared_articles.csv --train
    python rec_user_based_cf.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv --meta_data data/shared_articles.csv --eval
    python rec_user_based_cf.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv --meta_data data/shared_articles.csv --recommend --user_id 5974049584912996673

    python rec_item_based_cf.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv --meta_data data/shared_articles.csv --train
    python rec_item_based_cf.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv --meta_data data/shared_articles.csv --eval
    python rec_item_based_cf.py train_test_datasets/dataset_4/train_2016Q1_2016Q2_2016Q3_2016Q4.csv train_test_datasets/dataset_4/test_2017Q1.csv --meta_data data/shared_articles.csv --recommend --user_id 6756039155228175109
