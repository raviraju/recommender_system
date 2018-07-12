## Generate Train, Validation and Test Datasets

`python split_train_test_data.py --kfolds_split --no_of_kfolds 10 --min_no_of_items 10 --validation_size 0.2 preprocessed_data/latest_rating_new.csv learner_id media_id like_rating > preprocessed_data/kfolds_split.txt`

## HyperParam search using Validation Dataset

Configure search parameters in `param_search/experiments.py` and generate a pickle file `experiments.pickle`
`python hyper_param_search.py param_search/experiments.pickle`

Configure best hyperparameters in `configs/tuned_configs_new.py` and generate pickle file `tuned_configs_new.pickle`


## Generate recommendations for Validation dataset using best hyperparameters

`python generate_recommendations.py configs/tuned_configs_new.pickle --validation`
Summary of all recommendations are found in `model_validation/summary_results.json`


## Generate recommendations for Testing dataset using best hyperparameters

`python generate_recommendations.py configs/tuned_configs_new.pickle --testing`
Summary of all recommendations are found in `model_testing/summary_results.json`


## Compare rmse of validation and testing dataset

Use `merge validation and testing recommendations.ipynb` to generate `validation_testing_results.csv`

## Generate top N recommendations for Anti Testset(Predict ratings for all pairs (u, i) that are NOT in the training)

`python generate_top_n_recommendations.py preprocessed_data/latest_rating_new.csv configs/tuned_configs_new.pickle`

# Hybrid Recommenders

## Generate data for hybrid recommender

`python hybrid_recommender_generate_data.py configs/tuned_configs_new.pickle model_testing/`


## Search for hyper parameters of hybrid recommender

`python hybrid_recommender_search.py`

## Use best parameters of hybrid recommender to generate hybrid recommendations

`python hybrid_recommender_generate_recommendations.py`

## Generate top N recommendations using hybrid recommenders

`python hybrid_recommender_generate_top_n_recs.py configs/tuned_configs_new.pickle top_n_recs/`
