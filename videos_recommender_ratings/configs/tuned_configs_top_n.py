import pickle

from surprise import BaselineOnly
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp

tuned_configs = [
    {
        "name": "BaselineOnly_SGD_Tuned",
        "algo": BaselineOnly(
          bsl_options = {
            "method": "sgd",
            "n_epochs": 10,
            "reg": 0.5,
            "learning_rate": 0.1
        })
    },
    # {
        # "name": "BaselineOnly_ALS_Tuned",
        # "algo": BaselineOnly(
          # bsl_options = {
            # "method": "als",
            # "n_epochs": 10,
            # "reg_u": 5,
            # "reg_i": 30
        # })
    # },
    ######################################################################
    # {
        # "name": "Knn_UserBased_ZScore_MSD_Tuned",
        # "algo": KNNWithZScore(
            # k = 500,
            # min_k = 15,
            # sim_options = {
                # "name": "msd",
                # "min_support": 1,
                # "user_based": True #compute similarities between users
            # })
    # },
    # {
        # "name": "Knn_ItemBased_ZScore_MSD_Tuned",
        # "algo": KNNWithZScore(
            # k = 100,
            # min_k = 1,
            # sim_options = {
                # "name": "msd",
                # "min_support": 1,
                # "user_based": False #compute similarities between items
            # })
    # },
    ######################################################################
    {
        "name": "Knn_UserBased_Baseline_SGD_Tuned",
        "algo": KNNBaseline(
            k = 1000,
            min_k = 50,
            sim_options = {
                "name": "pearson_baseline",
                "shrinkage": 0,
                "min_support": 1,
                "user_based": True #compute similarities between users
            },
            bsl_options = {
                "method": "sgd",
                "n_epochs": 10,
                "reg": 0.5,
                "learning_rate": 0.1
            })
    },
    # {
        # "name": "Knn_UserBased_Baseline_ALS_Tuned",
        # "algo": KNNBaseline(
            # k = 1000,
            # min_k = 50,
            # sim_options = {
                # "name": "pearson_baseline",
                # "shrinkage": 0,
                # "min_support": 1,
                # "user_based": True #compute similarities between users
            # },
            # bsl_options = {
                # "method": "als",
                # "n_epochs": 10,
                # "reg_u": 5,
                # "reg_i": 30
            # })
    # },
    {
        "name": "Knn_ItemBased_Baseline_SGD_Tuned",
        "algo": KNNBaseline(
            k = 150,
            min_k = 100,
            sim_options = {
                "name": "pearson_baseline",
                "shrinkage": 100000,
                "min_support": 5,
                "user_based": False #compute similarities between items
            },
            bsl_options = {
                "method": "sgd",
                "n_epochs": 10,
                "reg": 0.5,
                "learning_rate": 0.1
            })
    },
    # {
        # "name": "Knn_ItemBased_Baseline_ALS_Tuned",
        # "algo": KNNBaseline(
            # k = 100,
            # min_k = 50,
            # sim_options = {
                # "name": "pearson_baseline",
                # "shrinkage": 100000,
                # "min_support": 5,
                # "user_based": False #compute similarities between items
            # },
            # bsl_options = {
                # "method": "als",
                # "n_epochs": 10,
                # "reg_u": 5,
                # "reg_i": 30
            # })
    # },
    ######################################################################
    # {
        # "name": "SVD_unbiased_Tuned",
        # "algo": SVD(
            # n_factors=10, n_epochs=50, biased=False,
            # lr_all=0.01, reg_all=0.01,                
            # random_state=123)
    # },
    {
        "name": "SVD_biased_Tuned",
        "algo": SVD(
            n_factors=10, n_epochs=10, biased=True,
            lr_all=0.01, reg_all=0.01,                
            random_state=123)
    },
    {
        "name": "SVDpp_biased_Tuned",
        "algo": SVDpp(
            n_factors=10, n_epochs=10,
            lr_all=0.01, reg_all=0.01,
            random_state=123, verbose=True)
    }
]

def main():
    cur_file_name = __file__
    pickle_file_name = cur_file_name.replace('.py', '.pickle')
    pickle_out = open(pickle_file_name, "wb")
    pickle.dump(tuned_configs, pickle_out)
    pickle_out.close()
    
if __name__ == '__main__':
    main()
