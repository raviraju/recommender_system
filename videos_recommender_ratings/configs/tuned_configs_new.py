import pickle

from surprise import BaselineOnly
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp

tuned_configs = [
    {
        "name": "BaselineOnly_SGD_Tuned",#0.70070116268
        "algo": BaselineOnly(
          bsl_options = {
            "method": "sgd",
            "n_epochs": 5,
            "reg": 0.5,
            "learning_rate": 0.1
        })
    },     
    ######################################################################
    {
        "name": "Knn_UserBased_ZScore_MSD_Tuned",#0.719219724908
        "algo": KNNWithZScore(
            k = 100,
            min_k = 15,
            sim_options = {
                "name": "msd",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_ItemBased_ZScore_MSD_Tuned",#0.7365081822837258
        "algo": KNNWithZScore(
            k = 100,
            min_k = 1,
            sim_options = {
                "name": "msd",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    ######################################################################
    {
        "name": "Knn_UserBased_Baseline_SGD_Tuned",#0.695556842098
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
                "n_epochs": 5,
                "reg": 0.5,
                "learning_rate": 0.1
            })
    },    
    {
        "name": "Knn_ItemBased_Baseline_SGD_Tuned",#0.70070116268
        "algo": KNNBaseline(
            k = 10,
            min_k = 20,
            sim_options = {
                "name": "pearson_baseline",
                "shrinkage": 0,
                "min_support": 1,
                "user_based": False #compute similarities between items
            },
            bsl_options = {
                "method": "sgd",
                "n_epochs": 5,
                "reg": 0.5,
                "learning_rate": 0.1
            })
    },
    ######################################################################
    {
        "name": "SVD_biased_Tuned",#0.703201233354
        "algo": SVD(
            n_factors=150, n_epochs=10, biased=True,
            lr_all=0.1, reg_all=0.1,                
            random_state=123)
    },
    {
        "name": "SVDpp_biased_Tuned",#0.712759271293
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
