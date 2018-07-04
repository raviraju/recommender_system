import pickle
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline

from surprise import SVD
from surprise import SVDpp
from surprise import NMF

tuned_configs = [
    {
        "name": "NormalPredictor_Tuned",
        "algo": NormalPredictor()
    },
    {
        "name": "BaselineOnly_ALS_Tuned",#0.643461
        "algo": BaselineOnly(
          bsl_options = {
            "method": "als",
            "n_epochs": 10,
            "reg_u": 2.7,
            "reg_i": 55
        })
    },
    {
        "name": "BaselineOnly_SGD_Tuned",#0.64556
        "algo": BaselineOnly(
          bsl_options = {
            "method": "sgd",
            "n_epochs": 12,
            "reg": 0.09,
            "learning_rate": 0.012
        })
    },
    
    
    
    {
        "name": "Knn_UserBased_Basic_MSD_Tuned",#0.698865
        "algo": KNNBasic(
            k = 50,
            min_k = 10,
            sim_options = {
                "name": "msd",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_Basic_Cosine_Tuned",#0.772782
        "algo": KNNBasic(
            k = 100,
            min_k = 19,
            sim_options = {
                "name": "cosine",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_Basic_Pearson_Tuned",#0.770547
        "algo": KNNBasic(
            k = 50,
            min_k = 18,
            sim_options = {
                "name": "pearson",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    
    
    
    {
        "name": "Knn_UserBased_Means_MSD_Tuned",#0.642110
        "algo": KNNWithMeans(
            k = 50,
            min_k = 18,
            sim_options = {
                "name": "msd",
                "min_support": 2,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_Means_Cosine_Tuned",#0.646371
        "algo": KNNWithMeans(
            k = 50,
            min_k = 30,
            sim_options = {
                "name": "cosine",
                "min_support": 5,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_Means_Pearson_Tuned",#0.643182
        "algo": KNNWithMeans(
            k = 100,
            min_k = 18,
            sim_options = {
                "name": "pearson",
                "min_support": 2,
                "user_based": True #compute similarities between users
            })
    },
    
    
    
    {
        "name": "Knn_UserBased_ZScore_MSD_Tuned",#0.641941
        "algo": KNNWithZScore(
            k = 100,
            min_k = 18,
            sim_options = {
                "name": "msd",
                "min_support": 2,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_ZScore_Cosine_Tuned",#0.647038
        "algo": KNNWithZScore(
            k = 50,
            min_k = 35,
            sim_options = {
                "name": "cosine",
                "min_support": 7,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_ZScore_Pearson_Tuned",#0.642795
        "algo": KNNWithZScore(
            k = 100,
            min_k = 11,
            sim_options = {
                "name": "pearson",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    
    
    {
        "name": "Knn_UserBased_Baseline_ALS_Tuned",#500, 1000 - 0.633908, #1000, 10000 - 0.63365, #1000, 100000 - 0.63364
        "algo": KNNBaseline(
            k = 1000,
            min_k = 20,
            sim_options = {
                "name": "pearson_baseline",
                "shrinkage": 100000,
                "min_support": 1,
                "user_based": True #compute similarities between users
            },
            bsl_options = {
                "method": "als",
                "n_epochs": 10,
                "reg_u": 2.7,
                "reg_i": 55
            })
    },
    {
        "name": "Knn_UserBased_Baseline_SGD_Tuned",#500, 1000 - 0.635857, #1000, 10000 - 0.63562, #1000, 100000 - 0.63561
        "algo": KNNBaseline(
            k = 1000,
            min_k = 20,
            sim_options = {
                "name": "pearson_baseline",
                "shrinkage": 100000,
                "min_support": 1,
                "user_based": True #compute similarities between users
            },
            bsl_options = {
                "method": "sgd",
                "n_epochs": 12,
                "reg": 0.09,
                "learning_rate": 0.012
            })
    },
    
    
    {
        "name": "Knn_ItemBased_Baseline_ALS_Tuned",#0.63341
        "algo": KNNBaseline(
            k = 50,
            min_k = 10,
            sim_options = {
                "name": "pearson_baseline",
                "shrinkage": 1000,
                "min_support": 4,
                "user_based": False #compute similarities between items
            },
            bsl_options = {
                "method": "als",
                "n_epochs": 10,
                "reg_u": 2.7,
                "reg_i": 55
            })
    },
    {
        "name": "Knn_ItemBased_Baseline_SGD_Tuned",#0.63624
        "algo": KNNBaseline(
            k = 50,
            min_k = 10,
            sim_options = {
                "name": "pearson_baseline",
                "shrinkage": 1000,
                "min_support": 4,
                "user_based": False #compute similarities between items
            },
            bsl_options = {
                "method": "sgd",
                "n_epochs": 12,
                "reg": 0.09,
                "learning_rate": 0.012
            })
    },
    
    
    
    {
        "name": "Knn_ItemBased_Basic_MSD_Tuned",#0.64969
        "algo": KNNBasic(
            k = 40,
            min_k = 4,
            sim_options = {
                "name": "msd",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_Basic_Cosine_Tuned",#0.65188
        "algo": KNNBasic(
            k = 70,
            min_k = 4,
            sim_options = {
                "name": "cosine",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_Basic_Pearson_Tuned",#0.66571
        "algo": KNNBasic(
            k = 60,
            min_k = 4,
            sim_options = {
                "name": "pearson",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    
    
    {
        "name": "Knn_ItemBased_Means_MSD_Tuned",#0.65817
        "algo": KNNWithMeans(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "msd",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_Means_Cosine_Tuned",#0.66001
        "algo": KNNWithMeans(
            k = 70,
            min_k = 1,
            sim_options = {
                "name": "cosine",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_Means_Pearson_Tuned",#0.67579
        "algo": KNNWithMeans(
            k = 60,
            min_k = 4,
            sim_options = {
                "name": "pearson",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    
    
    {
        "name": "Knn_ItemBased_ZScore_MSD_Tuned",#0.65958
        "algo": KNNWithZScore(
            k = 40,
            min_k = 2,
            sim_options = {
                "name": "msd",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_ZScore_Cosine_Tuned",#0.6612
        "algo": KNNWithZScore(
            k = 70,
            min_k = 2,
            sim_options = {
                "name": "cosine",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_ZScore_Pearson_Tuned",#0.67621
        "algo": KNNWithZScore(
            k = 60,
            min_k = 4,
            sim_options = {
                "name": "pearson",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "SVD_unbiased_Tuned",#0.65888
        "algo": SVD(
            n_factors=12, n_epochs=50, biased=False,
            lr_all=0.01, reg_all=0.1, 
            random_state=123, verbose=True)
    },
    {
        "name": "SVD_biased_Tuned",
        "algo": SVD(
            n_factors=500, n_epochs=150, biased=True,
            lr_all=0.01, reg_all=0.1,                
            random_state=123, verbose=True)
    },
    {
        "name": "SVDpp_biased_Tuned",
        "algo": SVDpp(
            n_factors=20, n_epochs=20,
            lr_all=0.007, reg_all=0.02,
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
