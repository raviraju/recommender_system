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

default_configs = [
    {
        "name": "NormalPredictor_Default",
        "algo": NormalPredictor()
    },
    {
        "name": "BaselineOnly_ALS_Default",
        "algo": BaselineOnly(
          bsl_options = {
            "method": "als",
            "n_epochs": 10,
            "reg_u": 15,
            "reg_i": 10
        })
    },
    {
        "name": "BaselineOnly_SGD_Default",
        "algo": BaselineOnly(
          bsl_options = {
            "method": "sgd",
            "n_epochs": 20,
            "reg": 0.02,
            "learning_rate": 0.005
        })
    },
    
    
    
    {
        "name": "Knn_UserBased_Basic_MSD_Default",
        "algo": KNNBasic(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "msd",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_Basic_Cosine_Default",
        "algo": KNNBasic(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "cosine",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_Basic_Pearson_Default",
        "algo": KNNBasic(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "pearson",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    
    
    
    {
        "name": "Knn_UserBased_Means_MSD_Default",
        "algo": KNNWithMeans(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "msd",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_Means_Cosine_Default",
        "algo": KNNWithMeans(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "cosine",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_Means_Pearson_Default",
        "algo": KNNWithMeans(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "pearson",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    
    
    
    {
        "name": "Knn_UserBased_ZScore_MSD_Default",
        "algo": KNNWithZScore(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "msd",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_ZScore_Cosine_Default",
        "algo": KNNWithZScore(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "cosine",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    {
        "name": "Knn_UserBased_ZScore_Pearson_Default",
        "algo": KNNWithZScore(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "pearson",
                "min_support": 1,
                "user_based": True #compute similarities between users
            })
    },
    
    
    
    {
        "name": "Knn_UserBased_Baseline_ALS_Default",
        "algo": KNNBaseline(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "pearson_baseline",
                "shrinkage": 100,
                "min_support": 1,
                "user_based": True #compute similarities between users
            },
            bsl_options = {
                "method": "als",
                "n_epochs": 10,
                "reg_u": 12,
                "reg_i": 5
            })
    },
    {
        "name": "Knn_UserBased_Baseline_SGD_Default",
        "algo": KNNBaseline(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "pearson_baseline",
                "shrinkage": 100,
                "min_support": 1,
                "user_based": True #compute similarities between users
            },
            bsl_options = {
                "method": "sgd",
                "n_epochs": 20,
                "reg": 0.02,
                "learning_rate": 0.005
            })
    },
    
    
    
    {
        "name": "Knn_ItemBased_Basic_MSD_Default",
        "algo": KNNBasic(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "msd",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_Basic_Cosine_Default",
        "algo": KNNBasic(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "cosine",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_Basic_Pearson_Default",
        "algo": KNNBasic(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "pearson",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_Means_MSD_Default",
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
        "name": "Knn_ItemBased_Means_Cosine_Default",
        "algo": KNNWithMeans(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "cosine",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_Means_Pearson_Default",
        "algo": KNNWithMeans(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "pearson",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_ZScore_MSD_Default",
        "algo": KNNWithZScore(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "msd",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_ZScore_Cosine_Default",
        "algo": KNNWithZScore(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "cosine",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_ZScore_Pearson_Default",
        "algo": KNNWithZScore(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "pearson",
                "min_support": 1,
                "user_based": False #compute similarities between items
            })
    },
    {
        "name": "Knn_ItemBased_Baseline_ALS_Default",
        "algo": KNNBaseline(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "pearson_baseline",
                "shrinkage": 100,
                "min_support": 1,
                "user_based": False #compute similarities between items
            },
            bsl_options = {
                "method": "als",
                "n_epochs": 10,
                "reg_u": 12,
                "reg_i": 5
            })
    },
    {
        "name": "Knn_ItemBased_Baseline_SGD_Default",
        "algo": KNNBaseline(
            k = 40,
            min_k = 1,
            sim_options = {
                "name": "pearson_baseline",
                "shrinkage": 100,
                "min_support": 1,
                "user_based": False #compute similarities between items
            },
            bsl_options = {
                "method": "sgd",
                "n_epochs": 20,
                "reg": 0.02,
                "learning_rate": 0.005
            })
    },
    {
        "name": "SVD_unbiased_Default",
        "algo": SVD(
            n_factors=100, n_epochs=20, biased=False, 
            init_mean=0, init_std_dev=0.1, 
            lr_all=0.005, reg_all=0.02, 
            lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
            reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
            random_state=123, verbose=True)
    },
    {
        "name": "SVD_biased_Default",
        "algo": SVD(
            n_factors=100, n_epochs=20, biased=True, 
            init_mean=0, init_std_dev=0.1, 
            lr_all=0.005, reg_all=0.02, 
            lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
            reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
            random_state=123, verbose=True)
    },
    {
        "name": "SVDpp_biased_Default",
        "algo": SVDpp(
            n_factors=20, n_epochs=20,
            init_mean=0, init_std_dev=0.1, 
            lr_all=0.007, reg_all=0.02, 
            lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
            reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
            random_state=123, verbose=True)
    }
]

'''
default_configs = [
    {
        "name": "NMF_unbiased_Default",
        "algo": NMF(
            n_factors=15, n_epochs=50, biased=False, 
            
            reg_pu=0.06, reg_qi=0.06, 
            
            init_low=0, init_high=1,
            random_state=123, verbose=True)
    },
    {
        "name": "NMF_biased_Default",
        "algo": NMF(
            n_factors=15, n_epochs=50, biased=True, 
            
            reg_pu=0.06, reg_qi=0.06,
            
            reg_bu=0.02, reg_bi=0.02,
            lr_bu=0.005, lr_bi=0.005,
            
            init_low=0, init_high=1,
            random_state=123, verbose=True)
    },
]
'''

def main():
    cur_file_name = __file__
    pickle_file_name = cur_file_name.replace('.py', '.pickle')
    pickle_out = open(pickle_file_name, "wb")
    pickle.dump(default_configs, pickle_out)
    pickle_out.close()
    
if __name__ == '__main__':
    main()
