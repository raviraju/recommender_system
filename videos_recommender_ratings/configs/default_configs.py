import pickle
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline

def main():
    configs = [
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
        }
    ]
    cur_file_name = __file__
    pickle_file_name = cur_file_name.replace('.py', '.pickle')
    pickle_out = open(pickle_file_name, "wb")
    pickle.dump(configs, pickle_out)
    pickle_out.close()
    
if __name__ == '__main__':
    main()