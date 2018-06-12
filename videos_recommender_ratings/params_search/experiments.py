import pickle
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline

def main():
    experiments = [
        ###########################################Baseline Rating Models###########################################
        #hyper_param_search_results_baselineonly_als
        {
            "results_dir_name" : "hyper_param_search_experiments/hyper_param_search_results_baselineonly_als",
            "algos" : [BaselineOnly],
            "param_grid" : {
                "bsl_options": {
                    "method": ["als"],
                    "n_epochs": [10, 40, 50, 60, 80, 100],#[56, 57],#[57, 58, 59], #[58, 60, 62, 65],#[50, 55, 60],
                    "reg_u": [2.6, 2.7, 2.8],#[2.3, 2.6, 2.7],#[2.7, 2.8, 2.9],#[1.9, 2, 2.2, 2.8],#[1.4, 1.5, 1.8, 2, 5],
                    "reg_i": [49, 50, 51, 52]#[10, 30, 50, 54]#[55, 56, 57]#[57, 58, 59]#[60, 65, 70, 75]#[70, 80, 85, 90, 100]
                }
            }            
        },
      
        #hyper_param_search_results_baselineonly_sgd
        {
            "results_dir_name" : "hyper_param_search_experiments/hyper_param_search_results_baselineonly_sgd",
            "algos" : [BaselineOnly],
            "param_grid" : {
                "bsl_options": {
                    "method": ["sgd"],
                    "n_epochs": [9, 10, 11, 12, 13],#[7, 8, 9, 10],#[4, 6, 8, 12],#[5, 8, 12], #[10, 15, 25],#[20, 50, 100],
                    "reg": [0.07, 0.08, 0.09, 0.1, 0.12],#[0.09, 0.1, 0.15],#[0.08, 0.1, 0.3], #[0.04, 0.06, 0.1, 0.5],#[0.01, 0.05, 1, 2],
                    "learning_rate": [0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015]#[0.015, 0.02, 0.025]#[0.01, 0.02, 0.04]#[0.03, 0.06, 0.09]#[0.002, 0.02]#[0.001, 0.01, 0.1, 1]
                }
            }     
        },
        
        ###########################################KNN UserBased CF Rating Models###########################################
        #hyper_param_search_results_knns user_based
        {
            "results_dir_name" : "hyper_param_search_experiments/user_based_hyper_param_search_results_knns",
            "algos" : [KNNBasic, KNNWithMeans, KNNWithZScore],
            "param_grid" : {
                #(max) number of neighbors
                "k": [5, 10, 50, 100, 500],
                #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
                "min_k" : [1, 5, 8, 9, 10, 11, 12, 18, 19, 20, 23, 25, 30, 35, 50, 100], #[1, 5, 10, 20, 50, 100], 
                "sim_options": {
                    "name": ["msd", "cosine", "pearson"],
                    "min_support": [1, 2, 5, 7, 10],# if number of common items < min_support, sim(u,v) = 0
                    "user_based": [True]
                }
            }
        },

        #hyper_param_search_results_knn_baseline_als user_based (1 - 31sec) (4*5*4*4=320 - 1.5323    hr)
        {
            "results_dir_name" : "hyper_param_search_experiments/user_based_hyper_param_search_results_knn_baseline_als1",
            "algos" : [KNNBaseline],
            "param_grid" : {
                #(max) number of neighbors
                "k": [10, 1000, 1500, 2000],
                #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
                "min_k" : [1, 5, 10, 20, 50],       
                "sim_options": {
                    "name": ["pearson_baseline"],
                    "shrinkage": [0, 1000, 10000, 100000],  # 0 : no shrinkage
                    "min_support": [1, 2, 5, 10],# if number of common items < min_support, sim(u,v) = 0
                    "user_based": [True]
                },
                "bsl_options": {
                    "method": ["als"],
                    "n_epochs": [10],
                    "reg_u": [2.7],
                    "reg_i": [55],
                }
            }
        },             

        #hyper_param_search_results_knn_baseline_sgd user_based (1 - 31sec) (4*5*4*4=320 - 1.5361    hr)
        {
            "results_dir_name" : "hyper_param_search_experiments/user_based_hyper_param_search_results_knn_baseline_sgd1",
            "algos" : [KNNBaseline],
            "param_grid" : {
                #(max) number of neighbors
                "k": [10, 1000, 1500, 2000],
                #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
                "min_k" : [1, 5, 10, 20, 50],       
                "sim_options": {
                    "name": ["pearson_baseline"],
                    "shrinkage": [0, 1000, 10000, 100000],  # 0 : no shrinkage
                    "min_support": [1, 2, 5, 10],# if number of common items < min_support, sim(u,v) = 0
                    "user_based": [True]
                },
                "bsl_options": {
                    "method": ["sgd"],            
                    "n_epochs": [12],
                    "reg": [0.09],
                    "learning_rate" : [0.012]
                }
            }
        },
        
        ###########################################KNN ItemBased CF Rating Models###########################################
        #hyper_param_search_results_knn_baseline_als item_based (1 - 16 sec) (5*5*4*7=700 1.3986    hr)
        {
            "results_dir_name" : "hyper_param_search_experiments/item_based_hyper_param_search_results_knn_baseline_als1",
            "algos" : [KNNBaseline],
            "param_grid" : {
                #(max) number of neighbors
                "k": [5, 50, 100, 500, 1000],
                #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
                "min_k" : [1, 5, 10, 20, 50],       
                "sim_options": {
                    "name": ["pearson_baseline"],
                    "shrinkage": [0, 100, 1000, 10000],  # 0 : no shrinkage
                    "min_support": [1, 2, 4, 5, 6, 10, 12],# if number of common items < min_support, sim(u,v) = 0
                    "user_based": [False]
                },
                "bsl_options": {
                    "method": ["als"],
                    "n_epochs": [10],
                    "reg_u": [2.7],
                    "reg_i": [55],
                }
            }
        },   
                
        #hyper_param_search_results_knn_baseline_sgd item_based (1 - 16 sec) (5*5*4*7=700 1.3909    hr)
        {
            "results_dir_name" : "hyper_param_search_experiments/item_based_hyper_param_search_results_knn_baseline_sgd1",
            "algos" : [KNNBaseline],
            "param_grid" : {
                #(max) number of neighbors
                "k": [5, 50, 100, 500, 1000],
                #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
                "min_k" : [1, 5, 10, 20, 50],       
                "sim_options": {
                    "name": ["pearson_baseline"],
                    "shrinkage": [0, 100, 1000, 10000],  # 0 : no shrinkage
                    "min_support": [1, 2, 4, 5, 6, 10, 12],# if number of common items < min_support, sim(u,v) = 0
                    "user_based": [False]
                },
                "bsl_options": {
                    "method": ["sgd"],            
                    "n_epochs": [12],
                    "reg": [0.09],
                    "learning_rate" : [0.012]
                }
            }
        },
        
        #hyper_param_search_results_knns item_based 1-2min 8*9*3*3-648*3-1944 - 3.8878    hr
        {
            "results_dir_name" : "hyper_param_search_experiments/item_based_hyper_param_search_results_knns1",
            "algos" : [KNNBasic, KNNWithMeans, KNNWithZScore],
            "param_grid" : {
                #(max) number of neighbors
                "k": [5, 10, 30, 40, 50, 60, 70, 100],
                #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
                "min_k" : [1, 2, 4, 5, 6, 7, 10, 50, 100],#[1, 5, 8, 9, 10, 11, 12, 18, 19, 20, 23, 25, 30, 35, 50, 100], 
                "sim_options": {
                    "name": ["msd", "cosine", "pearson"],
                    "min_support": [1, 5, 10],# if number of common items < min_support, sim(u,v) = 0
                    "user_based": [False]
                }
            }
        },        
    ]
    cur_file_name = __file__
    pickle_file_name = cur_file_name.replace(".py", ".pickle")
    pickle_out = open(pickle_file_name, "wb")
    pickle.dump(experiments, pickle_out)
    pickle_out.close()
    
if __name__ == "__main__":
    main()    
