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

'''
experiments = [
    ###########################################Baseline Rating Models###########################################
    #hyper_param_search_results_baselineonly_als
    {
        "results_dir_name" : "hyper_param_search_experiments/hyper_param_search_results_baselineonly_als",
        "algos" : [BaselineOnly],
        "param_grid" : {
            "bsl_options": {
                "method": ["als"],
                "n_epochs": [10, 50, 100, 500],
                "reg_u": [0.1, 1, 5, 10],
                "reg_i": [10, 20, 30, 50]
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
                "n_epochs": [1, 5, 10, 50],
                "reg": [0.02, 0.08, 0.1, 0.2, 0.5, 0.8],
                "learning_rate": [0.001, 0.01, 0.1, 0.5, 1]
            }
        }     
    },
]
'''

'''
experiments = [
    #hyper_param_search_results_knn_baseline_als user_based (1 - 31sec) (4*5*4*4=320 - 1.5323    hr)
    {
        "results_dir_name" : "hyper_param_search_experiments/user_based_hyper_param_search_results_knn_baseline_als",
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
                "n_epochs": [100],
                "reg_u": [5],
                "reg_i": [30],
            }
        }
    },             

    #hyper_param_search_results_knn_baseline_sgd user_based (1 - 31sec) (4*5*4*4=320 - 1.5361    hr)
    {
        "results_dir_name" : "hyper_param_search_experiments/user_based_hyper_param_search_results_knn_baseline_sgd",
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
                "n_epochs": [5],
                "reg": [0.5],
                "learning_rate" : [0.1]
            }
        }
    },

    {
        "results_dir_name" : "hyper_param_search_experiments/user_based_hyper_param_search_results_knns",
        "algos" : [KNNBasic, KNNWithMeans, KNNWithZScore],
        "param_grid" : {
            #(max) number of neighbors
            "k": [5, 10, 50, 100, 500],
            #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
            "min_k" : [1, 5, 10, 100], 
            "sim_options": {
                "name": ["msd", "cosine", "pearson"],
                "min_support": [1, 2, 5, 10],# if number of common items < min_support, sim(u,v) = 0
                "user_based": [True]
            }
        }
    },    
]
'''
'''
experiments = [
    {
        "results_dir_name" : "hyper_param_search_experiments/user_based_hyper_param_search_results_knn_baseline_sgd",
        "algos" : [KNNBaseline],
        "param_grid" : {
            #(max) number of neighbors
            "k": [10, 100, 500, 800, 1000, 1200],
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
                "n_epochs": [5],
                "reg": [0.5],
                "learning_rate" : [0.1]
            }
        }
    },
    
    {
        "results_dir_name" : "hyper_param_search_experiments/item_based_hyper_param_search_results_knn_baseline_sgd",
        "algos" : [KNNBaseline],
        "param_grid" : {
            #(max) number of neighbors
            "k": [10, 100, 500, 800, 1000, 1200],
            #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
            "min_k" : [1, 5, 10, 20, 50],       
            "sim_options": {
                "name": ["pearson_baseline"],
                "shrinkage": [0, 1000, 10000, 100000],  # 0 : no shrinkage
                "min_support": [1, 2, 5, 10],# if number of common items < min_support, sim(u,v) = 0
                "user_based": [False]
            },
            "bsl_options": {
                "method": ["sgd"],            
                "n_epochs": [5],
                "reg": [0.5],
                "learning_rate" : [0.1]
            }
        }
    },
    
    
    {
        "results_dir_name" : "hyper_param_search_experiments/svd_unbiased_hyper_param_search_results",
        "algos" : [SVD],
        "param_grid" : {
            #n_factors – The number of factors. Default is 100.
            "n_factors": [10, 50, 100, 150, 200],
            #n_epochs – The number of iteration of the SGD procedure. Default is 20.
            "n_epochs" : [10, 20, 50, 100],
            #biased – Whether to use baselines (or biases). See note above. Default is True.
            "biased" : [False],
            #lr_all – The learning rate for all parameters. Default is 0.005.
            "lr_all" : [0.0001, 0.001, 0.01, 0.1, 1],
            #reg_all – The regularization term for all parameters. Default is 0.02.
            "reg_all" : [0.0001, 0.001, 0.01, 0.1, 1]
        }
    },
    {
        "results_dir_name" : "hyper_param_search_experiments/svd_biased_hyper_param_search_results",
        "algos" : [SVD],
        "param_grid" : {
            #n_factors – The number of factors. Default is 100.
            "n_factors": [10, 50, 100, 150, 200],
            #n_epochs – The number of iteration of the SGD procedure. Default is 20.
            "n_epochs" : [10, 20, 50, 100],
            #biased – Whether to use baselines (or biases). See note above. Default is True.
            "biased" : [True],
            #lr_all – The learning rate for all parameters. Default is 0.005.
            "lr_all" : [0.0001, 0.001, 0.01, 0.1, 1],
            #reg_all – The regularization term for all parameters. Default is 0.02.
            "reg_all" : [0.0001, 0.001, 0.01, 0.1, 1]
        }
    }
    
]
'''
experiments = [
    {
        "results_dir_name" : "hyper_param_search_experiments/svdpp_biased_hyper_param_search_results",
        "algos" : [SVDpp],
        "param_grid" : {
            #n_factors – The number of factors. Default is 20.
            "n_factors": [5, 10, 100, 200, 250],
            #n_epochs – The number of iteration of the SGD procedure. Default is 20.
            "n_epochs" : [10, 20, 50, 100],
            #lr_all – The learning rate for all parameters. Default is 0.005.
            "lr_all" : [0.001, 0.01, 0.1],
            #reg_all – The regularization term for all parameters. Default is 0.02.
            "reg_all" : [0.001, 0.01, 0.1]
        }
    } 
]

experiments = [
    {
        "results_dir_name" : "hyper_param_search_experiments/svdpp_biased_hyper_param_search_results1",
        "algos" : [SVDpp],
        "param_grid" : {
            #n_factors – The number of factors. Default is 20.
            "n_factors": [10, 20, 30],
            #n_epochs – The number of iteration of the SGD procedure. Default is 20.
            "n_epochs" : [10, 20, 30],
            #lr_all – The learning rate for all parameters. Default is 0.005.
            "lr_all" : [0.001, 0.01],
            #reg_all – The regularization term for all parameters. Default is 0.02.
            "reg_all" : [0.001, 0.01]
        }
    },
    {
        "results_dir_name" : "hyper_param_search_experiments/svd_biased_hyper_param_search_results1",
        "algos" : [SVD],
        "param_grid" : {
            #n_factors – The number of factors. Default is 100.
            "n_factors": [100, 200, 250, 500],
            #n_epochs – The number of iteration of the SGD procedure. Default is 20.
            "n_epochs" : [5, 10, 20],
            #biased – Whether to use baselines (or biases). See note above. Default is True.
            "biased" : [True],
            #lr_all – The learning rate for all parameters. Default is 0.005.
            "lr_all" : [0.01, 0.1],
            #reg_all – The regularization term for all parameters. Default is 0.02.
            "reg_all" : [0.01, 0.1]
        }
    },
    
    {
        "results_dir_name" : "hyper_param_search_experiments/item_based_hyper_param_search_results_knn_baseline_sgd1",
        "algos" : [KNNBaseline],
        "param_grid" : {
            #(max) number of neighbors
            "k": [2, 5, 10],
            #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
            "min_k" : [15, 20, 25],       
            "sim_options": {
                "name": ["pearson_baseline"],
                "shrinkage": [0, 1000],  # 0 : no shrinkage
                "min_support": [1, 2],# if number of common items < min_support, sim(u,v) = 0
                "user_based": [False]
            },
            "bsl_options": {
                "method": ["sgd"],            
                "n_epochs": [5],
                "reg": [0.5],
                "learning_rate" : [0.1]
            }
        }
    },
    
    {
        "results_dir_name" : "hyper_param_search_experiments/user_based_hyper_param_search_results_knn_baseline_sgd1",
        "algos" : [KNNBaseline],
        "param_grid" : {
            #(max) number of neighbors
            "k": [700, 800, 900],
            #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
            "min_k" : [15, 20, 25],       
            "sim_options": {
                "name": ["pearson_baseline"],
                "shrinkage": [0, 1000],  # 0 : no shrinkage
                "min_support": [1, 2],# if number of common items < min_support, sim(u,v) = 0
                "user_based": [True]
            },
            "bsl_options": {
                "method": ["sgd"],            
                "n_epochs": [5],
                "reg": [0.5],
                "learning_rate" : [0.1]
            }
        }
    },
    
    {
        "results_dir_name" : "hyper_param_search_experiments/user_based_hyper_param_search_results_knns1",
        "algos" : [KNNWithZScore],
        "param_grid" : {
            #(max) number of neighbors
            "k": [100, 500, 1500, 2000],
            #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
            "min_k" : [5, 10, 15], 
            "sim_options": {
                "name": ["msd"],
                "min_support": [1, 2],# if number of common items < min_support, sim(u,v) = 0
                "user_based": [True]
            }
        }
    },
    
    {
        "results_dir_name" : "hyper_param_search_experiments/item_based_hyper_param_search_results_knns",
        "algos" : [KNNWithZScore],
        "param_grid" : {
            #(max) number of neighbors
            "k": [2, 5, 10],
            #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
            "min_k" : [5, 10, 15], 
            "sim_options": {
                "name": ["msd"],
                "min_support": [1, 2],# if number of common items < min_support, sim(u,v) = 0
                "user_based": [True]
            }
        }
    },
]




experiments = [
    {
        "results_dir_name" : "hyper_param_search_experiments/item_based_hyper_param_search_results_knns",
        "algos" : [KNNWithZScore],
        "param_grid" : {
            #(max) number of neighbors
            "k": [2, 5, 10],
            #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
            "min_k" : [5, 10, 15], 
            "sim_options": {
                "name": ["msd"],
                "min_support": [1, 2],# if number of common items < min_support, sim(u,v) = 0
                "user_based": [False]
            }
        }
    },
]
'''
experiments = [
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
]
'''
'''
experiments = [
    ###########################################KNN UserBased CF Rating Models###########################################
    #hyper_param_search_results_knns user_based
    {
        "results_dir_name" : "hyper_param_search_experiments/user_based_hyper_param_search_results_knns",
        "algos" : [KNNBasic, KNNWithMeans, KNNWithZScore],
        "param_grid" : {
            #(max) number of neighbors
            "k": [5, 10, 50, 100, 500],
            #(min) number of neighbors,  not enough neighbors, use default_prediction=gmean
            "min_k" : [1, 5, 10, 50, 100], 
            "sim_options": {
                "name": ["msd", "cosine", "pearson"],
                "min_support": [1, 2, 5, 7, 10],# if number of common items < min_support, sim(u,v) = 0
                "user_based": [True]
            }
        }
    },
]
'''
'''

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
experiments = [
    {
        "results_dir_name" : "hyper_param_search_experiments/svd_unbiased_hyper_param_search_results1",
        "algos" : [SVD],
        "param_grid" : {
            #n_factors – The number of factors. Default is 100.
            "n_factors": [5, 8, 10, 12, 15],#[10, 50, 100, 150, 200],
            #n_epochs – The number of iteration of the SGD procedure. Default is 20.
            "n_epochs" : [10, 40, 50, 70],#[10, 20, 50, 100],
            #biased – Whether to use baselines (or biases). See note above. Default is True.
            "biased" : [False],
            #lr_all – The learning rate for all parameters. Default is 0.005.
            "lr_all" : [0.01],#[0.0001, 0.001, 0.01, 0.1, 1],
            #reg_all – The regularization term for all parameters. Default is 0.02.
            "reg_all" : [0.1]#[0.0001, 0.001, 0.01, 0.1, 1]
        }
    },
    {
        "results_dir_name" : "hyper_param_search_experiments/svd_biased_hyper_param_search_results1",
        "algos" : [SVD],
        "param_grid" : {
            #n_factors – The number of factors. Default is 100.
            "n_factors": [150, 200, 250, 500],#[10, 50, 100, 150, 200],
            #n_epochs – The number of iteration of the SGD procedure. Default is 20.
            "n_epochs" : [50, 100, 150, 200],#[10, 20, 50, 100],
            #biased – Whether to use baselines (or biases). See note above. Default is True.
            "biased" : [True],
            #lr_all – The learning rate for all parameters. Default is 0.005.
            "lr_all" : [0.01],#[0.0001, 0.001, 0.01, 0.1, 1],
            #reg_all – The regularization term for all parameters. Default is 0.02.
            "reg_all" : [0.1]#[0.0001, 0.001, 0.01, 0.1, 1]
        }
    }
]

experiments = [
    {
        "results_dir_name" : "hyper_param_search_experiments/nmf_unbiased_hyper_param_search_results1",
        "algos" : [NMF],
        "param_grid" : {                
            # n_factors – The number of factors. Default is 15.
            "n_factors": [10, 50, 100],
            # n_epochs – The number of iteration of the SGD procedure. Default is 50.
            "n_epochs" : [10, 50, 100],
            # biased (bool) – Whether to use baselines (or biases). Default is False.
            "biased" : [False],
            # reg_pu – The regularization term for users λu. Default is 0.06.
            "reg_pu" : [0.001, 0.01, 0.1],
            # reg_qi – The regularization term for items λi. Default is 0.06.
            "reg_qi" : [0.001, 0.01, 0.1],

            # init_low – Lower bound for random initialization of factors. 
            # Must be greater than 0 to ensure non-negative factors. Default is 0.
            #"init_low" : [0, 0.5],
            # init_high – Higher bound for random initialization of factors. Default is 1.
            #"init_high" : [1, 10],
            # random_state Determines the RNG that will be used for initialization.
            #"random_state" : [123]
        }
    },
    {
        "results_dir_name" : "hyper_param_search_experiments/nmf_biased_hyper_param_search_results1",
        "algos" : [NMF],
        "param_grid" : {                
            # n_factors – The number of factors. Default is 15.
            "n_factors": [15, 50],
            # n_epochs – The number of iteration of the SGD procedure. Default is 50.
            "n_epochs" : [50, 100],
            # biased (bool) – Whether to use baselines (or biases). Default is False.
            "biased" : [True],
            # reg_pu – The regularization term for users λu. Default is 0.06.
            "reg_pu" : [0.001, 0.01],
            # reg_qi – The regularization term for items λi. Default is 0.06.
            "reg_qi" : [0.001, 0.01],

            # reg_bu – The regularization term for bu. Only relevant for biased version. Default is 0.02.
            "reg_bu" : [0.001, 0.01],
            # reg_bi – The regularization term for bi. Only relevant for biased version. Default is 0.02.
            "reg_bi" : [0.001, 0.01],
            # lr_bu – The learning rate for bu. Only relevant for biased version. Default is 0.005.
            "lr_bu" : [0.001, 0.01],
            # lr_bi – The learning rate for bi. Only relevant for biased version. Default is 0.005.
            "lr_bi" : [0.001, 0.01],

            # init_low – Lower bound for random initialization of factors. 
            # Must be greater than 0 to ensure non-negative factors. Default is 0.
            #"init_low" : [0, 0.5],
            # init_high – Higher bound for random initialization of factors. Default is 1.
            #"init_high" : [1, 10],
            # random_state – Determines the RNG that will be used for initialization.
            #"random_state" : [123]
        }
    }
]
'''

def main():
    cur_file_name = __file__
    pickle_file_name = cur_file_name.replace(".py", ".pickle")
    pickle_out = open(pickle_file_name, "wb")
    pickle.dump(experiments, pickle_out)
    pickle_out.close()
    
if __name__ == "__main__":
    main()    
