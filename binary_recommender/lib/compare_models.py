"""Compare recommender model results"""
import os
import argparse
import fnmatch

from pathlib import Path
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import utilities
from pprint import pprint

HOLD_OUT_STRATEGIES = ['assume_first_n', 'assume_ratio', 'hold_last_n']
#HOLD_OUT_STRATEGIES = ['hold_last_n']

def get_cmap(index, name='tab20b_r'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, index)

def plot_roc(recommender, results, plot_results_dir='results', hold_out_strategy=''):
    """plot roc(tpr vs fpr)"""
    no_of_items_to_recommend = []
    avg_tpr_models_dict = OrderedDict()
    avg_fpr_models_dict = OrderedDict()
    for no_of_items in results:
        no_of_items_to_recommend.append(int(no_of_items))
        for model in results[no_of_items]:
            if hold_out_strategy in model:
                model_name = model.split(hold_out_strategy+'_')[1]
                if model_name not in avg_tpr_models_dict:
                    avg_tpr_models_dict[model_name] = []
                if model_name not in avg_fpr_models_dict:
                    avg_fpr_models_dict[model_name] = []
    #print(no_of_items_to_recommend)
    #no_of_items_to_recommend = [5, 10]
    for no_of_items in no_of_items_to_recommend:
        for model_name in avg_tpr_models_dict.keys():
            model = hold_out_strategy + '_' + model_name
            avg_tpr_val = float(results[str(no_of_items)][model]['avg_tpr'])
            avg_tpr_models_dict[model_name].append(avg_tpr_val)

        for model_name in avg_fpr_models_dict.keys():
            model = hold_out_strategy + '_' + model_name
            avg_fpr_val = float(results[str(no_of_items)][model]['avg_fpr'])
            avg_fpr_models_dict[model_name].append(avg_fpr_val)
    if not avg_tpr_models_dict or not avg_fpr_models_dict:
        return

    fig = plt.figure(figsize=(15, 5))
    axis = fig.add_subplot(121)
    plt.suptitle('*Points represent no_of_items being recommended ' + str(no_of_items_to_recommend),
                 fontsize=10, ha='left')
    axis.set_title('ROC2')
    i = 0
    no_of_models = len(avg_tpr_models_dict)
    cmap = get_cmap(no_of_models)
    for model in avg_tpr_models_dict:
        tprs = avg_tpr_models_dict[model]
        fprs = avg_fpr_models_dict[model]

        x_fprs = np.array(fprs)
        y_tprs = np.array(tprs)
        color = cmap(i)
        #print(i, color, model)
        axis.plot(x_fprs, y_tprs, label=model, color=color, marker='o')
        i = i+1
    if len(hold_out_strategy) > 1:
        plt.title('hold_out_strategy = ' + hold_out_strategy)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # Shrink current axis by 30%
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    # Put a legend to the right of the current axis
    axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axis.grid('on')
    if len(hold_out_strategy) > 1:
        img_name = hold_out_strategy + '_'
    else:
        img_name = ''
    img_name += 'results_roc.png'

    results_dir = os.path.join(recommender, plot_results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    img_file = os.path.join(results_dir, img_name)
    print("Generated Plot : {}".format(img_file))
    plt.savefig(img_file)
    #plt.show()
    plt.close(fig)

def plot_graph(recommender, results, measure, plot_results_dir='results', hold_out_strategy=''):
    """plot precision recall and f1-score"""
    #pprint(results)

    #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    #http://matplotlib.org/1.3.1/api/pyplot_api.html#matplotlib.pyplot.plot

    no_of_items_to_recommend = []
    models_dict = OrderedDict()
    for no_of_items in results:
        no_of_items_to_recommend.append(int(no_of_items))
        for model in results[no_of_items]:
            if model not in models_dict:
                if hold_out_strategy in model:
                    model_name = model.split(hold_out_strategy+'_')[1]
                    models_dict[model_name] = []
    #print(no_of_items_to_recommend)
    for no_of_items in no_of_items_to_recommend:
        for model_name in models_dict.keys():
            model = hold_out_strategy + '_' + model_name
            #print(no_of_items, model, results[str(no_of_items)][model][measure])
            val = float(results[str(no_of_items)][model][measure])
            models_dict[model_name].append(val)
    #pprint(models_dict)
    if not models_dict:
        return
    x_no_of_items_to_recommend = np.array(no_of_items_to_recommend)
    y_param = np.row_stack(tuple(models_dict.values()))
    #print(x_no_of_items_to_recommend)
    #print(y_param)

    fig = plt.figure(figsize=(15, 5))
    axis = fig.add_subplot(121)

    #for model, values, col in zip(models_dict.keys(), y_param, colors):
        #axis.plot(x_no_of_items_to_recommend, values, label=model, color=col, marker='o')
    i = 0
    no_of_models = len(models_dict)
    cmap = get_cmap(no_of_models)
    for model, values in zip(models_dict.keys(), y_param):
        color = cmap(i)
        #print(i, color, model)
        axis.plot(x_no_of_items_to_recommend, values, label=model, color=color, marker='o')
        i = i+1

    plt.suptitle('*Points represent no_of_items being recommended ' + str(no_of_items_to_recommend),
                 fontsize=10, ha='left')
    plt.xticks(x_no_of_items_to_recommend)
    if len(hold_out_strategy) > 1:
        plt.title('hold_out_strategy = ' + hold_out_strategy)
    plt.ylabel(measure)
    plt.xlabel('no_of_items_to_recommend')

    # Shrink current axis by 30%
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    # Put a legend to the right of the current axis
    axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axis.grid('on')
    if len(hold_out_strategy) > 1:
        img_name = hold_out_strategy + '_'
    else:
        img_name = ''
    img_name += 'results_' + measure + '.png'
    results_dir = os.path.join(recommender, plot_results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    img_file = os.path.join(results_dir, img_name)
    print("Generated Plot : {}".format(img_file))
    plt.savefig(img_file)
    #plt.show()
    plt.close(fig)

def plot_hold_out_strategies(recommender, results,
                              measure, plot_results_dir,
                              no_of_items_recommended):
    """plot measure"""
    hold_out_strategies = []
    models_dict = OrderedDict()
    for hold_out_strategy in results:
        hold_out_strategies.append(hold_out_strategy)
        for model in results[hold_out_strategy]:
            if model not in models_dict:
                models_dict[model] = []

    for hold_out_strategy in hold_out_strategies:
        for model in models_dict.keys():
            #print(hold_out_strategy, model, results[hold_out_strategy][model][measure])
            #input()
            val = float(results[hold_out_strategy][model][measure])
            models_dict[model].append(val)

    x_hold_out_strategies = np.array(hold_out_strategies)
    y_param = np.row_stack(tuple(models_dict.values()))
    #print(x_hold_out_strategies)
    #print(y_param)

    fig = plt.figure(figsize=(15, 5))
    axis = fig.add_subplot(121)

    #for model, values, col in zip(models_dict.keys(), y_param, colors):
        #axis.plot(x_no_of_items_to_recommend, values, label=model, color=col, marker='o')
    i = 0
    no_of_models = len(models_dict)
    cmap = get_cmap(no_of_models)
    for model, values in zip(models_dict.keys(), y_param):
        color = cmap(i)
        #print(i, color, model)
        axis.plot(x_hold_out_strategies, values, label=model, color=color, marker='o')
        i = i+1

    plt.title("no_of_items_recommended = " + no_of_items_recommended)
    plt.xticks(x_hold_out_strategies)
    plt.ylabel(measure)
    plt.xlabel('hold_out_strategies')

    # Shrink current axis by 30%
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    # Put a legend to the right of the current axis
    axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axis.grid('on')
    img_name = no_of_items_recommended + '_compare_hold_out_strategy_results_' + measure + '.png'
    results_dir = os.path.join(recommender, plot_results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    img_file = os.path.join(results_dir, img_name)
    print("Generated Plot : {}".format(img_file))
    plt.savefig(img_file)
    #plt.show()
    plt.close(fig)

def analyse_hold_out_strategies(recommender, results, measure, plot_results_dir='results'):
    list_of_items_recommended = results.keys()
    for no_of_items_recommended in list_of_items_recommended:
        results_hold_out_strategies = dict()
        for hold_out_strategy in HOLD_OUT_STRATEGIES:
            results_hold_out_strategies[hold_out_strategy] = dict()

        results_recommended = results[no_of_items_recommended]
        #pprint(results_recommended)
        for strategy_model in results_recommended:
            for hold_out_strategy in HOLD_OUT_STRATEGIES:
                #print(hold_out_strategy)
                if hold_out_strategy in strategy_model:
                    #print(hold_out_strategy, strategy_model)
                    model = strategy_model.split(hold_out_strategy+'_')[1]
                    #print(model)
                    results_hold_out_strategies[hold_out_strategy][model] = results_recommended[strategy_model]
        
        #pprint(results_hold_out_strategies)
        for hold_out_strategy in HOLD_OUT_STRATEGIES:
            if not results_hold_out_strategies[hold_out_strategy]:
                del results_hold_out_strategies[hold_out_strategy]
        #pprint(results_hold_out_strategies)

        plot_hold_out_strategies(recommender,
                                  results_hold_out_strategies,
                                  measure,
                                  plot_results_dir,
                                  no_of_items_recommended)
def dump_scores(recommender, results, score_results_dir='results'):
    """Dump Scores"""
    results_dir = os.path.join(recommender, score_results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_file = os.path.join(results_dir, 'results.csv')
    all_results = pd.DataFrame()
    for no_of_items_to_recommend in results:
        result = pd.DataFrame(results[no_of_items_to_recommend]).transpose()
        result['no_of_items_to_recommend'] = no_of_items_to_recommend
        #print(result)
        all_results = all_results.append(result)
        #print(all_results)
    all_results.to_csv(result_file)
    print("Evaluation Scores : {}".format(result_file))

def analyse(recommender, hybrid_models_only=False):
    """analyse single experiment for recommender using evalution_results.json"""
    results = dict()

    for root, _, files in os.walk(recommender, topdown=False):
        for name in files:
            if fnmatch.fnmatch(name, 'evaluation_results.json'):
                file_path = (os.path.join(root, name))
                model_name = (Path(file_path).parent.name)
                result_dict = utilities.load_json_file(file_path)
                for no_of_items_to_recommend in result_dict['no_of_items_to_recommend']:
                    if no_of_items_to_recommend not in results:
                        results[no_of_items_to_recommend] = dict()
                    res = result_dict['no_of_items_to_recommend'][no_of_items_to_recommend]
                    results[no_of_items_to_recommend][model_name] = res
    dump_scores(recommender, results, score_results_dir='score_results')    
    plot_results_dir='plot_results'
    analyse_hold_out_strategies(recommender, results, 'avg_tpr', plot_results_dir, hybrid_models_only)
    for hold_out_strategy in HOLD_OUT_STRATEGIES:
        plot_graph(recommender, results, 'avg_f1_score', plot_results_dir, hybrid_models_only, hold_out_strategy)
        plot_graph(recommender, results, 'avg_mcc_score', plot_results_dir, hybrid_models_only, hold_out_strategy)
        plot_graph(recommender, results, 'avg_precision', plot_results_dir, hybrid_models_only, hold_out_strategy)
        plot_graph(recommender, results, 'avg_recall', plot_results_dir, hybrid_models_only, hold_out_strategy)
        plot_roc(recommender, results, plot_results_dir, hybrid_models_only, hold_out_strategy)

def analyse_kfold(recommender, hybrid_models_only=False):
    """analyse kfold summary for recommender using kfold_evaluation.json"""
    results = dict()
    if hybrid_models_only:
        print("Analysing hybrid_models_only")
    for root, _, files in os.walk(recommender, topdown=False):
        for name in files:
            if fnmatch.fnmatch(name, 'kfold_evaluation.json'):
                file_path = (os.path.join(root, name))
                model_name = (Path(file_path).parent).parent.name
                if hybrid_models_only:
                    if 'hybrid' not in model_name:
                        continue
                result_dict = utilities.load_json_file(file_path)
                for no_of_items_to_recommend in result_dict['no_of_items_to_recommend']:
                    if no_of_items_to_recommend not in results:
                        results[no_of_items_to_recommend] = dict()
                    res = result_dict['no_of_items_to_recommend'][no_of_items_to_recommend]
                    results[no_of_items_to_recommend][model_name] = res
    if results:
        dump_scores(recommender, results, score_results_dir='score_kfold_results')

        plot_results_dir='plot_kfold_results'
        analyse_hold_out_strategies(recommender, results, 'avg_tpr', plot_results_dir)
        for hold_out_strategy in HOLD_OUT_STRATEGIES:
            plot_graph(recommender, results, 'avg_f1_score', plot_results_dir, hold_out_strategy)
            plot_graph(recommender, results, 'avg_mcc_score', plot_results_dir, hold_out_strategy)
            plot_graph(recommender, results, 'avg_precision', plot_results_dir, hold_out_strategy)
            plot_graph(recommender, results, 'avg_recall', plot_results_dir, hold_out_strategy)
            plot_roc(recommender, results, plot_results_dir, hold_out_strategy)

def main():
    """analyse results of recommenders"""
    parser = argparse.ArgumentParser(description="Compare Recommender Models")
    parser.add_argument("model", help="Use Case Recommender Model Directory")
    parser.add_argument("--kfold", help="Compare Kfold evaluation", action="store_true")
    parser.add_argument("--hybrid_only", help="For hybrid models only", action="store_true")
    args = parser.parse_args()

    if args.model:
        if args.kfold:
            print("Analysing kfold results of ", args.model)
            analyse_kfold(args.model, args.hybrid_only)
        else:
            print("Analysing results of ", args.model)
            analyse(args.model, args.hybrid_only)
    else:
        print("Specify use case recommender model directory to analyse")

if __name__ == '__main__':
    main()
