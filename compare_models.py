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

from lib import utilities

def get_cmap(index, name='tab20b_r'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, index)

def plot_roc(recommender, results, plot_results_dir='results'):
    """plot roc(tpr vs fpr)"""
    no_of_items_to_recommend = []
    avg_tpr_models_dict = OrderedDict()
    avg_fpr_models_dict = OrderedDict()
    for no_of_items in results:
        no_of_items_to_recommend.append(int(no_of_items))
        for model in results[no_of_items]:
            #if 'hybrid' not in model:
            #    continue
            if model not in avg_tpr_models_dict:
                avg_tpr_models_dict[model] = []
            if model not in avg_fpr_models_dict:
                avg_fpr_models_dict[model] = []
    #print(no_of_items_to_recommend)
    #no_of_items_to_recommend = [5, 10]
    for no_of_items in no_of_items_to_recommend:
        for model in avg_tpr_models_dict.keys():
            avg_tpr_val = float(results[str(no_of_items)][model]['avg_tpr'])
            avg_tpr_models_dict[model].append(avg_tpr_val)
        for model in avg_fpr_models_dict.keys():
            avg_fpr_val = float(results[str(no_of_items)][model]['avg_fpr'])
            avg_fpr_models_dict[model].append(avg_fpr_val)
    #pprint(avg_tpr_models_dict)
    #pprint(avg_fpr_models_dict)

    fig = plt.figure(figsize=(10, 6))
    axis = fig.add_subplot(111)
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
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # Shrink current axis by 30%
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    # Put a legend to the right of the current axis
    axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axis.grid('on')
    img_name = 'results_roc.png'
    results_dir = os.path.join(recommender, plot_results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    img_file = os.path.join(results_dir, img_name)
    print("Generated Plot : {}".format(img_file))
    plt.savefig(img_file)
    plt.show()

def plot_graph(recommender, results, measure, plot_results_dir='results'):
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
                models_dict[model] = []
    #print(no_of_items_to_recommend)
    for no_of_items in no_of_items_to_recommend:
        for model in models_dict.keys():
            #print(no_of_items, model, results[str(no_of_items)][model][measure])
            val = float(results[str(no_of_items)][model][measure])
            models_dict[model].append(val)
    #pprint(models_dict)
    x_no_of_items_to_recommend = np.array(no_of_items_to_recommend)
    y_param = np.row_stack(tuple(models_dict.values()))
    #print(x_no_of_items_to_recommend)
    #print(y_param)

    fig = plt.figure(figsize=(10, 6))
    axis = fig.add_subplot(111)

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

    plt.xticks(x_no_of_items_to_recommend)
    plt.ylabel(measure)
    plt.xlabel('no_of_items_to_recommend')

    # Shrink current axis by 30%
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    # Put a legend to the right of the current axis
    axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axis.grid('on')
    img_name = 'results_' + measure + '.png'
    results_dir = os.path.join(recommender, plot_results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    img_file = os.path.join(results_dir, img_name)
    print("Generated Plot : {}".format(img_file))
    plt.savefig(img_file)
    #plt.show()

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

def analyse(recommender):
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
    plot_graph(recommender, results, 'avg_f1_score', plot_results_dir='plot_results')
    plot_graph(recommender, results, 'avg_mcc_score', plot_results_dir='plot_results')
    plot_graph(recommender, results, 'avg_precision', plot_results_dir='plot_results')
    plot_graph(recommender, results, 'avg_recall', plot_results_dir='plot_results')
    plot_roc(recommender, results, plot_results_dir='plot_results')

def analyse_kfold(recommender):
    """analyse kfold summary for recommender using kfold_evaluation.json"""
    results = dict()

    for root, _, files in os.walk(recommender, topdown=False):
        for name in files:
            if fnmatch.fnmatch(name, 'kfold_evaluation.json'):
                file_path = (os.path.join(root, name))
                model_name = (Path(file_path).parent).parent.name
                #print(model_name)
                result_dict = utilities.load_json_file(file_path)
                for no_of_items_to_recommend in result_dict['no_of_items_to_recommend']:
                    if no_of_items_to_recommend not in results:
                        results[no_of_items_to_recommend] = dict()
                    res = result_dict['no_of_items_to_recommend'][no_of_items_to_recommend]
                    results[no_of_items_to_recommend][model_name] = res
    dump_scores(recommender, results, score_results_dir='score_kfold_results')
    plot_graph(recommender, results, 'avg_f1_score', plot_results_dir='plot_kfold_results')
    plot_graph(recommender, results, 'avg_mcc_score', plot_results_dir='plot_kfold_results')
    plot_graph(recommender, results, 'avg_precision', plot_results_dir='plot_kfold_results')
    plot_graph(recommender, results, 'avg_recall', plot_results_dir='plot_kfold_results')
    plot_roc(recommender, results, plot_results_dir='plot_kfold_results')

def main():
    """analyse results of recommenders"""
    parser = argparse.ArgumentParser(description="Compare Recommender Models")
    parser.add_argument("model", help="Use Case Recommender Model Directory")
    parser.add_argument("--kfold", help="Compare Kfold evaluation", action="store_true")
    args = parser.parse_args()
    if args.model:
        if args.kfold:
            print("Analysing kfold results of ", args.model)
            analyse_kfold(args.model)
        else:
            print("Analysing results of ", args.model)
            analyse(args.model)
    else:
        print("Specify use case recommender model directory to analyse")

if __name__ == '__main__':
    main()
