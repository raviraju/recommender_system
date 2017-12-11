import os
import argparse
from pathlib import Path
from pprint import pprint
from lib import utilities
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

def get_cmap(n, name='Set1'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_roc(recommender, results):
    #pprint(results)
    no_of_items_to_recommend = []
    avg_precision_models_dict = OrderedDict()
    avg_recall_models_dict = OrderedDict()
    for no_of_items in results:
        no_of_items_to_recommend.append(int(no_of_items))
        for model in results[no_of_items]:
            if model not in avg_precision_models_dict:
                avg_precision_models_dict[model] = []
            if model not in avg_recall_models_dict:
                avg_recall_models_dict[model] = []
    #print(no_of_items_to_recommend)
    for no_of_items in no_of_items_to_recommend:
        for model in avg_precision_models_dict.keys():
            avg_precision_val = 1 - float(results[str(no_of_items)][model]['avg_precision'])           
            avg_precision_models_dict[model].append(avg_precision_val)
        for model in avg_recall_models_dict.keys():
            avg_recall_val = float(results[str(no_of_items)][model]['avg_recall'])           
            avg_recall_models_dict[model].append(avg_recall_val)            
    #pprint(avg_precision_models_dict)
    #pprint(avg_recall_models_dict)

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)

    i = 0
    no_of_models = len(avg_precision_models_dict)
    cmap = get_cmap(no_of_models)
    for model in avg_precision_models_dict:
        precisions = avg_precision_models_dict[model]
        recalls = avg_recall_models_dict[model]
       
        x_precisions = np.array(precisions)
        y_recalls = np.array(recalls)
        color = cmap(i)
        #print(i, color, model)
        ax1.plot(x_precisions, y_recalls, label=model, color=color, marker='o')
        i = i+1
    plt.ylabel('recall')
    plt.xlabel('1-precision')
        
    handles, labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.2,1))
    ax1.grid('on')
    img_name = 'results_roc.png'
    img_file = os.path.join(recommender, 'results', img_name)
    print("Generated Plot : {}".format(img_file))
    plt.savefig(img_file)
    plt.show()
    
def plot_graph(recommender, results, measure):
    #pprint(results)
    
    #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']#http://matplotlib.org/1.3.1/api/pyplot_api.html#matplotlib.pyplot.plot
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
    
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
        
    #for model, values, col in zip(models_dict.keys(), y_param, colors):
        #ax1.plot(x_no_of_items_to_recommend, values, label=model, color=col, marker='o')
    i=0
    no_of_models = len(models_dict)
    cmap = get_cmap(no_of_models)        
    for model, values in zip(models_dict.keys(), y_param):
        color = cmap(i)
        #print(i, color, model)
        ax1.plot(x_no_of_items_to_recommend, values, label=model, color=color, marker='o')
        i = i+1
    
    plt.xticks(x_no_of_items_to_recommend)
    plt.ylabel(measure)
    plt.xlabel('no_of_items_to_recommend')
    
    handles, labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.8,1))
    ax1.grid('on')
    img_name = 'results_' + measure + '.png'
    img_file = os.path.join(recommender, 'results', img_name)
    print("Generated Plot : {}".format(img_file))
    plt.savefig(img_file)
    #plt.show()

def analyse(recommender):
    results = dict() 

    for root, dirs, files in os.walk(recommender, topdown=False): 
        for name in files: 
            if name.endswith('results.json'): 
                file_path = (os.path.join(root, name))
                model_name = (Path(file_path).parent.name)
                result_dict = utilities.load_json_file(file_path)
                for no_of_items_to_recommend in result_dict['no_of_items_to_recommend']:
                    if no_of_items_to_recommend not in results:
                        results[no_of_items_to_recommend] = dict()
                    res = result_dict['no_of_items_to_recommend'][no_of_items_to_recommend]
                    results[no_of_items_to_recommend][model_name] = res


    plot_graph(recommender, results, 'avg_f1_score')
    plot_graph(recommender, results, 'avg_precision')
    plot_graph(recommender, results, 'avg_recall')
    plot_roc(recommender, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare Recommender Models")
    parser.add_argument("model", help="Use Case Recommender Model Directory")
    args = parser.parse_args()
    if args.model:
        print("Analysing results of ", args.model)
        analyse(args.model)
    else:
        print("Specify use case recommender model directory to analyse")