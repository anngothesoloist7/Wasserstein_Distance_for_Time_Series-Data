import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import TiOT_lib
import os
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing
from sklearn.metrics import accuracy_score
from tqdm import tqdm



eps_global = 0.01
w_global = 10

def eTiOT(X1, X2):
    return TiOT_lib.eTiOT(X1,X2, eps=eps_global)[0]

def eTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, w = w_global, eps = eps_global)[0]

def oriTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, w = w_global, eps = eps_global, costmatrix=TiOT_lib.costmatrix0)[0]

def process_data(dataset_name):
    train_file = os.path.join("time_series_kNN", dataset_name, dataset_name + "_TRAIN.txt" )
    test_file = os.path.join("time_series_kNN", dataset_name, dataset_name + "_TEST.txt")

    with open(train_file, "r") as file:
        data = np.array([line.strip().split() for line in file], dtype=float)

    Y_train = data[:, 0]    
    X_train = data[:, 1:]

    with open(test_file, "r") as file:
        data_test = np.array([line.strip().split() for line in file], dtype=float)

    Y_test = data_test[:, 0]
    X_test = data_test[:, 1:] 

    return [X_train, Y_train, X_test, Y_test]

def plot_results(results, plot_file):
    eps_list = results['eps']
    alg_names = [k for k in results.keys() if k != 'eps']
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(8, 5))
    plt.grid(False)
    markers = ['o', '^', 'D',  'v', 'P', 'X']
    linestyles = ['-', '-', "-", '-', '-']
    i = 0
    for name in alg_names:
        plt.plot(eps_list, np.array(results[name]), label = name, linewidth=1.75, marker=markers[i], linestyle = linestyles[i], markersize = 9)
        i+=1
    plt.tick_params(axis="both", which="major", labelsize=21, bottom=True, left=True)
    plt.xlabel(r"$\varepsilon$", fontsize = 21)
    plt.ylabel("Error", fontsize = 21)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)  
    plt.show()

def save_result(results, result_file):
    df = pd.DataFrame(results)
    df.to_csv(result_file, index=False)

def read_result(result_file):
    df = pd.read_csv(result_file)
    results = df.to_dict(orient='list')
    return results

def kNN(dataset_name, data, metric_name , eps , w, eta = None ):
    global w_global, eps_global
    w_global = w
    eps_global = eps
    if metric_name == "oriTAOT":
        metric = oriTAOT
    elif metric_name == "eTiOT":
        metric = eTiOT
    elif metric_name == 'euclidean':
        metric = 'euclidean'
    elif metric_name == 'eTAOT':
        metric = eTAOT

    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
    knn.fit(X_train, Y_train)
    with multiprocessing.Pool(4) as pool:
        y_pred = list(tqdm(pool.imap(knn.predict, [[x_test] for x_test in X_test]), total=len(X_test)))
    pool.close()
    accuracy = accuracy_score(Y_test, y_pred)
    error = 1 - accuracy
    print(f"  ====>  Completed dataset: {dataset_name}, Metric : {metric_name}, Error:",error)
    return error

def experiment_kNN(dataset_name, w_TAOT, RUN = True):
    eps_list = [0.01*i for i in range(1,11)]
    eps_name = f" ({eps_list[0]} to {eps_list[-1]})"       
    plot_file = os.path.join('Experimental_outputs',"robust_kNN_data","plots", "Comparison on " + dataset_name + eps_name + ".pdf")
    result_file = os.path.join('Experimental_outputs',"robust_kNN_data", "saved_results","Results on " + dataset_name + eps_name  + '.csv')

    if RUN :
        data = process_data(dataset_name = dataset_name)
        w_list = [ round(w_TAOT/5, 3), w_TAOT,w_TAOT*5]
        w_list_name = [r'\omega_{\text{grid}} \;/\; 5', r'\omega_{\text{grid}}', r'\omega_{\text{grid}} \times 5']
        alg_names = ["eTiOT"]   +  [fr"eTAOT$(\omega = {w})$" for w in w_list_name]
        results = {**{'eps': eps_list}, **{name: [] for name in alg_names}}
        for eps in eps_list:
            results['eTiOT'].append(kNN(dataset_name, data, metric_name='eTiOT', eps = eps, w = None))
            for i in range(len(w_list)):
                results[fr"eTAOT$(\omega = {w_list_name[i]})$"].append(kNN(dataset_name, data, metric_name='oriTAOT', eps = eps, w = w_list[i]))

        save_result(results, result_file)
        plot_results(results, plot_file)
    else:
        results = read_result(result_file)
        plot_results(results, plot_file)
 
if __name__ == "__main__":

    experiment_kNN('Adiac',0.1) 
    # experiment_kNN('ArrowHead', 3)
    # experiment_kNN("CBF", 1)
    # experiment_kNN('BirdChicken', 0.1)
    # experiment_kNN("DistalPhalanxOutlineAgeGroup", 1)
    # experiment_kNN('DistalPhalanxOutlineCorrect', 0.4)
    # experiment_kNN('DistalPhalanxTW', 0.5 )
    # experiment_kNN('Ham', 0.7)
    # experiment_kNN('MiddlePhalanxOutlineAgeGroup', 0.2)
    # experiment_kNN('MiddlePhalanxOutlineCorrect', 0.5)
    # experiment_kNN('MiddlePhalanxTW', 0.4)
    # experiment_kNN('ProximalPhalanxOutlineCorrect', 0.7)
    # experiment_kNN("ProximalPhalanxTW", 0.7)
    # experiment_kNN("SonyAIBORobotSurface1", 2)
    # experiment_kNN('SwedishLeaf',0.9) 


