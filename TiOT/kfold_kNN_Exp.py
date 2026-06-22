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
from sklearn.model_selection import KFold

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
    markers = ['o', '^', 'D',  'v', 'P', 'X']
    linestyles = ['-', '-', "-", '-', '-']
    i = 0
    for name in alg_names:
        plt.plot(eps_list, np.array(results[name]), label = name, linewidth=1.75, marker=markers[i], linestyle = linestyles[i], markersize = 7)
        i+=1
    plt.xlabel(r"$\varepsilon$", fontsize = 16)
    plt.ylabel("Error", fontsize = 16)
    plt.legend()
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

def my_cross_val_score(model, X, y, cv=3):
    kf = KFold(n_splits=cv, shuffle=True, random_state=54)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        with multiprocessing.Pool(4) as pool:
            y_pred = list(tqdm(pool.imap(model.predict, [[x_test] for x_test in X_test]), total=len(X_test)))
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
    error = 1 - np.array(scores).mean()
    return error

def kNN(dataset_name, data, metric_name , eps_list , w ):
    global w_global, eps_global
    w_global = w
    if metric_name == "oriTAOT":
        metric = oriTAOT
    elif metric_name == "eTiOT":
        metric = eTiOT
    elif metric_name == 'euclidean':
        metric = 'euclidean'
    elif metric_name == 'eTAOT':
        metric = eTAOT

    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
    eps_best = 0
    error_best = np.inf
    errors = []
    for eps in eps_list:
        eps_global = eps
        knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
        error = my_cross_val_score(knn, X_train, Y_train)
        errors.append(error)
        if error_best >= error:
            eps_best = eps
            error_best = error
        print(f"Complete cross validation with metric = {metric_name}, eps = {eps} : error = {error}")
    eps_global = eps_best
    print(f"After cross validation eps_best = {eps_best} with average error {error_best}")
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
    knn.fit(X_train, Y_train)
    with multiprocessing.Pool(4) as pool:
        y_pred = list(tqdm(pool.imap(knn.predict, [[x_test] for x_test in X_test]), total=len(X_test)))
    pool.close()
    accuracy = accuracy_score(Y_test, y_pred)
    final_error = 1 - accuracy
    print(f"  ====>  Completed dataset: {dataset_name}, Metric : {metric_name}, eps = {eps_best}, Error:",final_error)
    errors.append(final_error)
    return errors


def experiment_kNN(dataset_name, w_TAOT, RUN = True): 
    eps_list = [0.01*i for i in range(1,11)]
    eps_name = f" ({eps_list[0]} to {eps_list[-1]})"  
    plot_file = os.path.join('Experimental_outputs',"kfold_kNN_data","plots", "Comparison on " + dataset_name + eps_name  + ".pdf")
    result_file = os.path.join('Experimental_outputs',"kfold_kNN_data", "saved_results","Results on " + dataset_name + eps_name + '.csv')

    if RUN :
        data = process_data(dataset_name = dataset_name)
        results = {**{'eps': eps_list}}
        results['eTiOT'] = kNN(dataset_name, data, metric_name='eTiOT', eps_list= eps_list, w = None)
        results['eTAOT'] = kNN(dataset_name, data, metric_name='oriTAOT', eps_list= eps_list, w = w_TAOT)
        results['eps'].append('Final error')

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
