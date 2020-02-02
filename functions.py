import csv
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import collections


def get_train_data(G): 
    with open("train.csv", 'r') as f:
        train_data = f.read().splitlines()

    train_hosts = list()
    y_train = list()
    for row in train_data:
        host, label = row.split(",")
        train_hosts.append(host)
        y_train.append(label.lower())
        # Create the training matrix. Each row corresponds to a web host.
    # Use the following 3 features for each web host (unweighted degrees)
    # (1) out-degree of node
    # (2) in-degree of node
    # (3) average degree of neighborhood of node
    X_train = np.zeros((len(train_hosts), 3))
    avg_neig_deg = nx.average_neighbor_degree(G, nodes=train_hosts)
    for i in range(len(train_hosts)):
        X_train[i,0] = G.in_degree(train_hosts[i])
        X_train[i,1] = G.out_degree(train_hosts[i])
        X_train[i,2] = avg_neig_deg[train_hosts[i]]
    return X_train,y_train 

############################################################################

def get_test_data(G): 
    with open("test.csv", 'r') as f:
        test_hosts = f.read().splitlines()
    # Create the test matrix. Use the same 3 features as above
    X_test = np.zeros((len(test_hosts), 3))
    avg_neig_deg = nx.average_neighbor_degree(G, nodes=test_hosts)
    for i in range(len(test_hosts)):
        X_test[i,0] = G.in_degree(test_hosts[i])
        X_test[i,1] = G.out_degree(test_hosts[i])
        X_test[i,2] = avg_neig_deg[test_hosts[i]]
    return X_test

############################################################################

def create_prediction(clf,y_pred): 
    #y_pred = clf.predict_proba(X_test)
    with open("test.csv", 'r') as f:
        test_hosts = f.read().splitlines()

    # Write predictions to a file
    with open('graph_baseline.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        lst = clf.classes_.tolist()
        lst.insert(0, "Host")
        writer.writerow(lst)
        for i,test_host in enumerate(test_hosts):
            lst = y_pred[i,:].tolist()
            lst.insert(0, test_host)
            writer.writerow(lst)
    
############################################################################

def show_class_distribution(y_train):
    dict_classes = collections.Counter(y_train)
    classes = list(dict_classes.keys())
    occurence_classes = list(dict_classes.values())

    percent = 100/sum(occurence_classes) * np.array(occurence_classes)
    plt.figure(figsize=(8,8))

    patches, texts = plt.pie(occurence_classes, labels=None,
                      shadow=True, startangle=90)

    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(classes, percent)]
    plt.title("Classes distribution")
    plt.legend(patches, labels, loc='upper left', bbox_to_anchor=(-0.1, 1.),
               fontsize=8)
    plt.show()