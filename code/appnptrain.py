# -*- coding: utf-8 -*-
"""
"""

import random
import torch
import numpy as np
from tqdm import trange
from appnpmodel import APPNPModel

class APPNPTrainer(object):
    """
    Method to train PPNP/APPNP model:https://arxiv.org/abs/1810.05997
    """
    def __init__(self, graph,features, target,model,layers,dropout,iterations,alpha
                 ,train_size,lambd,learning_rate,epochs,early_stopping_rounds):
        """
        APPNTrainer : Training APPN model
        
        Parameters
        ----------
        graph : Networkx graph
        features : Features matrix
        target: Target labels
        layers
        model
        dropout: Dropout paramter
        iterations: Number of iterations
        alpha
        train_size :  Training size 
        lambd: Lambda parameter
        learning rate : Learning rate parameter
        epochs : Number of epochs
        Early_stopping_rounds : Early stopping parameter
        """
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph = graph
        self.features = features
        self.target = target
        self.model = model
        self.layers = layers
        self.dropout = dropout
        self.iterations = iterations
        self.alpha = alpha
        self.train_size = train_size
        self.lambd = lambd
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.loss = 0
        self.create_model()
        self.train_test_split()
        self.transfer_node_sets()
        self.process_features()
        self.transfer_features()
        
    
    def create_model(self):
        
        """
        model creation
        """
        
        self.node_count = self.graph.number_of_nodes()
        self.number_of_labels = np.max(self.target)+1
        self.number_of_features = max([f for _, feats  in self.features.items() for f in feats]) + 1

        self.model = APPNPModel(self.number_of_labels, 
                                self.number_of_features, 
                                self.graph, 
                                self.device,
                                self.model,
                                self.layers,
                                self.dropout,
                                self.iterations,
                                self.alpha)

        self.model = self.model.to(self.device)
 
    def train_test_split(self):
        """
        Creating a train/test split.
        """
        nodes = [node for node in range(self.node_count)]
        random.shuffle(nodes)
        self.train_nodes = nodes[0:self.train_size]
        self.validation_nodes = nodes[self.train_size:]

    def transfer_node_sets(self):
        
        """
        nodes transfering to device
        """
        
        self.train_nodes = torch.LongTensor(self.train_nodes).to(self.device)
        self.validation_nodes = torch.LongTensor(self.validation_nodes).to(self.device)

    def process_features(self):
        
        """
        sparse feature matrix and a vector for the target labels
        """
        
        index_1 = [int(node) for node in self.graph.nodes() for fet in self.features[int(node)]]
        index_2 = [fet for node in self.graph.nodes() for fet in self.features[int(node)]]
        values = [1.0/len(self.features[int(node)]) if self.features[int(node)] else 0 for node in self.graph.nodes() for fet in self.features[int(node)]]
        self.feature_indices = torch.LongTensor([index_1, index_2])
        self.feature_values = torch.FloatTensor(values)
        self.target = torch.LongTensor(self.target)


    def transfer_features(self):
        
        """
        the features and the target matrix transfer to the device
        """
        
        self.target = self.target.to(self.device)
        self.feature_indices = self.feature_indices.to(self.device)
        self.feature_values = self.feature_values.to(self.device)

    def score(self, index_set):
        
        """
        Accuracy score
        
        Parameters
        ----------
        index_set: Index of nodes to be included in calculation.
        acc: Accuracy score.
        """
        
        self.model.eval()
        _, pred = self.model(self.feature_indices, self.feature_values).max(dim=1)
        correct = pred[index_set].eq(self.target[index_set]).sum().item()
        acc = correct / index_set.size()[0]
        return acc

    def do_a_step(self):
        
        """
        Optimization step
        """
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(self.feature_indices, self.feature_values)
        loss = torch.nn.functional.nll_loss(prediction[self.train_nodes],
                                            self.target[self.train_nodes])
        self.loss = loss
        loss = loss+(self.lambd/2)*(torch.sum(self.model.layer_2.weight_matrix**2))
        loss.backward()
        self.optimizer.step()

    def train_neural_network(self):
        """
        Neural network training
        """
        
        print("\nTraining.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.best_accuracy = 0
        self.step_counter = 0
        iterator = trange(self.epochs, desc='Validation accuracy: ', leave=True)
        for _ in iterator:
            self.do_a_step()
            self.accuracy = self.score(self.validation_nodes)
            iterator.set_description("Validation loss: {:.4f}".format(self.loss))
            if self.accuracy >= self.best_accuracy:
                self.best_accuracy = self.accuracy
                self.step_counter = 0
            else:
                self.step_counter = self.step_counter + 1
                if self.step_counter > self.early_stopping_rounds:
                    iterator.close()
                    break

    def fit(self):
        
        """
        Network fitting and accuracy calculation
        """
        
        self.train_neural_network()
        print("\nBreaking from training process because of early stopping.\n")
        print("Test accuracy: {:.4f}".format(self.accuracy))
