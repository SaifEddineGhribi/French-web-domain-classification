# -*- coding: utf-8 -*-
"""
"""
import torch
from utils import create_propagator_matrix , uniform
from torch_sparse import spmm


class DenseFullyConnected(torch.nn.Module):
    
    """
    Approximate PageRank Network
    
    Parameters
    ----------
    in_channels: Number of input channels.
    out_channels: Number of output channels.
    density: Feature matrix.
    
    """
    def __init__(self, in_channels, out_channels):
        super(DenseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        
        """
        Weights matrices
        """
        
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        
        """
        Wieghts Initialization 
        """
        
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, features):
        
        """
        Forward Pass
        
        Parameters
        ----------
        features: Feature matrix
        
        Return 
        ----------
        filtered_features: Convolved features
        """
        
        filtered_features = torch.mm(features, self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class SparseFullyConnected(torch.nn.Module):
    
    """
    Approximate PageRank Network
    
    Parameters
    ----------
    in_channels: Number of input channels.
    out_channels: Number of output channels.
    density: Feature matrix.
    
    """
    
    def __init__(self, in_channels, out_channels):
        super(SparseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        
        """
        Weights matrices
        """
        
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        
        """
        Wieghts Initialization 
        """
        
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, feature_indices, feature_values):
        
        """
        Forward Pass
        
        Parameters
        ----------
        features: Feature matrix
        
        Return 
        ----------
        filtered_features: Convolved features
        """
        
        number_of_nodes = torch.max(feature_indices[0]).item()+1
        number_of_features = torch.max(feature_indices[1]).item()+1
        filtered_features = spmm(index = feature_indices,
                                 value = feature_values,
                                 m = number_of_nodes,
                                 n = number_of_features,
                                 matrix = self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class APPNPModel(torch.nn.Module):
    
    """
    APPNP Model
    
    Parameters
    ----------
    number_of_labels: Number of Target labels
    number_of_features : Number of features
    graph: Networkx graph
    device: Device type
    model: Model
    layers
    dropout: Dropout parameter
    iteration: Number of iterations
    alpha    
    """
    
    def __init__(self, number_of_labels, number_of_features, graph, device,model,layers,dropout,iterations,alpha):
        super(APPNPModel, self).__init__()
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.graph = graph
        self.device = device
        self.model = model
        self.layers = layers
        self.dropout = dropout
        self.iterations = iterations
        self.alpha = alpha
        self.setup_layers()
        self.setup_propagator()

    def setup_layers(self):
        
        """
        Layers creation
        """
        
        self.layer_1 = SparseFullyConnected(self.number_of_features, self.layers[0])
        self.layer_2 = DenseFullyConnected(self.layers[1], self.number_of_labels)

    def setup_propagator(self):
        """
        Propagation matrix creation
        """
        
        self.propagator = create_propagator_matrix(self.graph, self.alpha, self.model)
        if self.model == "exact":
            self.propagator = self.propagator.to(self.device)
        else:
            self.edge_indices = self.propagator["indices"].to(self.device)
            self.edge_weights = self.propagator["values"].to(self.device)

    def forward(self, feature_indices, feature_values):
        
        """
        Forward propagation pass
        
        Parameters
        ----------
        feature_indices: Feature indices for feature matrix.
        feature_values: Values in the feature matrix.
        
        Return
        ----------
        self.predictions: Predicted class label log softmaxes
        """
        
        feature_values = torch.nn.functional.dropout(feature_values,
                                                     p=self.dropout,
                                                     training=self.training)

        latent_features_1 = self.layer_1(feature_indices, feature_values)

        latent_features_1 = torch.nn.functional.relu(latent_features_1)

        latent_features_1 = torch.nn.functional.dropout(latent_features_1,
                                                        p=self.dropout,
                                                        training=self.training)

        latent_features_2 = self.layer_2(latent_features_1)
        if self.model == "exact":
            self.predictions = torch.nn.functional.dropout(self.propagator,
                                                           p=self.dropout,
                                                           training=self.training)

            self.predictions = torch.mm(self.predictions, latent_features_2)
        else:
            localized_predictions = latent_features_2
            edge_weights = torch.nn.functional.dropout(self.edge_weights,
                                                       p=self.dropout,
                                                       training=self.training)

            for iteration in range(self.iterations):

                new_features = spmm(index=self.edge_indices,
                                    value=edge_weights,
                                    n=localized_predictions.shape[0],
                                    m=localized_predictions.shape[0],
                                    matrix=localized_predictions)

                localized_predictions = (1-self.alpha)*new_features
                localized_predictions = localized_predictions + self.alpha*latent_features_2
            self.predictions = localized_predictions
        self.predictions = torch.nn.functional.log_softmax(self.predictions, dim=1)
        return self.predictions
