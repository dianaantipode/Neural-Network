# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 21:56:02 2016

@author: Chang Gao

Works Cited:
Britz, Denny. "Implementing a Neural Network from Scratch in Python – An Introduction." WildML. N.p., 3 Sept. 2015. 
Web. 21 Feb. 2016.
"""

import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    """
    This class implements a simple 3 layer neural network.
    """
    
    def __init__(self, input_dim, output_dim, epsilon):
        """
        Initializes the parameters of the neural network to random values
        We think that 4 hidden layer nodes are good enough to process train 
        the network
        """
        self.Weight1 = np.random.randn(input_dim, 4) / np.sqrt(input_dim)
        self.bias1 = np.zeros((1, 4))
        self.Weight2 = np.random.randn(4, output_dim) / np.sqrt(4)
        self.bias2 = np.zeros((1, output_dim))
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total loss on the dataset
        """
        
        num_samples = len(X)
        # Do Forward propagation to calculate our predictions
        z = X.dot(self.W) + self.b
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        # Calculate the cross-entropy loss
        cross_ent_err = -np.log(softmax_scores[range(num_samples), y])
        data_loss = np.sum(cross_ent_err)
        return 1./num_samples * data_loss

    
    #--------------------------------------------------------------------------
 
    def predict(self,x):
        z = x.dot(self.Weight1) +self.bias1
        a1 = np.tanh(z)
        z1 = a1.dot(self.Weight2) + self.bias2
        exp_z = np.exp(z1)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return np.argmax(softmax_scores, axis=1)
      
        
    #--------------------------------------------------------------------------
    # Using the method disscussed in the article "Implementing a Neural Network 
    # from Scratch in Python – An Introduction". We are able to implement a 
    # good fit function. 
    
    
    
    def fit(self,X,y,num_epochs):
        
        
         #For each epoch
         for i in xrange(0,num_epochs):
            #   Do Forward Propagation
            z1 = X.dot(self.Weight1) + self.bias1
            a1 = np.tanh(z1)
            z2 = a1.dot(self.Weight2) +self.bias2
            exp_z = np.exp(z2)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            #   Do Back Propagation
            change3 = softmax_scores
            change3[range(len(X)), y] -= 1
            changeW2 = (a1.T).dot(change3)
            changeb2 = np.sum(change3, axis=0, keepdims=True)
            change2 = change3.dot(self.Weight2.T) * (1 - np.power(a1, 2))
            changeW1 = np.dot(X.T, change2)
            changeb1 = np.sum(change2, axis=0)
            
            #Regularization 
            changeW2 += rL * self.Weight2
            changeW1 += rL * self.Weight1
 
            #   Update model parameters using gradients
            self.Weight2 += -epsilon * changeW2
            self.bias2 += -epsilon * changeb2
            
            self.Weight1 += -epsilon * changeW1
            self.bias1 += -epsilon * changeb1
           
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ###TODO:
        
        
        
        

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def plot_decision_boundary(pred_func):
    """
    Helper function to print the decision boundary given by model
    """
    # Set min and max values
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#Train Neural Network on
linear = False

#A. linearly separable data
if linear:
    #load data
    X = np.genfromtxt('C:\Users\laiw12\Desktop\Lab4_Soln\DATA\ToyLinearX.csv', delimiter=',')
    y = np.genfromtxt('C:\Users\laiw12\Desktop\Lab4_Soln\DATA\ToyLineary.csv', delimiter=',')
    y = y.astype(int)
    #plot data
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
#B. Non-linearly separable data
else:
    #load data
    X = np.genfromtxt('C:\Users\laiw12\Desktop\Lab4_Soln\DATA\ToyMoonX.csv', delimiter=',')
    y = np.genfromtxt('C:\Users\laiw12\Desktop\Lab4_Soln\DATA\ToyMoony.csv', delimiter=',')
    y = y.astype(int)
    #plot data
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

input_dim = 2 # input layer dimensionality
output_dim = 2 # output layer dimensionality

# Gradient descent parameters 
epsilon = 0.01
num_epochs = 5000
rL = 0.01

# Fit model
#----------------------------------------------
#Uncomment following lines after implementing NeuralNet
#----------------------------------------------
NN = NeuralNet(input_dim, output_dim, epsilon)
NN.fit(X,y,num_epochs)
#
# Plot the decision boundary
plot_decision_boundary(lambda x: NN.predict(x))
plt.title("Neural Net Decision Boundary")
            
    
