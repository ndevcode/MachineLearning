# -*- coding: utf-8 -*-
"""
   Copyright 2020-2023 Nilabha Dev 
   
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License

   README
   ------
   The problem domain is to recognize digits in a handwriting recognition system.
   For this we use a neural network based system that we first train
   using the MNIST dataset.
   
   The program is set up to do experiments by varying 
    - learning rate
    - number of iterations
    - activation function
    - mini batch size
    - dropout, regularization
   and generating graphical plots of training loss vs number of iterations 
   for the experiments
   
   This program uses python and the numpy library to implement 
   forward and backward propagation in a neural network
   
   v1 : Created on Tue Dec  8 08:41:53 2020      
   v2 : Created on Thu May 11 09:31:41 2023

   We create 4 classes,
      HyperParameters            : encompasses all the hyperparameters for the learning process
      NLayerNetwork              : instantiates a configurable N layer fully connected neural network
      PreprocessData             : loads data and creates one hot labels
      MNistHandWritingRecognizer : top level class that instantiates the neural network and runs the neural network's train and test routines
   
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist #only needed for loading the MNIST dataset
#import os

class HyperParameters:
    def __init__(self,alpha,num_iterations,batch_size,
                 lambd,hidden_layer_sizes,use_dropout,use_regularization,activation_scheme):
        assert(alpha > 0.0)
        self.alpha = alpha #0.05 #learning rate
        
        assert(num_iterations > 0)
        self.num_iterations = num_iterations #500
        
        assert(batch_size > 0)
        self.batch_size = batch_size #128
        
        assert(lambd > 0.0)
        self.lambd = lambd #0.7 # for regularization
        
        assert(len(hidden_layer_sizes) >= 1)
        for num_neurons in hidden_layer_sizes :
            assert(num_neurons > 0)
        self.hidden_layer_sizes = hidden_layer_sizes #[100]
        
        self.use_dropout = use_dropout
        self.use_regularization = use_regularization
        
        assert(activation_scheme == "tanh" or activation_scheme == "relu")
        self.activation_scheme = activation_scheme
    
    def get_learning_rate(self):
        return self.alpha

    def get_num_iterations(self):
        return self.num_iterations
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_regularization_lambda(self):
        return self.lambd
    
    def get_hidden_layers(self):
        return self.hidden_layer_sizes
    
    def get_use_dropout(self):
        return self.use_dropout
    
    def get_use_regularization(self):
        return self.use_regularization
    
    def get_activation_scheme(self):
        return self.activation_scheme

class NLayerNetwork:
    def __init__(self,hyper_parameters,layers,x,y):

        self.init_hyperparameters(hyper_parameters)

        self.L = len(layers)-1 #the last index 0,1,...,L
        self.layers = layers #L+1 layers including input and output

        self.x = x
        self.y = y
        self.m = x.shape[0] #number of examples

        self.weights = {}
        self.dropout = {}
        self.init_weight_and_bias_parameters()

        #internal data structures for the neural network
        self.cache = {}
        self.gradients = {}   
        
    def init_weight_and_bias_parameters(self):
        np.random.seed(1) #for repeatability of results always initialize with seed of 1

        for i in range(1,self.L+1):
            self.weights["W"+str(i)] = np.random.randn(self.layers[i],self.layers[i-1])
            self.weights["B"+str(i)] = np.zeros((self.layers[i],1))
             
    def init_hyperparameters(self,hyper_parameters):
        self.alpha = hyper_parameters.get_learning_rate() #0.05 learning rate
        self.num_iterations = hyper_parameters.get_num_iterations() #500
        self.batch_size = hyper_parameters.get_batch_size() #128
        self.lambd = hyper_parameters.get_regularization_lambda()#0.7 # for regularization
        self.use_dropout = hyper_parameters.get_use_dropout()
        self.use_regularization = hyper_parameters.get_use_regularization()
        self.activation_scheme = hyper_parameters.get_activation_scheme()
        
    def create_mini_batches(self):
        mini_batches = []
        num_complete_batches = math.ceil(self.m/self.batch_size)
        for i in range(num_complete_batches):
            batch_start = i*self.batch_size
            batch_end = min(self.m,(i+1)*self.batch_size)
            mini_batch = (batch_start,batch_end)
            mini_batches.append(mini_batch)
        return mini_batches
    
    def get_activation_fn(self,activation_scheme):
        def relu(x):
            return (x>=0)*x
        
        def tanh(x):
            return np.tanh(x)
        
        if (activation_scheme == "tanh") :
            return tanh
        elif (activation_scheme == "relu") :
            return relu
        
    def get_activation_fn_deriv(self,activation_scheme):
        def relu2deriv(y):
            return y>=0
           
        def tanh2deriv(y):
            return 1 - (y ** 2)
        
        if (activation_scheme == "tanh") :
            return tanh2deriv
        elif (activation_scheme == "relu") :
            return relu2deriv

    def softmax(self,x):
        temp = np.exp(x)
        return temp/np.sum(temp,axis=1,keepdims=True)
    
    def arrayLog(self,arr):
        result = np.where(arr > 0.0, arr, 0)
        np.log(result, out=result, where=result > 0)
        return result
        
    def compute_cross_entropy_cost(self,batch_start,batch_end,use_regularization):
        inner_m = batch_end - batch_start 
        An = self.cache["A"+str(self.L)]
        Y = self.y[batch_start:batch_end]
        logprobs = np.multiply(-self.arrayLog(An),Y) + np.multiply(-self.arrayLog(1-An), 1 - Y)
        cost_total =  np.sum(logprobs)   #entropy cost
        if (use_regularization):
            tmp = 0
            for i in range(1,self.L+1):
                tmp += np.sum(np.square(self.weights["W"+str(i)]))
            L2_regularization_cost = (1. /inner_m)*(self.lambd / 2) * tmp
            cost_total += L2_regularization_cost
        return cost_total
    
    def forward_propagation_inner(self,iData,use_dropout):
        activation_fn = self.get_activation_fn(self.activation_scheme)
        
        local_cache = {}
        local_cache["A0"] = iData
        for i in range(1,self.L):
            local_cache["Z"+str(i)] = np.dot(local_cache["A"+str(i-1)],self.weights["W"+str(i)].T)+self.weights["B"+str(i)].T
            local_cache["A"+str(i)] = activation_fn(local_cache["Z"+str(i)])
            if (use_dropout):
                self.dropout[i] = np.random.randint(2,size=local_cache["A"+str(i)].shape)
                local_cache["A"+str(i)] *= self.dropout[i] 
        local_cache["Z"+str(self.L)] = np.dot(local_cache["A"+str(self.L-1)],self.weights["W"+str(self.L)].T)+self.weights["B"+str(self.L)].T
        local_cache["A"+str(self.L)] = self.softmax(local_cache["Z"+str(self.L)])        
        return local_cache
    
    def forward_propagation(self,batch_start,batch_end,use_dropout):
        self.cache = self.forward_propagation_inner(self.x[batch_start:batch_end],use_dropout)
        return self.cache["A"+str(self.L)]
        
    def backward_propagation(self,batch_start,batch_end,use_dropout,use_regularization):
        deriv_activation_fn = self.get_activation_fn_deriv(self.activation_scheme)
        
        inner_m = batch_end - batch_start
        
        grads = {}
        
        grads["dZ"+str(self.L)] = self.cache["A"+str(self.L)] - self.y[batch_start:batch_end]
        grads["dW"+str(self.L)] = 1./inner_m * np.dot(grads["dZ"+str(self.L)].T,self.cache["A"+str(self.L-1)])
        if (use_regularization):
            grads["dW"+str(self.L)] += (1./inner_m * (self.lambd * self.weights["W"+str(self.L)]))
        grads["dB"+str(self.L)] = 1./inner_m * np.sum(grads["dZ"+str(self.L)].T,axis=1,keepdims=True) 
        
        for i in range(self.L-1,0,-1):    
            grads["dA"+str(i)] = np.dot(grads["dZ"+str(i+1)],self.weights["W"+str(i+1)])
            if (use_dropout):
                grads["dA"+str(i)] *= self.dropout[i]
            grads["dZ"+str(i)] = np.multiply(grads["dA"+str(i)],deriv_activation_fn(self.cache["A"+str(i)]))
            grads["dW"+str(i)] = 1./inner_m * np.dot(grads["dZ"+str(i)].T,self.cache["A"+str(i-1)])
            if (use_regularization):
                grads["dW"+str(i)] += (1./inner_m * (self.lambd * self.weights["W"+str(i)]))
            grads["dB"+str(i)] = 1./inner_m * np.sum(grads["dZ"+str(i)].T,axis=1,keepdims=True) 

        self.gradients = grads
    
    def update_parameters_gradient_descent(self):
        for i in range(1,self.L+1):
            self.weights["W"+str(i)] -= self.alpha * self.gradients["dW"+str(i)]
            self.weights["B"+str(i)] -= self.alpha * self.gradients["dB"+str(i)]
    
    #use mini batch gradient descent for the training
    def train(self):
        costs = []
        iters = []
        mini_batches = self.create_mini_batches()
        for i in range(self.num_iterations):
            y_cumulative_error = 0.0
            correct_count = 0
            for mini_batch in mini_batches:
                #standard loop of forward propagation, error computation, 
                #backward propagation and parameter update via mini batch gradient descent
                (batch_start,batch_end) = mini_batch
                yhat = self.forward_propagation(batch_start,batch_end,self.use_dropout)
                y_cumulative_error += self.compute_cross_entropy_cost(batch_start,batch_end,self.use_regularization)
                for k in range(batch_end-batch_start):
                    correct_count += int(np.argmax(yhat[k:k+1]) == np.argmax(self.y[batch_start+k:batch_start+k+1]))
                self.backward_propagation(batch_start,batch_end,self.use_dropout,self.use_regularization)
                self.update_parameters_gradient_descent()
            if ((i%10) == 9):
                print("\r"+ \
                    " I:"+str(i)+ \
                    " Error:" + str(y_cumulative_error/float(self.m))[0:5] +\
                    " Correct:" + str(correct_count/float(self.m)))
                costs.append(y_cumulative_error/float(self.m))
                iters.append(i)
        return costs,iters

        
    def test(self,x,y):
        y_cumulative_error = 0.0
        correct_count = 0
        
        local_cache = self.forward_propagation_inner(x,False)
        An = local_cache["A"+str(self.L)]
        
        y_cumulative_error += np.sum(np.multiply(-self.arrayLog(An),y) + np.multiply(-self.arrayLog(1 - An), 1 - y))
        for k in range(len(x)):
            correct_count += int(np.argmax(An[k:k+1]) == np.argmax(y[k:k+1]))
        print("\r"+ \
            " Error:" + str(y_cumulative_error/float((len(x))))[0:5] +\
            " Correct:" + str(correct_count/float(len(x))))  

class PreprocessData:
    def __init__(self):
        # Step 1 : load the MNIST data from local directory
        #for local usage and to use load_data functionality download the mnist data from
        #https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
        #current_dir = os.getcwd()
        #(self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data(path=current_dir+'/mnist/mnist.npz')
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data() # This works but may download data repeatedly
        
        # Step 2 : train with only 2000 examples to save runtime
        #num_cases = len(self.x_train)
        num_cases = 2000 
        
        #each black and white 28 X 28 image has a pixel with value between 0 to 255. 
        #linearize this image array and normalize its values to between 0 and 1
        self.train_images = self.x_train[0:num_cases].reshape(num_cases,28*28)/255
        print(self.train_images.shape)
        
        #create one hot labels 
        #e.g. if image is an 8 then one hot array is [0,0,0,0,0,0,0,0,1,0]
        self.train_labels = self.make_one_hot_labels(self.y_train[0:num_cases])
        print(self.train_labels.shape)
        
        # Step 3 : ditto for the test images
        self.test_images = self.x_test.reshape(len(self.x_test),28*28) / 255
        self.test_labels = self.make_one_hot_labels(self.y_test)

    def make_one_hot_labels(self,labels):
        num_cases = len(labels)
        one_hot_labels = np.zeros((num_cases,10))
        for i,l in enumerate(labels):
            one_hot_labels[i][l] = 1
        return one_hot_labels
        
    def get_train_data_and_labels(self):
        return self.train_images, self.train_labels
    
    def get_test_data_and_labels(self):
        return self.test_images, self.test_labels
               
class MNISTHandwritingRecognizer:
    def __init__(self,hyper_parameters,theData):        
        # Step 1 : create the layers of the neural network
        #number of neurons in input and output layers are fixed 
        #input : 784 = 28 * 28, the size of the image
        #output: 10, the 10 categories 0,1,..., 9
        #get the number of neurons in the hidden layers
        #in practise didn't find any improvement in test/train accuracy by increasing 
        #the number of hidden layers so here we use only 1 hidden layer
        self.layers = [784]
        for num_neurons in hyper_parameters.get_hidden_layers():
            self.layers.append(num_neurons)
        self.layers.append(10)
        
        # Step 2 : get the test and train images
        self.train_images,self.train_labels = theData.get_train_data_and_labels()
        self.test_images,self.test_labels = theData.get_test_data_and_labels()
        
        # Step 3 : instantiate the neural network
        self.nn = NLayerNetwork(hyper_parameters,self.layers,self.train_images,self.train_labels)
       
    def run(self):           
        costs,iters = self.nn.train()
        self.nn.test(self.test_images,self.test_labels)
        return costs,iters;

if __name__ == "__main__":
    theData = PreprocessData()
    #experiment 1 : use learning rate of 0.05
    experiment1 = HyperParameters(alpha = 0.05, 
                              num_iterations = 500,  
                              batch_size = 128,  
                              lambd = 0.7,  
                              hidden_layer_sizes = [100],
                              activation_scheme = "tanh",
                              use_dropout = True,
                              use_regularization = True) 
    hwr = MNISTHandwritingRecognizer(experiment1,theData)
    costs,iters = hwr.run()
    experiment1_plot, = plt.plot(iters,costs,color='red')
    experiment1_legend = "Learning rate = " + str(experiment1.get_learning_rate())
       
    #experiment 2 : use learning rate of 0.02
    experiment2 = HyperParameters(alpha = 0.02, 
                              num_iterations = 500,  
                              batch_size = 128,  
                              lambd = 0.7,  
                              hidden_layer_sizes = [100],
                              activation_scheme = "tanh",
                              use_dropout = True,
                              use_regularization = True) 
    hwr = MNISTHandwritingRecognizer(experiment2,theData)
    costs,iters = hwr.run()
    experiment2_plot, = plt.plot(iters,costs,color='blue')
    experiment2_legend = "Learning rate = " + str(experiment2.get_learning_rate())
    
    plt.legend([experiment1_plot,experiment2_plot],
               [experiment1_legend,experiment2_legend])
                                         
    plt.ylabel('Training error')
    plt.xlabel('Number of iterations')
    plt.title("Training error vs number of iterations for various learning rates")
    plt.show()
