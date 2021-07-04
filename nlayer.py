# -*- coding: utf-8 -*-

"""
   Copyright 2020-2021 N. Dev : github user id: ndevcode

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
Created on Tue Dec  8 08:41:53 2020

@author: ndev

This file  creates 2 classes, a n-layer network called NLayerNetwork and a handwriting number recognition system 
class called MNISTHandwritingRecognizer that instantiates and uses the class. 

This class uses python without the use of any library such as Tensorflow or Pytorch. 

It has about 90% accuracy rate

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

class NLayerNetwork:
    def __init__(self,layers,x,y):

        self.init_hyperparameters()

        self.L = len(layers)-1
        self.layers = layers

        self.x = x
        self.y = y
        self.m = x.shape[0] #number of examples

        self.weights = {}
        self.dropout = {}
        self.init_weight_and_bias_parameters()

        self.cache = {}
        self.gradients = {}   
        
    def init_weight_and_bias_parameters(self):
        np.random.seed(1)

        for i in range(1,self.L+1):
            self.weights["W"+str(i)] = np.random.randn(self.layers[i],self.layers[i-1])
            self.weights["B"+str(i)] = np.zeros((self.layers[i],1))
             
    def init_hyperparameters(self):
        self.alpha = 0.05 #learning rate
        self.num_iterations = 500
        self.batch_size = 128
        self.lambd = 0.7 # for regularization
        
    def create_mini_batches(self):
        mini_batches = []
        num_complete_batches = math.ceil(self.m/self.batch_size)
        for i in range(num_complete_batches):
            batch_start = i*self.batch_size
            batch_end = min(self.m,(i+1)*self.batch_size)
            mini_batch = (batch_start,batch_end)
            mini_batches.append(mini_batch)
        return mini_batches

    def relu(self,x):
        return (x>=0)*x
    
    def relu2deriv(self,y):
        return y>=0
    
    def tanh(self,x):
        return np.tanh(x)
    
    def tanh2deriv(self,y):
        return 1 - (y ** 2)
    
    def softmax(self,x):
        temp = np.exp(x)
        return temp/np.sum(temp,axis=1,keepdims=True)
    
    def compute_mse_cost(self,batch_start,batch_end):
        A4 = self.cache["A4"]
        val = np.sum((A4 - self.y[batch_start:batch_end])**2)
        return val
    
    def compute_cross_entropy_cost(self,batch_start,batch_end,useRegul):
        inner_m = batch_end - batch_start 
        An = self.cache["A"+str(self.L)]
        Y = self.y[batch_start:batch_end]
        logprobs = np.multiply(-np.log(An),Y) + np.multiply(-np.log(1 - An), 1 - Y)
        cost_total =  np.sum(logprobs)   #entropy cost
        if (useRegul):
            tmp = 0
            for i in range(1,self.L+1):
                tmp += np.sum(np.square(self.weights["W"+str(i)]))
            L2_regularization_cost = (1. /inner_m)*(self.lambd / 2) * tmp
            cost_total += L2_regularization_cost
        return cost_total
    
    def forward_propagation_inner(self,iData,useDropout):
        local_cache = {}
        local_cache["A0"] = iData
        for i in range(1,self.L):
            local_cache["Z"+str(i)] = np.dot(local_cache["A"+str(i-1)],self.weights["W"+str(i)].T)+self.weights["B"+str(i)].T
            local_cache["A"+str(i)] = self.tanh(local_cache["Z"+str(i)])
            if (useDropout):
                self.dropout[i] = np.random.randint(2,size=local_cache["A"+str(i)].shape)
                local_cache["A"+str(i)] *= self.dropout[i] 
        local_cache["Z"+str(self.L)] = np.dot(local_cache["A"+str(self.L-1)],self.weights["W"+str(self.L)].T)+self.weights["B"+str(self.L)].T
        local_cache["A"+str(self.L)] = self.softmax(local_cache["Z"+str(self.L)])        
        return local_cache
    
    def forward_propagation(self,batch_start,batch_end,useDropout):
        self.cache = self.forward_propagation_inner(self.x[batch_start:batch_end],useDropout)
        return self.cache["A"+str(self.L)]
        
    def backward_propagation(self,batch_start,batch_end,useDropout,useRegul):
 
        inner_m = batch_end - batch_start
        
        grads = {}
        
        grads["dZ"+str(self.L)] = self.cache["A"+str(self.L)] - self.y[batch_start:batch_end]
        grads["dW"+str(self.L)] = 1./inner_m * np.dot(grads["dZ"+str(self.L)].T,self.cache["A"+str(self.L-1)])
        if (useRegul):
            grads["dW"+str(self.L)] += (1./inner_m * (self.lambd * self.weights["W"+str(self.L)]))
        grads["dB"+str(self.L)] = 1./inner_m * np.sum(grads["dZ"+str(self.L)].T,axis=1,keepdims=True) 
        
        for i in range(self.L-1,0,-1):    
            grads["dA"+str(i)] = np.dot(grads["dZ"+str(i+1)],self.weights["W"+str(i+1)])
            if (useDropout):
                grads["dA"+str(i)] *= self.dropout[i]
            grads["dZ"+str(i)] = np.multiply(grads["dA"+str(i)],self.tanh2deriv(self.cache["A"+str(i)]))
            grads["dW"+str(i)] = 1./inner_m * np.dot(grads["dZ"+str(i)].T,self.cache["A"+str(i-1)])
            if (useRegul):
                grads["dW"+str(i)] += (1./inner_m * (self.lambd * self.weights["W"+str(i)]))
            grads["dB"+str(i)] = 1./inner_m * np.sum(grads["dZ"+str(i)].T,axis=1,keepdims=True) 

        self.gradients = grads
    
    def update_parameters_with_gd(self):
        for i in range(1,self.L+1):
            self.weights["W"+str(i)] -= self.alpha * self.gradients["dW"+str(i)]
            self.weights["B"+str(i)] -= self.alpha * self.gradients["dB"+str(i)]
    
    def train(self):
        costs = []
        mini_batches = self.create_mini_batches()
        for i in range(self.num_iterations):
            y_cumulative_error = 0.0
            correct_count = 0
            for mini_batch in mini_batches:
                (batch_start,batch_end) = mini_batch
                yhat = self.forward_propagation(batch_start,batch_end,True)
                y_cumulative_error += self.compute_cross_entropy_cost(batch_start,batch_end,True)
                for k in range(batch_end-batch_start):
                    correct_count += int(np.argmax(yhat[k:k+1]) == np.argmax(self.y[batch_start+k:batch_start+k+1]))
                self.backward_propagation(batch_start,batch_end,True,True)
                self.update_parameters_with_gd()
            if ((i%10) == 9):
                print("\r"+ \
                    " I:"+str(i)+ \
                    " Error:" + str(y_cumulative_error/float(self.m))[0:5] +\
                    " Correct:" + str(correct_count/float(self.m)))
                costs.append(y_cumulative_error/float(self.m))
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per 10)')
        plt.title("Learning rate = " + str(self.alpha))
        plt.show()
        
    def test(self,x,y):
        y_cumulative_error = 0.0
        correct_count = 0
        
        local_cache = self.forward_propagation_inner(x,False)
        An = local_cache["A"+str(self.L)]
        
        y_cumulative_error += np.sum(np.multiply(-np.log(An),y) + np.multiply(-np.log(1 - An), 1 - y))
        for k in range(len(x)):
            correct_count += int(np.argmax(An[k:k+1]) == np.argmax(y[k:k+1]))
        print("\r"+ \
            " Error:" + str(y_cumulative_error/float((len(x))))[0:5] +\
            " Correct:" + str(correct_count/float(len(x))))  
               
class MNISTHandwritingRecognizer:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        #num_cases = len(self.x_train)
        num_cases = 2000
        self.train_images = self.x_train[0:num_cases].reshape(num_cases,28*28)/255
        print(self.train_images.shape)
        self.train_labels = self.make_one_hot_labels(self.y_train[0:num_cases])
        print(self.train_labels.shape)
        self.test_images = self.x_test.reshape(len(self.x_test),28*28) / 255
        self.test_labels = self.make_one_hot_labels(self.y_test)
        self.layers = [784,89,10]
        self.nn = NLayerNetwork(self.layers,self.train_images,self.train_labels)
    
    def make_one_hot_labels(self,labels):
        num_cases = len(labels)
        one_hot_labels = np.zeros((num_cases,10))
        for i,l in enumerate(labels):
            one_hot_labels[i][l] = 1
        return one_hot_labels
    
    def run(self):           
        self.nn.train()
        self.nn.test(self.test_images,self.test_labels)

hwr = MNISTHandwritingRecognizer()
hwr.run()
