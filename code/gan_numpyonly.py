# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 18:24:35 2024

@author: ndev
"""

import numpy as np
import matplotlib.pyplot as plt

def weight_initializer(in_channels, out_channels):
    scale = np.sqrt(2./ (in_channels + out_channels))
    return np.random.uniform(-scale,scale,(in_channels,out_channels))

class Constants():
    def __init__(self,table):
        self.parameters = table
            
    def get_parm(self,key) :
        return self.parameters[key]
    

class Calculus():
    def __init__(self):
        self.functions_ = dict()
        self.derivatives_ = dict()
        self.functions_["relu"] = self._make_relu
        self.functions_["leaky_relu"] = self._make_leaky_relu
        self.functions_["tanh"] = self._make_tanh
        self.functions_["sigmoid"] = self._make_sigmoid
        self.derivatives_["relu"] = self._make_relu_deriv
        self.derivatives_["leaky_relu"] = self._make_leaky_relu_deriv
        self.derivatives_["tanh"] = self._make_tanh_deriv
        self.derivatives_["sigmoid"] = self._make_sigmoid_deriv
        
    def get_function(self,fname) :
        return self.functions_[fname]
        
    def get_derivative(self,drvname) :
        return self.derivatives_[drvname] 

    def _make_relu(self,x):
        return np.maximum(x,0.)

    def _make_leaky_relu(self,x, k = 0.2):
        return np.where(x >= 0,x, x * k)

    def _make_tanh(self,x):
        return np.tanh(x)

    def _make_sigmoid(self,x):
        return 1./(1. + np.exp(-x))

    def _make_relu_deriv(self,x):
        return self._make_relu(x)

    def _make_leaky_relu_deriv(self,x, k = 0.2):
        return np.where(x >= 0,1., k)

    def _make_tanh_deriv(self,x):
        return 1. - self._make_tanh(x)**2

    def _make_sigmoid_deriv(self,x):
        return self._make_sigmoid(x) * (1. - self._make_sigmoid(x))

class LossFunction():
    def __init__(self):
        self.logit = None
        self.label = None

    def forward(self, logit, label):
        if logit[0,0] < 1e-7:
            logit[0,0] = 1e-7
        if 1. - logit[0,0] < 1e-7:
            logit[0,0] = 1. - 1e-7
        self.logit = logit
        self.label = label
        return - (label * np.log(logit) + (1-label)*np.log(1-logit))
        
    def backward(self) :
        return (1 - self.label) / (1 - self.logit)  - self.label/self.logit

def get_samples(parms,random=True):
    if random:
        x0 = np.random.uniform(0,1)
        freq = np.random.uniform(1.2,1.5)
        mult = np.random.uniform(0.5,0.8)
    else :
        x0 = 0
        freq = 0.2
        mult = 1
    signal = [mult * np.sin(x0 + freq*i) for i in range(0,parms.get_parm("x_dim"))]
    return np.array(signal)

class Generator():
    def __init__(self,parms,calculus):
        self.parms = parms
        self.calculus = calculus
        self.z  = None
        self.w1 = weight_initializer(in_channels = self.parms.get_parm("z_dim"), out_channels = self.parms.get_parm("g_hidden"))
        self.b1 = weight_initializer(in_channels = 1, out_channels = self.parms.get_parm("g_hidden"))
        self.x1 = None
        self.w2 = weight_initializer(in_channels = self.parms.get_parm("g_hidden"), out_channels = self.parms.get_parm("g_hidden"))
        self.b2 = weight_initializer(in_channels = 1, out_channels = self.parms.get_parm("g_hidden"))
        self.x2 = None
        self.w3 = weight_initializer(in_channels = self.parms.get_parm("g_hidden"), out_channels = self.parms.get_parm("x_dim"))
        self.b3 = weight_initializer(in_channels = 1, out_channels = self.parms.get_parm("x_dim"))
        self.x3 = None
        self.x  = None

    def forward(self,inputs):
        self.z = inputs.reshape(1, self.parms.get_parm("z_dim"))

        self.x1 = np.matmul(self.z , self.w1) + self.b1
        self.x1 = self.calculus.get_function("relu")(self.x1)

        self.x2 = np.matmul(self.x1,self.w2) + self.b2
        self.x2 = self.calculus.get_function("relu")(self.x2)

        self.x3 = np.matmul(self.x2,self.w3) + self.b3
        self.x = self.calculus.get_function("tanh")(self.x3)

        return self.x

    def backward(self,yhat) :  
        delta = yhat
        delta *= self.calculus.get_derivative("tanh")(self.x)

        d_w3 = np.matmul(np.transpose(self.x2),delta)
        d_b3 = delta.copy()

        delta = np.matmul(delta, np.transpose(self.w3))

        if (np.linalg.norm(d_w3) > self.parms.get_parm("gradient_clip")) :
            d_w3 = self.parms.get_parm("gradient_clip") / np.linalg.norm(d_w3) * d_w3
        self.w3 -= self.parms.get_parm("step_size_G") * d_w3
        self.w3 = np.maximum(-self.parms.get_parm("weight_clip"),np.minimum(self.parms.get_parm("weight_clip"),self.w3))
        
        self.b3 -= self.parms.get_parm("step_size_G") * d_b3
        self.b3 = np.maximum(-self.parms.get_parm("weight_clip"), np.minimum(self.parms.get_parm("weight_clip"),self.b3))
        delta *= self.calculus.get_derivative("relu")(self.x2)

        d_w2 = np.matmul(np.transpose(self.x1),delta)
        d_b2 = delta.copy()

        delta = np.matmul(delta, np.transpose(self.w2))

        if (np.linalg.norm(d_w2) > self.parms.get_parm("gradient_clip")) :
            d_w2 = self.parms.get_parm("gradient_clip") / np.linalg.norm(d_w2) * d_w2
        self.w2 -= self.parms.get_parm("step_size_G") * d_w2
        self.w2 = np.maximum(-self.parms.get_parm("weight_clip"),np.minimum(self.parms.get_parm("weight_clip"),self.w2))
        
        self.b2 -= self.parms.get_parm("step_size_G") * d_b2
        self.b2 = np.maximum(-self.parms.get_parm("weight_clip"), np.minimum(self.parms.get_parm("weight_clip"),self.b2))
        delta *= self.calculus.get_derivative("relu")(self.x1)

        d_w1 = np.matmul(np.transpose(self.z),delta)
        d_b1 = delta.copy()

        if (np.linalg.norm(d_w1) > self.parms.get_parm("gradient_clip")) :
            d_w1 = self.parms.get_parm("gradient_clip") / np.linalg.norm(d_w1) * d_w1
        self.w1 -= self.parms.get_parm("step_size_G") * d_w1
        self.w1 = np.maximum(-self.parms.get_parm("weight_clip"),np.minimum(self.parms.get_parm("weight_clip"),self.w1))
        
        self.b1 -= self.parms.get_parm("step_size_G") * d_b1
        self.b1 = np.maximum(-self.parms.get_parm("weight_clip"), np.minimum(self.parms.get_parm("weight_clip"),self.b1))
        delta *= self.calculus.get_derivative("relu")(self.x1)

class Discriminator():
    def __init__(self,parms,calculus):
        self.parms = parms
        self.calculus = calculus
        self.x  = None
        self.w1 = weight_initializer(in_channels = self.parms.get_parm("x_dim"), out_channels = self.parms.get_parm("d_hidden"))
        self.b1 = weight_initializer(in_channels = 1, out_channels = self.parms.get_parm("d_hidden"))
        self.y1 = None
        self.w2 = weight_initializer(in_channels = self.parms.get_parm("d_hidden"), out_channels = self.parms.get_parm("d_hidden"))
        self.b2 = weight_initializer(in_channels = 1, out_channels = self.parms.get_parm("d_hidden"))
        self.y2 = None
        self.w3 = weight_initializer(in_channels = self.parms.get_parm("d_hidden"), out_channels = self.parms.get_parm("z_dim"))
        self.b3 = weight_initializer(in_channels = 1, out_channels = self.parms.get_parm("z_dim"))
        self.y3 = None
        self.y  = None

    def forward(self,inputs):
        self.x = inputs.reshape(1,self.parms.get_parm("x_dim"))
        self.y1 = np.matmul(self.x,self.w1) + self.b1
        self.y1 = self.calculus.get_function("leaky_relu")(self.y1)
        self.y2 = np.matmul(self.y1,self.w2) + self.b2
        self.y2 = self.calculus.get_function("leaky_relu")(self.y2)
        self.y3 = np.matmul(self.y2,self.w3) + self.b3
        self.y = self.calculus.get_function("sigmoid")(self.y3)
        return self.y

    def backward(self,outputs,apply_grad=True):
        delta = outputs
        delta *= self.calculus.get_derivative("sigmoid")(self.y)
        d_w3 = np.matmul(np.transpose(self.y2),delta)
        d_b3 = delta.copy()
        delta = np.matmul(delta,np.transpose(self.w3))
        if apply_grad:
            if np.linalg.norm(d_w3) > self.parms.get_parm("gradient_clip"):
                d_w3 = self.parms.get_parm("gradient_clip") / np.linalg.norm(d_w3) * d_w3
            self.w3 += self.parms.get_parm("step_size_D") * d_w3
            self.w3 = np.maximum(-self.parms.get_parm("weight_clip"),np.minimum(self.parms.get_parm("weight_clip"), self.w3))
            self.b3 += self.parms.get_parm("step_size_D") * d_b3
            self.b3 = np.maximum(-self.parms.get_parm("weight_clip"),np.minimum(self.parms.get_parm("weight_clip"), self.b3))
        delta *= self.calculus.get_derivative("leaky_relu")(self.y2)
        d_w2 = np.matmul(np.transpose(self.y1),delta)
        d_b2 = delta.copy()
        delta = np.matmul(delta,np.transpose(self.w2))
        if apply_grad:
            if np.linalg.norm(d_w2) > self.parms.get_parm("gradient_clip"):
                d_w2 = self.parms.get_parm("gradient_clip") / np.linalg.norm(d_w2) * d_w2
            self.w2 += self.parms.get_parm("step_size_D") * d_w2
            self.w2 = np.maximum(-self.parms.get_parm("weight_clip"),np.minimum(self.parms.get_parm("weight_clip"), self.w2))
            self.b2 += self.parms.get_parm("step_size_D") * d_b2
            self.b2 = np.maximum(-self.parms.get_parm("weight_clip"),np.minimum(self.parms.get_parm("weight_clip"), self.b2))
        delta *= self.calculus.get_derivative("leaky_relu")(self.y1)
        d_w1 = np.matmul(np.transpose(self.x),delta)
        d_b1 = delta.copy()
        delta = np.matmul(delta,np.transpose(self.w1))
        if apply_grad:
            if np.linalg.norm(d_w1) > self.parms.get_parm("gradient_clip"):
                d_w1 = self.parms.get_parm("gradient_clip") / np.linalg.norm(d_w1) * d_w1
            self.w1 += self.parms.get_parm("step_size_D") * d_w1
            self.w1 = np.maximum(-self.parms.get_parm("weight_clip"),np.minimum(self.parms.get_parm("weight_clip"), self.w1))
            self.b1 += self.parms.get_parm("step_size_D") * d_b1
            self.b1 = np.maximum(-self.parms.get_parm("weight_clip"),np.minimum(self.parms.get_parm("weight_clip"), self.b1))
        return delta

def run():
    table = {"z_dim":1,
             "g_hidden":10,
             "x_dim":10,
             "d_hidden":10,
             "step_size_G":0.01,
             "step_size_D":0.01,
             "iter_num":50000,
             "gradient_clip":0.2,
             "weight_clip":0.25}
    parms = Constants(table)
    calculus = Calculus()
    print(parms.get_parm("step_size_G"))
    G = Generator(parms,calculus)
    D = Discriminator(parms,calculus)
    loss_function = LossFunction()
    real_label = 1
    fake_label = 0
    for i in range(parms.get_parm("iter_num")):
        #update D with real data
        x_real = get_samples(parms,True)
        y_real = D.forward(x_real)
        loss_D_r = loss_function.forward(y_real,real_label)
        d_loss_D = loss_function.backward()
        D.backward(d_loss_D)
        #update D with fake data
        z_noise = np.random.randn(parms.get_parm("z_dim"))
        x_fake = G.forward(z_noise)
        y_fake = D.forward(x_fake)
        loss_D_f = loss_function.forward(y_fake,fake_label)
        d_loss_D = loss_function.backward()
        D.backward(d_loss_D)
        #update G with fake data
        y_fake_r = D.forward(x_fake)
        loss_G = loss_function.forward(y_fake_r,real_label)
        d_loss_G = D.backward(loss_G, apply_grad=False)
        G.backward(d_loss_G)
        loss_D = loss_D_r + loss_D_f
        if i % 100 == 0:
            print('{} {} {}'.format(loss_D_r.item((0, 0)), loss_D_f.item((0,0)), loss_G.item((0, 0))))

    x_axis = np.linspace(0, 10, 10)
    for i in range(50):
        z_noise = np.random.randn(parms.get_parm("z_dim"))
        x_fake = G.forward(z_noise)
        plt.plot(x_axis, x_fake.reshape(parms.get_parm("x_dim")))
    plt.ylim((-1, 1))
    plt.show() 

if __name__ == "__main__" :
    run()
