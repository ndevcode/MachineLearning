# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:32:40 2023

@author: ndev
"""

import numpy as np
np.random.seed(1)

def generate_image(m,H,W,C):
    the_image = np.random.randn(m,H,W,C)
    min_val = the_image.min()
    max_val = the_image.max()
    the_image = ((the_image - min_val) * (1./(max_val - min_val)) * 255).astype('uint8')
    return the_image

def pad_image(the_image,pad):
    (m, H, W, C) = the_image.shape
    padded_image = np.pad(the_image,
                          ((0,0),(pad,pad),(pad,pad),(0,0)),
                          mode='constant',
                          constant_values = (0,0)
                          )
    assert (padded_image.shape == (m,H + 2*pad,W + 2*pad, C))
    return padded_image

def conv_single_step(a_slice_prev,W,b):
    print(a_slice_prev.shape)
    print(W.shape)
    assert(W.shape == a_slice_prev.shape)
    s = np.multiply(a_slice_prev,W)
    Z = np.sum(s)
    Z = Z + b.astype('float')
    return Z

def pool_max(a_slice_prev,W,b):
    return np.max(a_slice_prev)

def pool_average(a_slice_prev,W,b):
    return np.mean(a_slice_prev)
    
def generate_windows(n_H,n_W,f,stride):
    windows = []
    for h in range(n_H):
        vert_start = h * stride
        vert_end = vert_start + f
        for w in range(n_W):
            horiz_start = w * stride
            horiz_end = horiz_start + f
            windows.append((h,w,vert_start,vert_end,horiz_start,horiz_end))
    return windows


def apply_forward_fn(A_prev, W, b, stride, pad, fn):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f,f,n_C_prev,n_C) = W.shape
    #stride = hparameters["stride"]
    #pad = hparameters["pad"]
    print("b shape",b.shape)
    
    n_H = int((n_H_prev - f + 2*pad)/stride)+1
    n_W = int((n_W_prev - f + 2*pad)/stride)+1
    
    Z = np.zeros((m,n_H,n_W,n_C))
    A_prev_pad = pad_image(A_prev,pad)
    
    windows = generate_windows(n_H, n_W, f, stride)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h,w,vert_start, vert_end, horiz_start, horiz_end in windows:
                for c in range(n_C):
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    weights = W[:,:,:,c]
                    biases = b[:,:,:,c]
                    Z[i,h,w,c] = fn(a_slice_prev,weights,biases)
    
    assert(Z.shape == (m,n_H,n_W,n_C))
    return Z

def conv_forward(A_prev, W, b, hparameters):
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    return apply_forward_fn(A_prev,W,b,stride,pad,conv_single_step)

def pool_forward(A_prev, W, b, hparameters, mode="max"):
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    return apply_forward_fn(A_prev,W,b,stride,pad,pool_max)

def relu(x):
    return (x >= 0)*x

def conv_backward(dZ,A_prev,W,b,hparameters):
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (f,f,n_C_prev,n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m,n_H,n_W,n_C) = dZ.shape
    
    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dW = np.zeros((f,f,n_C_prev,n_C))
    db = np.zeros((1,1,1,n_C))
    
    A_prev_pad = pad_image(A_prev,pad)
    dA_prev_pad = pad_image(dA_prev,pad)
    
    windows = generate_windows(n_H, n_W, f, stride)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h,w,vert_start, vert_end, horiz_start, horiz_end in windows:
            for c in range(n_C):
                a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                da_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:] += (W[:,:,:,c] * dZ[i,h,w,c])
                dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                db[:,:,:,c] += dZ[i,h,w,c]
        dA_prev[i,:,:,:] = da_prev_pad[pad:-pad,pad:-pad,:]
    
    return dA_prev,dW,db

def create_mask_from_window(x):
    mx = np.max(x)
    mask = (x == mx)
    return mask

def distribute_value(dz, shape):
    (H,W) = shape
    average = (dz/(H*W))
    a = np.ones((H,W))*average
    return a

def pool_backward(dA,A_prev,hparameters,mode="max"):
    f = hparameters["f"]
    
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    
    windows = generate_windows(n_H, n_W, f, stride=1)
     
    for i in range(m):
        a_prev = A_prev[i]
        for h,w,vert_start, vert_end, horiz_start, horiz_end in windows:
            for c in range(n_C):
                if (mode == "max"):
                    a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end,c]
                    mask = create_mask_from_window(a_prev_slice)
                    dA_prev[i,vert_start:vert_end, horiz_start:horiz_end,c] += np.multiply(mask,dA[i,h,w,c])
                elif (mode == "average"):
                    da = dA[i,h,w,c]
                    shape = (f,f)
                    dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += distribute_value(da, shape)

    return dA_prev


W = np.random.randn(3,3,3,2)
b = np.random.randn(1,1,1,2)
an_image = generate_image(1,28,28,3)

hparameters = {}
hparameters["stride"] = 2
hparameters["pad"] = 1
hparameters["f"] = 3

Z = conv_forward(an_image, W, b, hparameters)
A = relu(Z)

Y = generate_image(1,14,14,2)

print("final shape = ",A.shape)
M = pool_forward(A,W,b,hparameters)
print("after pooling shape = ",M.shape)
    
dZ = A - Y
dA, dW, db = conv_backward(dZ,an_image,W,b,hparameters)
print("dA shape = ",dA.shape)
print("dW shape = ",dW.shape)
print("db shape = ",db.shape)
assert(dA.shape == an_image.shape)
assert(dW.shape == W.shape)
assert(db.shape == b.shape)

dA1 = np.random.randn(1,14,14,2)
dA1_prev = pool_backward(dA1,an_image,hparameters)
print("dA1_prev shape = ",dA1_prev.shape)
