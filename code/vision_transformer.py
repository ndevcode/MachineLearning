# -*- coding: utf-8 -*-
"""
Created on Sat Apr 05 19:47:35 2025

@author: ndev
"""

#1. Import libraries

import re
import os
import time
import itertools
import math
import logging
from time import time
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#2. Load the dataset

########################################################################################################
#  PART 0 : The ML Training system
########################################################################################################
class MLSystem():
  def __init__(self,model,optimizer,loss_function, callback, early_stopping_callback = None):
    self.model = model
    self.optimizer = optimizer
    self.loss_function = loss_function
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model.to(self.device)
    self.callback = callback
    self.early_stopping_callback = early_stopping_callback

    #the data loaders
    self.train_loader = None
    self.val_loader = None

    #the losses
    self.train_losses = []
    self.val_losses = []
    self.total_epochs = 0

    #the functions for the training and validation steps
    self.training_function = self.make_training_function()
    self.validation_function = self.make_validation_function()

  def get_correct_count(self,y_pred, y_true) :
    pred_classes = torch.argmax(y_pred, dim=1)
    correct = (pred_classes == y_true).sum().item()
    return correct


  def to(self,device) :
    self.device = device
    self.model.to(self.device)

  def set_loaders(self,train_loader,val_loader=None):
    self.train_loader = train_loader
    self.val_loader = val_loader

  def make_training_function(self) : 
    def dummy_trainer(x,y) :
      #0. sets model to train mode
      self.model.train()
      #1. compute model predictions
      #print(x.shape)
      yhat = self.model(x)  
      #2. computes the loss
      loss = self.loss_function(yhat,y)
      #3. computes gradient
      loss.backward()
      #4. updates parameters
      self.optimizer.step()
      self.optimizer.zero_grad()
      predicted = self.get_correct_count(yhat, y)
      return loss.item(), predicted
    return dummy_trainer

  def make_validation_function(self):
    def dummy_validator(x,y) :
      with torch.no_grad() :
        #0. sets model to eval mode
        self.model.eval()
        #1. compute output
        yhat = self.model(x)
        #2. compute loss
        loss = self.loss_function(yhat,y)
        predicted = self.get_correct_count(yhat,y)
      return loss.item(),predicted
    return dummy_validator

  def run_mini_batch(self,validation=False):
    data_loader = self.val_loader if validation==True else self.train_loader
    step_to_run = self.validation_function if validation==True else self.training_function
    if data_loader is None :
      return None
    mini_batch_losses = []
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0
    for batch_idx, (x_batch, y_batch) in enumerate(data_loader) :
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        mini_batch_loss,predicted = step_to_run(x_batch,y_batch)
        running_loss += mini_batch_loss
        #print(mini_batch_loss)
        mini_batch_losses.append(mini_batch_loss)
        correct_predictions += predicted
        total_predictions += len(y_batch)
        self.callback.on_batch_end(batch_idx,logs={
              'loss' : mini_batch_loss / (batch_idx + 1),
              'accuracy' : predicted / len(y_batch)
            })
    
    avg_loss = np.mean(mini_batch_losses)
    return avg_loss, running_loss, correct_predictions, total_predictions
  
  def set_seed(self,seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

  def train(self, n_epochs, seed=42) :
    #print("using : %s"%(self.device))
    self.set_seed(seed)
    for epoch in range(n_epochs) :
      self.callback.on_epoch_begin(epoch)
      self.total_epochs += 1

      #1. run mini batch
      avg_loss,running_loss,correct_predictions,total_predictions = self.run_mini_batch(validation=False)
      self.train_losses.append(avg_loss)

      #2. run validation
      with torch.no_grad() :
        avg_val_loss,running_val_loss,correct_val_predictions,total_val_predictions = self.run_mini_batch(validation=True) # need to track validation loss ?
        self.val_losses.append(avg_val_loss)

      logs = {
        'loss' : running_loss / len(self.train_loader),
        'accuracy' : correct_predictions / total_predictions,
        'val_loss' : running_val_loss / len(self.val_loader)
      }
      self.callback.on_epoch_end(self.save_checkpoint,epoch,logs)
      if self.early_stopping_callback and self.early_stopping_callback.on_epoch_end(epoch, logs = { 
                'val_loss':running_val_loss,
                'model':self.model,
                'optimizer': self.optimizer
             } ) : 
         break
      #print("Loss after epoch %d is %.6f"%(epoch,avg_loss))

  def predict(self,x) :
    #0. set model in eval
    status = self.model.training
    self.model.eval()
    x_tensor = torch.as_tensor(x).float()
    y_hat_tensor = self.model(x_tensor.to(self.device))
    #1. reset model to train again
    if status == True :
        self.model.train()
    return y_hat_tensor

  def save_checkpoint(self,filename) :
     checkpoint = {
       'epoch' : self.total_epochs,
       'model_state_dict' : self.model.state_dict(),
       'optimizer_state_dict' : self.optimizer.state_dict(),
       'loss' : self.train_losses,
       'val_loss' : self.val_losses
     }
     torch.save(checkpoint,filename)

training_logs = []

class TrainingLogger:
    def __init__(self, log_interval=10, dump_checkpoint_interval=4, log_dir='runs/experiment'):
        self.log_interval = log_interval
        self.dump_checkpoint_interval = dump_checkpoint_interval
        self.filenameIncr = 0
        self.writer = SummaryWriter(log_dir)
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.current_epoch = 0

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time()
        self.current_epoch = epoch
        logging.info(f"Epoch {epoch + 1} starting.")

    def on_epoch_end(self, save_checkpoint_function, epoch, logs=None):
        if (epoch + 1) % self.dump_checkpoint_interval == 0:
            logging.info(f" Dumping checkpoint at end of interval {epoch+1}")
            filename = "model_checkpoint_" + str(self.filenameIncr) + ".pth"
            save_checkpoint_function(filename)
            self.filenameIncr += 1

        elapsed_time = time() - self.epoch_start_time
        logging.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logs['epoch_time'] = elapsed_time

        # Write logs to TensorBoard
        self.writer.add_scalar("Loss/train", logs['loss'], epoch)
        self.writer.add_scalar("Loss/val", logs['val_loss'], epoch)
        self.writer.add_scalar("Accuracy/train", logs['accuracy'], epoch)
        self.writer.add_scalar("Epoch Time", elapsed_time, epoch)

        training_logs.append(logs)

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_interval == 0:
            logging.info(f"           Batch {batch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")

    def close(self):
        self.writer.close()


#4.2 : The early stopping callback
class EarlyStopping():
    def __init__(self,patience=3,delta=0.0,checkpoint_path='checkpoint.pth',best_model_path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.checkpoint_path = checkpoint_path
        self.best_model_path = best_model_path
        self.early_stop = False

    def on_epoch_end(self, epoch, logs = None):
        current_loss = logs['val_loss']
        model = logs['model']
        optimizer = logs['optimizer']
        if self.best_loss is None or current_loss < self.best_loss :
            self.best_loss = current_loss
            self.counter = 0
            self.save_best_model(model,optimizer,epoch)
        else :
            self.counter += 1
            if self.counter >= self.patience :
                logging.info("Early stopping required.")
                self.early_stop = True
                return True
        return False

    def save_best_model(self,model,optimizer,epoch):
        torch.save({
          'epoch': epoch,
          'model_state_dict':model.state_dict(),
          'optimizer_state_dict':optimizer.state_dict(),
          'best_loss':self.best_loss
          }, self.best_model_path)

class ExperimentRunner:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None, 
                 log_interval=10, checkpoint_interval=4, patience=3, log_dir="runs/experiment"):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Setup callbacks with TensorBoard log dir
        self.logger = TrainingLogger(log_interval=log_interval,
                                     dump_checkpoint_interval=checkpoint_interval,
                                     log_dir=log_dir)
        self.early_stopping = EarlyStopping(patience=patience)

        # Initialize ML system
        self.ml_system = MLSystem(model=self.model,
                                  optimizer=self.optimizer,
                                  loss_function=self.loss_fn,
                                  callback=self.logger,
                                  early_stopping_callback=self.early_stopping)

        self.ml_system.set_loaders(train_loader, val_loader)

    def run(self, num_epochs=10, seed=42):
        self.ml_system.train(n_epochs=num_epochs, seed=seed)
        self.logger.close()

    def predict(self, x):
        return self.ml_system.predict(x)

    def save(self, filename="final_model.pth"):
        self.ml_system.save_checkpoint(filename)


########################################################################################################
#  PART 1 : Load the data
########################################################################################################

def load_data(path):
    # Data generation process
    with np.load(path,allow_pickle=True) as mnistData:
      x_train,y_train = mnistData['x_train'],mnistData['y_train']
      x_test,y_test = mnistData['x_test'],mnistData['y_test']

    # Displaying the data that we generated (first 5 values)
    #print("X:", x[:5], "\n\nY:", y[:5])

    return x_train,y_train,x_test,y_test 

def unsqueeze_data(x_train,x_test) :
    num_train_examples = len(x_train)
    #x_train = x_train[0:num_train_examples].unsqueeze_data(1)
    x_train = np.expand_dims(x_train[0:num_train_examples],axis=1)
    num_test_examples = len(x_test)
    #x_test = x_test[0:num_test_examples].unsqueeze_data(1)
    x_test = np.expand_dims(x_test[0:num_test_examples],axis=1)
    return x_train,x_test

def prepare_data(x_train,y_train,x_test,y_test):
    x_tensor = torch.as_tensor(x_train).float()                          # 1)
    y_tensor = torch.as_tensor(y_train).long()                          # 1)
    dataset = TensorDataset(x_tensor,y_tensor)
    x_test_tensor = torch.as_tensor(x_test).float()
    y_test_tensor = torch.as_tensor(y_test).long()
    test_dataset = TensorDataset(x_test_tensor,y_test_tensor)

    # train and validation split
    ratio = .8
    n_total = len(dataset)
    n_train = int(n_total * ratio)
    n_val = n_total - n_train
    train_data, val_data = random_split(dataset, [n_train, n_val])  # 2)
    
    train_loader = DataLoader(dataset=train_data,batch_size=16,shuffle=True)
    val_loader = DataLoader(dataset=val_data,batch_size=16)
    test_loader = DataLoader(dataset=test_dataset,batch_size=16)

    return train_loader,val_loader,test_loader

#3. Visualize the dataset

########################################################################################################
#  PART 2 : Visualize the data
########################################################################################################

def visualize_dataset(train_loader):
    """
    Visualize a few samples from the training dataset.
    """
    # Get a batch of data from the train_loader
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Convert images and labels to numpy for visualization
    images = images.numpy()
    labels = labels.numpy()

    print(images[0].shape)
    # Plot the first 6 images in the batch
    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i][0], cmap='gray')  # Assuming grayscale images
        ax.axis('off')
        ax.set_title(f"Label: {np.argmax(labels[i])}")  # Display the class label

    plt.show()

########################################################################################################
#  PART 3 : Create patches
########################################################################################################
def create_patches(x, patch_size=7):
    """
    Create patches from the input images.
    Args:
        x: Input images of shape (batch_size, channels, height, width)
        patch_size: Size of the patches to be created
    Returns:
        patches: Patches of shape (batch_size, num_patches, patch_size*patch_size*channels)
    """
    # shape (1,1,28,28)
    assert x.dim() == 4, "Input tensor must be of shape (batch_size, channels, height, width)"
    assert x.shape[2] % patch_size == 0 and x.shape[3] % patch_size == 0, "Height and width must be divisible by patch_size"
    batch_size, channels, height, width = x.shape # only for debug
    num_patches = (height // patch_size) * (width // patch_size) # only for debug

    # Reshape and permute to create patches
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # shape (1, 1, 4, 4, 7, 7)
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # (1, 4, 4, 1, 7, 7)
    patches = patches.reshape(batch_size, -1, patch_size * patch_size)  # (1, 16, 49)

    # shape (batch, num_patches, patch_size*patch_size*channels)
    # shape (1, 16, 49)
    return patches

########################################################################################################
#  PART 4 : Create the vision transformer model
########################################################################################################
class VisionTransformer(nn.Module):
    def __init__(self,patch_dim = 49, embed_dim = 128, num_patches = 16, num_layers = 4, num_heads = 4, num_classes = 10):
        super(VisionTransformer, self).__init__()
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Linear projection of patches
        self.linear_proj = nn.Linear(patch_dim, embed_dim)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))  # +1 for CLS token

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Create patches
        patchx = create_patches(x)
        batch_size, num_patches, _ = patchx.shape

        # Linear projection of patches
        patch_embeddings = self.linear_proj(patchx) # (batch_size, num_patches, embed_dim) (B, 16, 128)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch_size, 1, embed_dim) (B, 1, 128)

        # Concatenate CLS token and patch embeddings
        sequence = torch.cat((cls_tokens, patch_embeddings), dim=1) # (batch_size, num_patches+1, embed_dim) (B, 17, 128)
        sequence += self.positional_encoding  # Add positional encoding (batch_size, num_patches+1, embed_dim) (B, 17, 128)

        # Pass through transformer encoder
        encoded_sequence = self.transformer_encoder(sequence) # (batch_size, num_patches+1, embed_dim) (B, 17, 128)
        cls_output = encoded_sequence[:, 0]  # CLS token output (batch_size, embed_dim) (B, 128)
        output = self.classification_head(cls_output)
        return output
    
########################################################################################################
#  PART 5 : Run the training
########################################################################################################

def train_vision_transformer():
    path = '/home/ndev/ml/datasets/mnist.npz'
    x_train_orig,y_train,x_test_orig,y_test = load_data(path)
    x_train,x_test = unsqueeze_data(x_train_orig, x_test_orig)
    train_loader,val_loader,test_loader = prepare_data(x_train,y_train,x_test,y_test)
    #visualize_dataset(train_loader)
    model = VisionTransformer(patch_dim=49, embed_dim=128, num_patches=16, num_layers=4, num_heads=4, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    runner = ExperimentRunner(model=model,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            log_interval=10,
                            checkpoint_interval=4,
                            patience=3,
                            log_dir="runs/experiment")

    runner.run(num_epochs=10, seed=42)
    return runner,test_loader


########################################################################################################
#  PART 6 : Run the inference
########################################################################################################

def load_model(model,checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path,weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded from {checkpoint_path}, trained until epoch {checkpoint['epoch']}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Initializing a new model.")
    return model

def test_accuracy(device = "cuda",  model_path="best_model.pth"):
    path = '/home/ndev/ml/datasets/mnist.npz'
    x_train_orig,y_train,x_test_orig,y_test = load_data(path)
    x_train,x_test = unsqueeze_data(x_train_orig, x_test_orig)
    train_loader,val_loader,test_loader = prepare_data(x_train,y_train,x_test,y_test)
    model = VisionTransformer(patch_dim=49, embed_dim=128, num_patches=16, num_layers=4, num_heads=4, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    model = load_model(model,model_path)
    runner = ExperimentRunner(model=model,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            log_interval=10,
                            checkpoint_interval=4,
                            patience=3,
                            log_dir="runs/experiment")
    runner.ml_system.model.to(device)
    runner.ml_system.model.eval()

    correct, total = 0,0

    with torch.no_grad():
        for batch in test_loader :
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = runner.ml_system.predict(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct,total


#run the model
if __name__ == "__main__":
    runner,test_loader = train_vision_transformer()
    correct,total = test_accuracy()
    print(f"Test Accuracy: {correct}/{total} ({100 * correct / total:.2f}%)")
