# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:49:03 2024

@author: ndev
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import logging
from time import time


training_logs = []
class TrainingLogger :
    def __init__(self,log_interval=10, dump_checkpoint_interval = 25):
        self.log_interval = log_interval
        self.dump_checkpoint_interval = dump_checkpoint_interval
        self.filenameIncr = 0
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    def on_epoch_begin(self,epoch):
        self.epoch_start_time = time()
        logging.info(f"Epoch {epoch + 1} starting.")

    def on_epoch_end(self,save_checkpoint_function,epoch,logs=None):
        if (epoch + 1) % self.dump_checkpoint_interval == 0:
            logging.info(f" Dumping checkpoint at end of interval {epoch+1}")
            filename = "model_checkpoint_" + str(self.filenameIncr) + ".pth"
            save_checkpoint_function(filename)
            self.filenameIncr += 1
        elapsed_time = time() - self.epoch_start_time
        logging.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logs['epoch_time'] = elapsed_time
        training_logs.append(logs)

    def on_batch_end(self,batch,logs=None):
        if (batch + 1) % self.log_interval == 0:
           logging.info(f"           Batch {batch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")

#1. The ML system that runs training and prediction
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
      _,predicted = torch.max(yhat.data,1)
      correct = (predicted == y).sum().item()
      #2. computes the loss
      loss = self.loss_function(yhat,y)
      #3. computes gradient
      loss.backward()
      #4. updates parameters
      self.optimizer.step()
      self.optimizer.zero_grad()
      #print("NDDEBUG %d"%(predicted))
      return loss.item(),correct
    return dummy_trainer

  def make_validation_function(self):
    def dummy_validator(x,y) :
      #0. sets model to eval mode
      self.model.eval()
      #1. compute output
      yhat = self.model(x)
      #2. compute loss
      loss = self.loss_function(yhat,y)
      _,predicted = torch.max(yhat.data,1)
      correct = (predicted == y).sum().item()
      return loss.item(),correct
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
    self.set_seed(seed)
    for epoch in range(n_epochs) :
      self.callback.on_epoch_begin(epoch)
      self.total_epochs += 1
      #1. run mini batch
      avg_loss,running_loss,correct_predictions,total_predictions = self.run_mini_batch(validation=False)
      self.train_losses.append(avg_loss)
      logs = {
        'loss' : running_loss / len(self.train_loader),
        'accuracy' : correct_predictions / total_predictions
      }
      self.callback.on_epoch_end(self.save_checkpoint,epoch,logs)
      if self.early_stopping_callback and self.early_stopping_callback.on_epoch_end(epoch, logs) : 
         break
      print("Loss after epoch %d is %.6f"%((epoch+1),avg_loss))

      #2. run validation
      with torch.no_grad() :
        avg_val_loss,_,_,_ = self.run_mini_batch(validation=True) # need to track validation loss ?
        self.val_losses.append(avg_val_loss)

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
    #return y_hat.item

  def save_checkpoint(self,filename) :
     checkpoint = {
       'epoch' : self.total_epochs,
       'model_state_dict' : self.model.state_dict(),
       'optimizer_state_dict' : self.optimizer.state_dict(),
       'loss' : self.train_losses,
       'val_loss' : self.val_losses
     }
     torch.save(checkpoint,filename)

def validate(mls, test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = mls.predict(images)
        total += len(outputs)
        _,predicted = torch.max(outputs.data,1)
        correct += (predicted == labels).sum().item()
    return correct

def get_train_valid_loader(data_dir,batch_size,augment,random_seed = 42,valid_size = 0.1, shuffle = True):
    normalize = transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2023, 0.1994, 0.2010]
        )
    
    #define transforms
    validation_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize
        ])
    if augment:
        train_transform = transforms.Compose([
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
    else :
        train_transform = transforms.Compose([
                transforms.Resize((227,227)),
                transforms.ToTensor(),
                normalize
            ])
    #load the dataset
    train_dataset = datasets.CIFAR10(root=data_dir, train=True,download = True, transform = train_transform)
    validation_dataset = datasets.CIFAR10(root=data_dir, train=True, download = True, transform = validation_transform)
    #num_train = len(train_dataset)
    num_train = 256
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    train_idx, validation_idx = indices[split:],indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    
    train_loader = DataLoader(validation_dataset,batch_size=batch_size,sampler = train_sampler)
    validation_loader = DataLoader(train_dataset,batch_size=batch_size,sampler = validation_sampler)
    
    return train_loader, validation_loader

def get_test_loader(data_dir,batch_size,shuffle=True):
    normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    #define transforms
    transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize
        ])
    num_test = 100
    indices = list(range(num_test))
    test_sampler = SubsetRandomSampler(indices)
    dataset = datasets.CIFAR10(root=data_dir,train=False,download=True,transform=transform)
    data_loader = DataLoader(dataset,batch_size=batch_size, sampler = test_sampler)
    return data_loader

class MyAlexNet(nn.Module) :  
    def __init__(self,output_size) : #output size is 1000
        super(MyAlexNet,self).__init__()
        self.layer1 = nn.Sequential(
            #layer 1 : conv : (batch_size X 3 X 227 X 227) -> (batch_size X 96 X 55 X 55)
            nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4, padding = 0),
            #layer 2 : batch norm
            nn.BatchNorm2d(num_features = 96),
            #layer 3 : relu
            nn.ReLU(),
            #layer 4 : maxpool : (batch_size X 96 X 55 X 55) -> (batch_size X 96 X 27 X 27)
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            #layer 1 : conv : (batch_size X 96 X 27 X 27) -> (batch_size X 256 X 27 X 27)
            nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1, padding=2),
            #layer 2 : batch norm
            nn.BatchNorm2d(num_features = 256),
            #layer 3 : relu
            nn.ReLU(),
            #layer 4 : maxpool : (batch_size X 256 X 27 X 27) -> (batch_size X 256 X 13 X 13)
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            #layer 1 : conv : (batch_size X 256 X 13 X 13) -> (batch_size X 384 X 13 X 13)
            nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding=1),
            #layer 2 : batch norm
            nn.BatchNorm2d(num_features = 384),
            #layer 3 : relu
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            #layer 1 : conv : (batch_size X 384 X 13 X 13) -> (batch_size X 384 X 13 X 13)
            nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1, padding=1),
            #layer 2 : batch norm
            nn.BatchNorm2d(num_features = 384),
            #layer 3 : relu
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            #layer 1 : conv : (batch_size X 384 X 13 X 13) -> (batch_size X 256 X 13 X 13)
            nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1, padding=1),
            #layer 2 : batch norm
            nn.BatchNorm2d(num_features = 256),
            #layer 3 : relu
            nn.ReLU(),
            #layer 4 : maxpool : (batch_size X 256 X 13 X 13) -> (batch_size X 256 X 6 X 6)
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 9216, out_features=4096), # 256 X 6 X 6 = 9216
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 4096, out_features=4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = 4096, out_features = output_size)
        )
    def forward(self,x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = output.reshape(output.size(0),-1)
        output = self.fc(output)
        output = self.fc1(output)
        output = self.fc2(output)
        return output
     
def run():
    num_classes = 10
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.005
    train_loader,validation_loader = get_train_valid_loader(data_dir='./data', batch_size=batch_size, augment = False)
    test_loader = get_test_loader(data_dir='./data',batch_size=batch_size)
    model = MyAlexNet(num_classes)
    optimizer = optim.SGD(model.parameters(),lr=learning_rate)
    loss_function = nn.CrossEntropyLoss() #no need to add a softmax layer at the output
    callback = TrainingLogger(log_interval = 10)
    mls = MLSystem(model,optimizer,loss_function,callback)
    mls.set_loaders(train_loader=train_loader,val_loader=validation_loader)
    mls.train(n_epochs=num_epochs)
    correct = validate(mls,test_loader)
    print(correct)

run()
