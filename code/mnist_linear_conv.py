import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
from time import time
import matplotlib.pyplot as plt


#0 : Some utility functions
def get_correct_count(y_hat_train,y_train) :
    y_hat_train = (y_hat_train > 0).astype(int)
    correct_count = 0
    for i,_ in enumerate(y_train) : 
        correct_count += 1 if np.all(y_hat_train[i] == y_train[i]) else 0
    return correct_count

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
      #2. computes the loss
      loss = self.loss_function(yhat,y)
      #3. computes gradient
      loss.backward()
      #4. updates parameters
      self.optimizer.step()
      self.optimizer.zero_grad()
      predicted = get_correct_count(yhat.detach().cpu().numpy(), y.detach().cpu().numpy())
      #print("NDDEBUG %d"%(predicted))
      return loss.item(), predicted
    return dummy_trainer

  def make_validation_function(self):
    def dummy_validator(x,y) :
      #0. sets model to eval mode
      self.model.eval()
      #1. compute output
      yhat = self.model(x)
      #2. compute loss
      loss = self.loss_function(yhat,y)
      predicted = get_correct_count(yhat.detach().cpu().numpy(),y.detach().cpu().numpy())
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
      print("Loss after epoch %d is %.6f"%(epoch,avg_loss))

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
    return y_hat_tensor.detach().cpu().numpy()
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

#2 : The various models
#2.1 : A custom linear model
class MyLinearLayer(nn.Module):
  def __init__(self,input_size,output_size):
    super(MyLinearLayer,self).__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.weight = nn.Parameter(torch.Tensor(output_size,input_size))
    self.bias = nn.Parameter(torch.Tensor(output_size))
    self.reset_parameters()
 
  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
    if self.bias is not None :
      fan_in,_ = nn.init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1/math.sqrt(fan_in)
      nn.init.uniform_(self.bias,-bound,bound)

  def forward(self,x) :
    return x.matmul(self.weight.t()) + self.bias

class MyCustomLinearFCModel(nn.Module):
  def __init__(self,input_size,hidden_states,output_size) :
    super(MyCustomLinearFCModel, self).__init__()
    self.conn_i_h1 = MyLinearLayer(input_size,hidden_states)
    #(16 X 768) -> (16 X hidden_states)
    self.conn_h1_o = MyLinearLayer(hidden_states,output_size)
    #(16 X hidden_states) -> (16 X output_size) 

  def forward(self,x):
    out = self.conn_i_h1(x)
    out = nn.functional.relu(out)
    out = self.conn_h1_o(out)
    #out = nn.functional.softmax(out) # required only with MyLoss and nn.MSELoss
    return out

#2.2 : A convolutional model
class MyConvModel(nn.Module):
  def __init__(self,output_size) : 
    super(MyConvModel,self).__init__()
    self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels = 32, kernel_size = 3) #MNIST data is only 1 channel
    # (16 X 1 X 28 X 28) -> (16 X 32 X 26 X 26)  Hout = floor((Hin + 2*p - f)/s + 1); Wout = floor((Win + 2*p - f)/s + 1)
    self.relu1 = nn.ReLU()
    self.max_pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
    # (16 X 32 X 26 X 26) -> (16 X 32 X 13 X 13) Hout = floor((Hin + 2*p - f)/s + 1); Wout = floor((Win + 2*p - f)/2 + 1)

    self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels = 16, kernel_size = 3)
    # (16 X 32 X 13 X 13) -> (16 X 64 X 11 X 11)
    self.relu2 = nn.ReLU()
    self.max_pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
    # (16 X 64 X 11 X 11) -> (16 X 64 X 5 X 5)

    #after flattening  (16 X 400)
    self.fc1 = nn.Linear(400,65)
    self.relu3 = nn.ReLU()
    self.fc2 = nn.Linear(65,10)
 
  def forward(self,x) :
    out = self.conv_layer1(x)
    out = self.relu1(out)
    out = self.max_pool1(out)
    out = self.conv_layer2(out)
    out = self.relu2(out)
    out = self.max_pool2(out)
    
    out = out.reshape(out.size(0),-1) #flatten
    
    
    out = self.fc1(out)
    out = self.relu3(out)
    out = self.fc2(out)
    return out

#2.3 : A linear layer using nn.Linear
class MyLinearFCModel(nn.Module):
  def __init__(self,input_size,hidden_states,output_size) :
    super(MyLinearFCModel, self).__init__()
    self.conn_i_h1 = nn.Linear(input_size,hidden_states)
    #(16 X 768) -> (16 X hidden_states)
    self.conn_h1_o = nn.Linear(hidden_states,output_size)
    #(16 X hidden_states) -> (16 X output_size) 

  def forward(self,x):
    out = self.conn_i_h1(x)
    out = nn.functional.relu(out)
    out = self.conn_h1_o(out)
    #out = nn.functional.softmax(out) # required only with MyLoss and nn.MSELoss
    return out

#3 Custom loss functions if used
#3.1 A simple mean loss function
class MyLoss(nn.Module):
  def __init__(self):
    super(MyLoss,self).__init__()

  def forward(self,inputs,targets):
    loss = torch.mean((inputs-targets)**2)
    return loss

#4. The callback system
#4.1 : The training logger
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


#4.2 : The early stopping callback
class EarlyStopping():
    def __init__(self,patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = None

    def on_epoch_end(self, epoch, logs = None):
        current_loss = logs['loss']
        if self.best_loss is None or current_loss < self.best_loss :
            self.best_loss = current_loss
            self.counter = 0
        else :
            self.counter += 1
            if self.counter >= self.patience :
                logging.info("Early stopping required.")
                return True
        return False


 
#5. The data processing functions
#5.1 : Load data
def load_data(num_examples):
    def make_one_hot_labels(examples):
       num_examples = len(examples)
       one_hot_labels = np.zeros((num_examples,10))
       for i,l in enumerate(examples):
         one_hot_labels[i][l] = 1
       return one_hot_labels

    assert num_examples >= 1, "Use atleast one example"
    # Data generation process
    path = 'D:/work/deep-learning/grokking/work/mnist/mnist.npz'
    with np.load(path,allow_pickle=True) as mnistData:
      x_train,y_train = mnistData['x_train'],mnistData['y_train']
      x_test,y_test = mnistData['x_train'],mnistData['y_train']

    # Displaying the data that we generated (first 5 values)
    #print("X:", x[:5], "\n\nY:", y[:5])
    x_train = x_train[0:num_examples]
    y_train = make_one_hot_labels(y_train[0:num_examples])

    x_test = x_test[0:num_examples]
    y_test = make_one_hot_labels(y_test[0:num_examples])

    return x_train,y_train,x_test,y_test 

#5.2 : For CNN add an extra dimension
def unsqueeze_data(x_train,x_test) :
    num_train_examples = len(x_train)
    #x_train = x_train[0:num_train_examples].unsqueeze_data(1)
    x_train = np.expand_dims(x_train[0:num_train_examples],axis=1)
    num_test_examples = len(x_test)
    #x_test = x_test[0:num_test_examples].unsqueeze_data(1)
    x_test = np.expand_dims(x_test[0:num_test_examples],axis=1)
    return x_train,x_test

#5.3 : For FC NN flatten the data
def flatten_data(x_train,x_test) :
    num_train_examples = len(x_train)
    x_train = x_train[0:num_train_examples].reshape(num_train_examples,28*28)/255
    num_test_examples = len(x_test)
    x_test = x_test[0:num_test_examples].reshape(num_test_examples,28*28)/255
    return x_train,x_test

#6 : Create the data loader
def prepare_data(x_train,y_train,x_test,y_test):
    x_tensor = torch.as_tensor(x_train).float()                          # 1)
    y_tensor = torch.as_tensor(y_train).float()                          # 1)
    dataset = TensorDataset(x_tensor,y_tensor)
    x_test_tensor = torch.as_tensor(x_test).float()
    y_test_tensor = torch.as_tensor(y_test).float()
    test_dataset = TensorDataset(x_test_tensor,y_test_tensor)
    ################################################################
    #step 2 : train and validation set split
    ################################################################
    #Shuffles the indices
    ratio = .8
    n_total = len(dataset)
    n_train = int(n_total * ratio)
    n_val = n_total - n_train
    train_data, val_data = random_split(dataset, [n_train, n_val])  # 2)
    
    train_loader = DataLoader(dataset=train_data,batch_size=16,shuffle=True)
    val_loader = DataLoader(dataset=val_data,batch_size=16)
    test_loader = DataLoader(dataset=test_dataset,batch_size=16)

    return train_loader,val_loader,test_loader

#7 : Validate the prediction
def validate(mls,x_train,y_train) :
  y_hat_train = mls.predict(x_train)
  return get_correct_count(y_hat_train,y_train)

#8 : Plot graphs
def print_plots():
   loss_values = [logs['loss'] for logs in training_logs]
   accuracy_values = [logs['accuracy'] for logs in training_logs]
   epochs = range(1, len(loss_values) + 1)
   
   # Plot loss over epochs
   plt.figure(figsize=(10, 5))
   plt.plot(epochs, loss_values, label='Loss', marker='o')
   plt.title('Training Loss Over Epochs')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend()
   plt.grid(True)
   plt.show()
   
   # Plot accuracy over epochs
   plt.figure(figsize=(10, 5))
   plt.plot(epochs, accuracy_values, label='Accuracy', marker='o', color='green')
   plt.title('Training Accuracy Over Epochs')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.grid(True)
   plt.show()

#9 : run linear model
def run_linear() :
  x_train_orig,y_train,x_test_orig,y_test = load_data(1000)
  x_train,x_test = flatten_data(x_train_orig,x_test_orig)

  train_loader,val_loader,test_loader = prepare_data(x_train,y_train,x_test,y_test)
  lr = 0.1
  torch.manual_seed(42)

  # model = nn.Sequential(nn.Linear(1,1))
  model = MyLinearFCModel(x_train[0].size,100,y_train[0].size)
  optimizer = optim.SGD(model.parameters(),lr=lr)
  #loss_function = nn.MSELoss(reduction='mean') #to use this add a softmax layer at the output
  loss_function = nn.BCEWithLogitsLoss(reduction='mean') #no need to add a softmax layer at the output
  #loss_function = MyLoss() #to use this add a softmax layer at the output
  callback = TrainingLogger(log_interval = 10)
  early_stopper = EarlyStopping()

  mls = MLSystem(model,optimizer,loss_function,callback,early_stopping_callback=early_stopper)
  mls.set_loaders(train_loader=train_loader,val_loader=val_loader)

  mls.train(n_epochs=100)

  # new_data = np.array([.5, .3, .7]).reshape(-1, 1)
  new_data = x_test[0]
  #print(new_data)
  predictions = mls.predict(new_data)
  print(predictions)
  print(y_test[0])

  correct = validate(mls,x_test,y_test)
  print(correct)
  print_plots()

#10 : run convolutional model
def run_conv():
  x_train_orig,y_train,x_test_orig,y_test = load_data(1000)
  x_train,x_test = unsqueeze_data(x_train_orig, x_test_orig)
  
  train_loader,val_loader,test_loader = prepare_data(x_train,y_train,x_test,y_test)
  lr = 0.1
  torch.manual_seed(42)

  assert y_train[0].size == 10 , "foobar "
  model = MyConvModel(y_train[0].size)
  optimizer = optim.SGD(model.parameters(),lr=lr)
  loss_function = nn.BCEWithLogitsLoss(reduction='mean') #no need to add a softmax layer at the output
  callback = TrainingLogger(log_interval = 10)
  early_stopper = EarlyStopping()
  
  mls = MLSystem(model,optimizer,loss_function,callback,early_stopping_callback=early_stopper)
  mls.set_loaders(train_loader=train_loader,val_loader=val_loader)

  mls.train(n_epochs=100)
  #mls.train(n_epochs=10)

  new_data = x_test[0]
  #new_data = np.expand_dims(x_test_orig[0],axis=(0,1))
  new_data = np.expand_dims(x_test[0],axis=1)
  predictions = mls.predict(new_data)
  print(predictions)
  print(y_test[0])

  correct = validate(mls,x_test,y_test)
  print(model.state_dict())
  print(correct)
  print_plots()

#11: run custom linear model
def run_custom_linear() :
  x_train_orig,y_train,x_test_orig,y_test = load_data(1000)
  x_train,x_test = flatten_data(x_train_orig,x_test_orig)

  train_loader,val_loader,test_loader = prepare_data(x_train,y_train,x_test,y_test)
  lr = 0.1
  torch.manual_seed(42)

  # model = nn.Sequential(nn.Linear(1,1))
  model = MyCustomLinearFCModel(x_train[0].size,100,y_train[0].size)
  optimizer = optim.SGD(model.parameters(),lr=lr)
  #loss_function = nn.MSELoss(reduction='mean') #to use this add a softmax layer at the output
  loss_function = nn.BCEWithLogitsLoss(reduction='mean') #no need to add a softmax layer at the output
  #loss_function = MyLoss() #to use this add a softmax layer at the output
  callback = TrainingLogger(log_interval = 10)
  early_stopper = EarlyStopping()

  mls = MLSystem(model,optimizer,loss_function,callback,early_stopping_callback=early_stopper)
  mls.set_loaders(train_loader=train_loader,val_loader=val_loader)

  mls.train(n_epochs=100)

  # new_data = np.array([.5, .3, .7]).reshape(-1, 1)
  new_data = x_test[0]
  #print(new_data)
  predictions = mls.predict(new_data)
  print(predictions)
  print(y_test[0])

  correct = validate(mls,x_test,y_test)
  print(correct)
  print_plots()


#run_linear()
#run_conv()
run_custom_linear()
