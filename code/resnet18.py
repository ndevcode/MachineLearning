# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 09:28:21 2024

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


class TrainingLogger:
    def __init__(self, log_interval=10, dump_checkpoint_interval=25):
        self.log_interval = log_interval
        self.dump_checkpoint_interval = dump_checkpoint_interval
        self.filenameIncr = 0
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time()
        logging.info(f"Epoch {epoch + 1} starting.")

    def on_epoch_end(self, save_checkpoint_function, epoch, logs=None):
        if (epoch + 1) % self.dump_checkpoint_interval == 0:
            logging.info(f" Dumping checkpoint at end of interval {epoch+1}")
            filename = "model_checkpoint_" + str(self.filenameIncr) + ".pth"
            save_checkpoint_function(filename)
            self.filenameIncr += 1
        elapsed_time = time() - self.epoch_start_time
        logging.info(
            f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logs['epoch_time'] = elapsed_time
        training_logs.append(logs)

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_interval == 0:
            logging.info(f"           Batch {
                         batch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")

# 1. The ML system that runs training and prediction


class MLSystem():
    def __init__(self, model, optimizer, loss_function, callback, early_stopping_callback=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.callback = callback
        self.early_stopping_callback = early_stopping_callback

        # the data loaders
        self.train_loader = None
        self.val_loader = None

        # the losses
        self.train_losses = []
        self.val_losses = []
        self.total_epochs = 0

        # the functions for the training and validation steps
        self.training_function = self.make_training_function()
        self.validation_function = self.make_validation_function()

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def make_training_function(self):
        def dummy_trainer(x, y):
            # 0. sets model to train mode
            self.model.train()
            # 1. compute model predictions
            yhat = self.model(x)
            _, predicted = torch.max(yhat.data, 1)
            correct = (predicted == y).sum().item()
            # 2. computes the loss
            loss = self.loss_function(yhat, y)
            # 3. computes gradient
            loss.backward()
            # 4. updates parameters
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item(), correct
        return dummy_trainer

    def make_validation_function(self):
        def dummy_validator(x, y):
            # 0. sets model to eval mode
            self.model.eval()
            # 1. compute output
            yhat = self.model(x)
            # 2. compute loss
            loss = self.loss_function(yhat, y)
            _, predicted = torch.max(yhat.data, 1)
            correct = (predicted == y).sum().item()
            return loss.item(), correct
        return dummy_validator

    def run_mini_batch(self, validation=False):
        data_loader = self.val_loader if validation == True else self.train_loader
        step_to_run = self.validation_function if validation == True else self.training_function
        if data_loader is None:
            return None
        mini_batch_losses = []
        running_loss = 0.0
        total_predictions = 0
        correct_predictions = 0
        for batch_idx, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mini_batch_loss, predicted = step_to_run(x_batch, y_batch)
            running_loss += mini_batch_loss
            # print(mini_batch_loss)
            mini_batch_losses.append(mini_batch_loss)
            correct_predictions += predicted
            total_predictions += len(y_batch)
            self.callback.on_batch_end(batch_idx, logs={
                'loss': mini_batch_loss / len(y_batch),
                'accuracy': predicted / len(y_batch)
            })

        avg_loss = np.mean(mini_batch_losses)
        return avg_loss, running_loss, correct_predictions, total_predictions

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, n_epochs, seed=42):
        self.set_seed(seed)
        for epoch in range(n_epochs):
            self.callback.on_epoch_begin(epoch)
            self.total_epochs += 1
            # 1. run mini batch
            avg_loss, running_loss, correct_predictions, total_predictions = self.run_mini_batch(
                validation=False)
            self.train_losses.append(avg_loss)
            logs = {
                'loss': running_loss / len(self.train_loader),
                'accuracy': correct_predictions / total_predictions
            }
            self.callback.on_epoch_end(self.save_checkpoint, epoch, logs)
            if self.early_stopping_callback and self.early_stopping_callback.on_epoch_end(epoch, logs):
                break
            print("Loss after epoch %d is %.6f" % ((epoch+1), avg_loss))

            # 2. run validation
            with torch.no_grad():
                avg_val_loss, _, _, _ = self.run_mini_batch(
                    validation=True)  # need to track validation loss ?
                self.val_losses.append(avg_val_loss)

    def predict(self, x):
        # 0. set model in eval
        status = self.model.training
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # 1. reset model to train again
        if status == True:
            self.model.train()
        return y_hat_tensor
        # return y_hat.item

    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.train_losses,
            'val_loss': self.val_losses
        }
        torch.save(checkpoint, filename)


def validate(mls, test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = mls.predict(images)
        total += len(outputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    return correct


def get_train_valid_loader(data_dir, batch_size, augment, random_seed=42, valid_size=0.1, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    # define transforms
    validation_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize
        ])
    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    validation_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=validation_transform)
    # num_train = len(train_dataset)
    num_train = 256
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, validation_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, sampler=validation_sampler)

    return train_loader, validation_loader


def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # define transforms
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize
    ])
    num_test = 100
    indices = list(range(num_test))
    test_sampler = SubsetRandomSampler(indices)
    dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)
    return data_loader

class BasicBlock(nn.Module):
    def __init__(self,c_in,c_out,downSample = False, stride=1) :
        super(BasicBlock,self).__init__()
        self.downSample = downSample
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = c_in, out_channels = c_out, kernel_size = 3, padding = 1, stride = stride),
            nn.BatchNorm2d(num_features = c_out),
            nn.ReLU(),
            nn.Conv2d(in_channels = c_out, out_channels = c_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = c_out)
        )

        # Projection shortcut: 1x1 conv (matching the block's stride) + BN, no activation.
        # The shortcut must stay linear so it can be added to the residual branch.
        self.downsample_net = nn.Sequential(
            nn.Conv2d(in_channels = c_in, out_channels = c_out, kernel_size = 1, stride = stride),
            nn.BatchNorm2d(num_features = c_out)
        )

    def forward(self,x) :
        z = self.net(x)
        if self.downSample :
            x = self.downsample_net(x)
        out = z + x
        out = nn.functional.relu(out)
        return out

class BottleneckBlock(nn.Module):
    def __init__(self,c_in,c_bottleneck) :
        super(BottleneckBlock,self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = c_in, out_channels = c_bottleneck, kernel_size = 1,  padding = 0, stride = 1),
            nn.BatchNorm2d(num_features = c_bottleneck),
            nn.ReLU(),
            nn.Conv2d(in_channels = c_bottleneck, out_channels = c_bottleneck, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(num_features = c_bottleneck),
            nn.ReLU(),
            nn.Conv2d(in_channels = c_bottleneck, out_channels = c_in, kernel_size = 1, padding = 0, stride = 1)
        )

    def forward(self,x) :
        z = self.net(x)
        out = z + x
        out = nn.functional.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self,image_channels,num_classes):
        super(ResNet18,self).__init__()
        self.first = nn.Sequential(
            # (B X 3 X 227 X 227) -> (B X 64 X 114 X 114)
            nn.Conv2d(in_channels = image_channels, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            # (B X 64 X 114 X 114) -> (B X 64 X 57 X 57 )
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        self.layer1 = self._make_layer(in_channels = 64, out_channels = 64, stride = 1)
        self.layer2 = self._make_layer(in_channels = 64, out_channels = 128, stride = 2)
        self.layer3 = self._make_layer(in_channels = 128, out_channels = 256, stride = 2)
        self.layer4 = self._make_layer(in_channels = 256, out_channels = 512, stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features = 512, out_features = num_classes)

    def _make_layer(self,in_channels,out_channels,stride) :
        identityDownSample = True if stride != 1 else False
        return nn.Sequential(
            BasicBlock(c_in = in_channels, c_out = out_channels, downSample = identityDownSample, stride=stride),
            BasicBlock(c_in = out_channels, c_out = out_channels)
        )

    def forward(self,x):
        yhat = self.first(x)
        yhat = self.layer1(yhat)
        yhat = self.layer2(yhat)
        yhat = self.layer3(yhat)
        yhat = self.layer4(yhat)
        yhat = self.avgpool(yhat)
        yhat = yhat.reshape(yhat.shape[0],-1)
        yhat = self.fc(yhat)
        return yhat

        

def run():
    num_classes = 10
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.005
    train_loader, validation_loader = get_train_valid_loader(
        data_dir='./data', batch_size=batch_size, augment=False)
    test_loader = get_test_loader(data_dir='./data', batch_size=batch_size)
    model = ResNet18(3,num_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # no need to add a softmax layer at the output
    loss_function = nn.CrossEntropyLoss()
    callback = TrainingLogger(log_interval=10)
    mls = MLSystem(model, optimizer, loss_function, callback)
    mls.set_loaders(train_loader=train_loader, val_loader=validation_loader)
    mls.train(n_epochs=num_epochs)
    correct = validate(mls, test_loader)
    print(correct)

if __name__ == "__main__":
    run()
