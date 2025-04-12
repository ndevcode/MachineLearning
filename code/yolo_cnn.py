# -*- coding: utf-8 -*-
"""
Created on Tue Apr 08 23:20:15 2025

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
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from PIL import Image
import cv2

import xml.etree.ElementTree as ET

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################################################################################################
#  PART 0 : The ML Training system
########################################################################################################

class MLSystem():
  def __init__(self,model,optimizer,loss_function, callback, get_correct_count = None, early_stopping_callback = None):
    self.model = model
    self.optimizer = optimizer
    self.loss_function = loss_function
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model.to(self.device)
    self.get_correct_count_fn = get_correct_count or self.default_get_correct_count
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

  def default_get_correct_count(self,y_pred,y_true):
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
      predicted = self.get_correct_count_fn(yhat, y)
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
        predicted = self.get_correct_count_fn(yhat,y)
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
        #print(type(x_batch),type(y_batch))
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
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None, get_correct_count = None,
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
                                  get_correct_count = get_correct_count,
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

class VOCDataset(Dataset):
    def __init__(self, root_dir, image_set='trainval', transform=None, S=3, B=2, C=20):
        self.root_dir = root_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.class_to_idx = {
            'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
            'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
            'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
            'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
        }
        image_set_path = os.path.join(root_dir, "ImageSets", "Main", f"{image_set}.txt")
        with open(image_set_path) as f:
            self.image_ids = [line.strip() for line in f]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, 'JPEGImages', f'{image_id}.jpg')
        anno_path = os.path.join(self.root_dir, 'Annotations', f'{image_id}.xml')

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_width, original_height = image.size

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # After transform
        w = image.shape[2]
        h = image.shape[1]

        # Parse annotation
        boxes, labels = self.parse_annotation(anno_path)

        # Create target tensor
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box

            # Normalize relative to image dimensions
            x_center = ((xmin + xmax) / 2) / original_width
            y_center = ((ymin + ymax) / 2) / original_height
            box_width = (xmax - xmin) / original_width
            box_height = (ymax - ymin) / original_height

            i = int(x_center * self.S)
            j = int(y_center * self.S)

            if i >= self.S or j >= self.S:
                continue  # skip out-of-bounds boxes

            if target[j, i, 4] == 0:  # if no object already assigned
                x_cell = x_center * self.S - i
                y_cell = y_center * self.S - j

                target[j, i, 0:4] = torch.tensor([x_cell, y_cell, box_width, box_height])
                target[j, i, 4] = 1.0  # objectness
                target[j, i, 5 + label] = 1.0  # one-hot class
        #print(type(image),type(target))
        return image, target

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = obj.find("name").text.lower().strip()
            if label not in self.class_to_idx:
                continue

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[label])

        return boxes, labels

def load_data_voc(voc_root_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = VOCDataset(root_dir=voc_root_path, image_set='trainval', transform=transform)
    test_dataset = None # no training data with VOC
    return full_dataset, test_dataset

def prepare_data_voc(full_dataset, test_dataset):
    ratio = 0.8
    total = len(full_dataset)
    train_size = int(total * ratio)
    val_size = total - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Get one batch from the DataLoader
#     for batch in train_loader:
#         x_batch, y_batch = batch  # assuming each batch is a (inputs, labels) pair
#         first_image = x_batch[0]
#         print("Shape of the first image:", type(first_image),first_image.shape)
#         break  # only need the first batch


    return train_loader, val_loader, test_loader

# def collate_fn(batch):
#     images = [item[0] for item in batch]
#     targets = [item[1] for item in batch]
#     return images, targets

########################################################################################################
#  PART 2 : Visualize the data
########################################################################################################

########################################################################################################
#  PART 3 : Create the YOLO model
########################################################################################################

class YOLOBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLOBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class SimpleYOLO(nn.Module):
    def __init__(self, S = 3, B = 2, C = 20):
        super(SimpleYOLO, self).__init__()

        self.S = S 
        self.B = B
        self.C = C
        self.channels = [3, 16, 32, 64, 128, 256, 512, 1024]

        self.backbone = self.make_layers()

        out_channels = self.channels[-1]
        pred_out_channels = self.B * 5 + self.C
        self.pred = nn.Conv2d(out_channels, pred_out_channels, kernel_size=1, stride = 1, padding=0)

    def make_layers(self):
        layers = []
        for i in range(len(self.channels) - 1):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            layers.append(YOLOBlock(in_ch, out_ch))

            is_last_block = (i == len(self.channels) - 2)
            stride = 1 if is_last_block else 2
            kernel_size = 3 if is_last_block else 2
            padding = 1 if is_last_block else 0
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding = padding))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(type(x),x.shape)
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            #print(f"After layer {i}: {x.shape}")
        x = self.pred(x)
        #print(f"After pred: {x.shape}")
        x = x.permute(0, 2, 3, 1)
        return x

def get_yolo_correct_count(y_pred, y_true) :
    # Extract class logits from the last 25 positions
    class_logits = y_pred[..., 5:]         # shape: [16, 3, 3, 25]
    pred_classes = torch.argmax(class_logits, dim=-1)  # shape: [16, 3, 3]
    # Do the same for y_true if it has the same format
    true_classes = y_true[..., 5:].argmax(dim=-1)      # assuming one-hot or logits
    # Now they match in shape and can be compared
    correct = (pred_classes == true_classes).sum().item()
    return correct


########################################################################################################
#  PART 4 : Create the YOLO loss function
########################################################################################################

class YOLOLoss(nn.Module):
    def __init__(self,S=3,B=2,C=20):
        super(YOLOLoss,self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def get_bb_corners(self,bb_tensor : torch.Tensor) -> torch.Tensor :
        llx = bb_tensor[...,0] - bb_tensor[...,2] / 2
        lly = bb_tensor[...,1] - bb_tensor[...,3] / 2
        urx = bb_tensor[...,0] + bb_tensor[...,2] / 2
        ury = bb_tensor[...,1] + bb_tensor[...,3] / 2
        box = torch.stack([llx,lly,urx,ury], dim = -1)
        return box 

    def iou(self,bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> torch.Tensor:
        # Inputs: [N, S, S, B, 4] with format [llx, lly, urx, ury]
    
        llx = torch.max(bboxes1[..., 0], bboxes2[..., 0])
        lly = torch.max(bboxes1[..., 1], bboxes2[..., 1])
        urx = torch.min(bboxes1[..., 2], bboxes2[..., 2])
        ury = torch.min(bboxes1[..., 3], bboxes2[..., 3])
    
        intersection = torch.stack([llx, lly, urx, ury], dim=-1)
    
        def compute_area(boxes: torch.Tensor) -> torch.Tensor:
            return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    
        inter_area = compute_area(intersection).clamp(min=0)
        area1 = compute_area(bboxes1)
        area2 = compute_area(bboxes2)
    
        union_area = area1 + area2 - inter_area
        return inter_area / union_area.clamp(min=1e-6)

    def forward(self,Yhat,Y):
        """
        Yhat.shape = (batch,S,S,B*5+C)
        Y.shape    = (batch,S,S,B*5+C)
        """
        #print(type(Yhat),Yhat.shape)
        #print(type(Y),Y.shape)
        N = Yhat.shape[0]
        S,B,C = self.S,self.B,self.C

        exists_obj_i = Y[...,0:1]

        # Split into components
        Yhat_boxes = Yhat[..., :B*5].view(N, S, S, B, 5)
        Yhat_classes = Yhat[..., B*5:]

        Y_boxes = Y[..., :B*5].view(N, S, S, B, 5)
        Y_classes = Y[..., B*5:]

        ious = self.iou(
            self.get_bb_corners(Yhat_boxes[...,:4]),
            self.get_bb_corners(Y_boxes[...,:4])
        )

        best_box = ious.argmax(dim=-1).unsqueeze(-1)

        best_box_mask = torch.zeros_like(ious)
        best_box_mask.scatter_(-1,best_box,1)
        best_box_mask = best_box_mask.unsqueeze(-1)

        pred_box = (Yhat_boxes * best_box_mask).sum(dim=3)
        true_box = (Y_boxes * best_box_mask).sum(dim=3)

        coord_loss = self.lambda_coord * self.mse(
          exists_obj_i * pred_box[...,:2],
          exists_obj_i * true_box[...,:2]
        )
        size_loss = self.lambda_coord * self.mse(
          exists_obj_i * torch.sqrt(pred_box[...,2:4].clamp(min=1e-6)),
          exists_obj_i * torch.sqrt(true_box[...,2:4].clamp(min=1e-6))
        )
        pred_conf = (Yhat_boxes[...,4] * best_box_mask.squeeze(-1)).sum(dim=-1, keepdim = True)
        true_conf = (Y_boxes[...,4] * best_box_mask.squeeze(-1)).sum(dim=-1, keepdim = True)

        obj_loss = self.mse(exists_obj_i * pred_conf, exists_obj_i * true_conf)
        noobj_loss = self.lambda_noobj * self.mse((1 - exists_obj_i) * pred_conf, torch.zeros_like(pred_conf))

        class_loss = self.mse(exists_obj_i * Yhat_classes, exists_obj_i * Y_classes)

        total_loss = coord_loss + size_loss + obj_loss + noobj_loss + class_loss
        return total_loss

########################################################################################################
#  PART 5 : Run the training
########################################################################################################
def train_yolo_on_voc():
    voc_path = '/home/ndev/ml/datasets/VOCdevkit/VOC2012'
    grid_size = 3
    num_boxes = 2
    num_classes = 20

    # Step 1: Load and prepare the VOC data
    train_dataset, test_dataset = load_data_voc(voc_path)
    train_loader, val_loader, test_loader = prepare_data_voc(train_dataset,test_dataset)

    # Step 2: Initialize model, loss, optimizer
    model = SimpleYOLO(S=grid_size, B=num_boxes, C=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = YOLOLoss(S=grid_size, B=num_boxes, C=num_classes)

    # Step 3: Setup the training system with logging, early stopping, etc.
    runner = ExperimentRunner(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        get_correct_count = get_yolo_correct_count,
        val_loader=val_loader,
        log_interval=5,
        checkpoint_interval=2,
        patience=5,
        log_dir="runs/voc_yolo_experiment"
    )

    # Step 4: Train the model
    runner.run(num_epochs=5, seed=42)

    return runner, test_loader

########################################################################################################
#  PART 6 run inference
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

def load_image(img_path,IMG_SIZE):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image)  # shape: [3, 224, 224]
    return tensor.unsqueeze(0), image  # [1,3,224,224], and original PIL image for display

import torch

def decode_yolo_output(output, conf_threshold=0.5, image_size=(224, 224)):
    S = output.shape[0]  # grid size
    B = 2                # number of boxes per cell
    C = 20               # number of classes
    boxes = []

    cell_size_x = image_size[0] / S
    cell_size_y = image_size[1] / S

    for i in range(S):  # row
        for j in range(S):  # col
            cell = output[i, j]  # shape (30,)
            for b in range(B):
                offset = b * 5
                # Extract box info
                x = torch.sigmoid(cell[offset + 0])  # relative to cell
                y = torch.sigmoid(cell[offset + 1])
                w = torch.exp(cell[offset + 2])  # you may need anchors
                h = torch.exp(cell[offset + 3])
                conf = torch.sigmoid(cell[offset + 4])

                if conf < conf_threshold:
                    continue

                # Class probabilities
                class_probs = torch.softmax(cell[B * 5:], dim=0)
                class_id = torch.argmax(class_probs)
                class_score = class_probs[class_id].item()
                final_conf = conf.item() * class_score

                # Rescale to image
                cx = (j + x.item()) * cell_size_x
                cy = (i + y.item()) * cell_size_y
                bw = w.item() * cell_size_x  # these may need anchors!
                bh = h.item() * cell_size_y

                x1 = max(0, cx - bw / 2)
                y1 = max(0, cy - bh / 2)
                x2 = min(image_size[0], cx + bw / 2)
                y2 = min(image_size[1], cy + bh / 2)

                boxes.append([x1, y1, x2, y2, class_id.item(), final_conf])

    return boxes

def draw_boxes(image, boxes, class_names=None, output_path="output.jpg"):
    img = image.copy()
    img = np.array(image.convert("RGB"))  # Ensure RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for box in boxes:
        x1, y1, x2, y2, class_id, conf = box

        label = f"{class_names[class_id] if class_names else class_id}: {conf:.2f}"
        color = (0, 255, 0)  # Green box

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        category_name = class_names[class_id] if class_names else str(class_id)
        print(f"Detected: {category_name} ({conf:.2f})")

    # Write to disk
    cv2.imwrite(output_path, img)
    print(f"Saved image with boxes to {output_path}")

def run_inference(img_path, device = 'cuda', model_path = 'best_model.pth'):
    model = SimpleYOLO(S=3, B=2, C=20)
    model = load_model(model,model_path)
    model.eval()
    x,pil_image = load_image(img_path,224)
    x = x.to(device)
    with torch.no_grad():
        output = model(x)
    output = output[0].cpu()
    print("output shape:",output.shape)
    boxes = decode_yolo_output(output)
    draw_boxes(pil_image, boxes)

########################################################################################################
#  PART 7 : Test the model
########################################################################################################
if __name__ == "__main__":
    train_yolo_on_voc()
    run_inference("myimage.jpg")
      
