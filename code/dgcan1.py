# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 22:57:29 2024

@author: ndev
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import shutil


#DGCAN v1

def clear_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over the files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Remove files and folders
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {folder_path} does not exist.')

constants = {
  "CUDA" : False,
  "DATA_PATH" : "D:/work/deep-learning/educativePytorchIntro",
  "OUT_PATH" : "D:/work/deep-learning/educativePytorchIntro/output",
  "LOG_FILE" : os.path.join('D:/work/deep-learning/educativePytorchIntro/output','log.txt'),
  "CHECKPOINT_FILE" : os.path.join('D:/work/deep-learning/educativePytorchIntro/output','dcgan_checkpoint.pth'),
  "BATCH_SIZE" : 128,
  "IMAGE_CHANNEL" : 1,
  "Z_DIM" :  100,
  "G_HIDDEN" :  64,
  "X_DIM" :  64,
  "D_HIDDEN" :  64,
  "EPOCH_NUM" :  25,
  "REAL_LABEL" :  1,
  "FAKE_LABEL" :  0,
  "lr" :  2e-4,
  "seed" : 42 
}

constants["cuda"] = True if torch.cuda.is_available() else False
clear_folder(constants["OUT_PATH"])

def set_seed(constants):
    np.random.seed(constants["seed"])
    torch.manual_seed(constants["seed"])
    if (constants["cuda"]) :
        torch.cuda.manual_seed(constants["seed"])

class Generator(nn.Module):
    def __init__(self,constants):
        super(Generator,self).__init__()
        self.constants = constants 
        layer_parameters = [
                { "in_channels" : self.constants["Z_DIM"], "out_channels" : self.constants["G_HIDDEN"]*8, "kernel_size" : 4, "stride" : 1, "padding" : 0 },
                { "in_channels" : self.constants["G_HIDDEN"]*8, "out_channels" : self.constants["G_HIDDEN"]*4, "kernel_size" : 4, "stride" : 2, "padding" : 1 },
                { "in_channels" : self.constants["G_HIDDEN"]*4, "out_channels" : self.constants["G_HIDDEN"]*2, "kernel_size" : 4, "stride" : 2, "padding" : 1 },
                { "in_channels" : self.constants["G_HIDDEN"]*2, "out_channels" : self.constants["G_HIDDEN"], "kernel_size" : 4, "stride" : 2, "padding" : 1 }
            ]
        def get_layer(layer):
            return nn.Sequential(
                        nn.ConvTranspose2d(in_channels = layer["in_channels"],
                               out_channels = layer["out_channels"], 
                               kernel_size = layer["kernel_size"],
                               stride = layer["stride"],
                               padding = layer["padding"],
                               bias = False),
                        nn.BatchNorm2d(num_features = layer["out_channels"]),
                        nn.ReLU(True)
            )
        self.layers = nn.ModuleList([get_layer(layer) for layer in layer_parameters])
        self.layers.append(nn.Sequential(
                        nn.ConvTranspose2d(in_channels = constants["G_HIDDEN"],
                               out_channels = constants["IMAGE_CHANNEL"], 
                               kernel_size = 4,
                               stride = 2,
                               padding = 1,
                               bias = False),
                        nn.Tanh()
            ))
    def forward(self,x):
      yhat = x
      for layer in self.layers :
          yhat = layer(yhat)
      return yhat


def weights_init(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('Batchnorm') != -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)
        
G = Generator(constants=constants)
G.apply(weights_init)
x = torch.randn((1,100,28,28), dtype = torch.float)
yhat = G(x)
print(yhat.shape)

class Discriminator(nn.Module):
    def __init__(self,constants):
        super(Discriminator,self).__init__()
        self.constants = constants
        layer_parameters = [
                { "in_channels" : self.constants["IMAGE_CHANNEL"], "out_channels" : self.constants["D_HIDDEN"], "kernel_size" : 4, "stride" : 2, "padding" : 1 },
                { "in_channels" : self.constants["D_HIDDEN"], "out_channels" : self.constants["D_HIDDEN"]*2, "kernel_size" : 4, "stride" : 2, "padding" : 1 },
                { "in_channels" : self.constants["D_HIDDEN"]*2, "out_channels" : self.constants["D_HIDDEN"]*4, "kernel_size" : 4, "stride" : 2, "padding" : 1 },
                { "in_channels" : self.constants["D_HIDDEN"]*4, "out_channels" : self.constants["D_HIDDEN"]*8, "kernel_size" : 4, "stride" : 2, "padding" : 1 }
            ]
        #B X 1 X 64 X 64 -> B X 64 X 32 X 32
        #B X 64 X 32 x 32 -> B X 128 X 16 X 16
        #B X 128 X 16 X 16 -> B X 256 X 8 X 8
        #B X 256 X 8 X 8 -> B X 512 X 4 X 4
        #B X 512 X 4 X 4 -> B X 1 X 1 X 1
        def get_layer(layer):
            return nn.Sequential(
                        nn.Conv2d(in_channels = layer["in_channels"],
                               out_channels = layer["out_channels"], 
                               kernel_size = layer["kernel_size"],
                               stride = layer["stride"],
                               padding = layer["padding"],
                               bias = False),
                        nn.BatchNorm2d(num_features = layer["out_channels"]),
                        nn.LeakyReLU(0.2,inplace=True)
            ) 
        self.layers = nn.ModuleList([get_layer(layer) for layer in layer_parameters])
        self.layers.append(nn.Sequential(
                        nn.Conv2d(in_channels = constants["D_HIDDEN"]*8,
                              out_channels = 1, 
                              kernel_size = 4,
                              stride = 1,
                              padding = 0,
                              bias = False),
                       nn.Sigmoid()
           ))
        
    def forward(self,x):
        yhat = x
        for layer in self.layers :
            yhat = layer(yhat)
        return yhat.view(-1,1).squeeze(1)

D = Discriminator(constants=constants)
D.apply(weights_init)
z = torch.randn((1,1,496,496), dtype = torch.float)
yhat = D(z)
print(yhat.shape)

loss_function = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=constants["lr"], betas=(0.5, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=constants["lr"], betas=(0.5, 0.999))

dataset = datasets.MNIST(root=constants["DATA_PATH"], download=True,
                     transform=transforms.Compose([
                     transforms.Resize(constants["X_DIM"]),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))
                     ]))
assert dataset
if __name__ == "__main__":
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=constants["BATCH_SIZE"],
                                         shuffle=True, num_workers=0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Reached here")
for epoch in range(constants["EPOCH_NUM"]):
    for i, data in enumerate(dataloader):
        #print(data[0].shape)
        x_real = data[0].to(device)
        real_label = torch.full((x_real.size(0),), constants["REAL_LABEL"], device=device).float()
        fake_label = torch.full((x_real.size(0),), constants["FAKE_LABEL"], device=device).float()
        # Update D with real data
        D.zero_grad()
        y_real = D(x_real)
        #print(y_real,real_label)
        loss_D_real = loss_function(y_real, real_label)
        loss_D_real.backward()
        # Update D with fake data
        z_noise = torch.randn(x_real.size(0), constants["Z_DIM"], 1, 1, device=device)
        x_fake = G(z_noise)
        y_fake = D(x_fake.detach())
        loss_D_fake = loss_function(y_fake, fake_label)
        loss_D_fake.backward()
        optimizerD.step()
        # Update G with fake data
        G.zero_grad()
        y_fake_r = D(x_fake)
        loss_G = loss_function(y_fake_r, real_label)
        loss_G.backward()
        optimizerG.step()
        if i % 100 == 0:
            print('Epoch {} [{}/{}] loss_D_real: {:.4f} loss_D_fake:{:.4f} loss_G: {:.4f}'.format(
                epoch, i, len(dataloader),
                loss_D_real.mean().item(),
                loss_D_fake.mean().item(),
                loss_G.mean().item()
            ))

torch.save({
    'generator_state_dict': G.state_dict(),
    'discriminator_state_dict': D.state_dict(),
    'optimizer_G_state_dict': optimizerG.state_dict(),
    'optimizer_D_state_dict': optimizerD.state_dict(),
}, constants["CHECKPOINT_FILE"])

#Generate an image
viz_noise = torch.randn(constants["BATCH_SIZE"], constants["Z_DIM"], 1, 1, device=device).float()
with torch.no_grad():
    viz_sample = G(viz_noise)
    utils.save_image(viz_sample, os.path.join(constants["OUT_PATH"],'fake_samples_{}.png'.format(epoch)), normalize=True)
